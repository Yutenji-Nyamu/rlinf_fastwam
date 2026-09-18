"""CPU checks for opt-in RLT chunk dropout and runner-round annealing."""
import pytest
import torch

from rlinf.algorithms.rlt.dvac_controls import (
    apply_chunk_dropout, effective_alphas, rlt_controls_contract,
)
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from test_rlt_dvac_two_level_worker import (
    worker_factory, _batch_and_expected_weights, _actor_loss,
)

def control(**kwargs):
    return rlt_controls_contract(dict(mode="apply", mapping="two_level_batch",
        factor_mapping="exp_mean", **kwargs))

def schedule():
    return dict(enabled=True, local=dict(start_step=1, end_step=500, end_alpha=0.),
                chunk=dict(start_step=1, end_step=500, end_alpha=0.))

@pytest.mark.parametrize("p", [0., .2, 1.])
def test_chunk_dropout_probability_rng_and_failure_preservation(p):
    weights = torch.full((10000, 10), 2.)
    eligible = torch.arange(10000) % 2 == 0
    cfg = control(chunk_dropout=dict(enabled=True, probability=p, seed=42))
    rng = torch.get_rng_state().clone()
    result, dropped = apply_chunk_dropout(weights, eligible, cfg, update_step=100)
    again, same = apply_chunk_dropout(weights, eligible, cfg, update_step=100)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(result, again) and torch.equal(dropped, same)
    assert torch.equal(result[~eligible], weights[~eligible])
    assert torch.equal(result[dropped], torch.ones_like(result[dropped]))
    assert torch.equal(result[~dropped], weights[~dropped])
    assert abs(float(dropped.sum()) / int(eligible.sum()) - p) < .02
    if p == .2:
        _, other = apply_chunk_dropout(weights, eligible, cfg, update_step=102)
        assert not torch.equal(other, dropped)

def test_schedule_exact_endpoints_and_independent_layers():
    cfg = control(alpha_schedule=schedule())
    assert effective_alphas(1., 1., cfg, runner_step=0) == (1., 1.)
    assert effective_alphas(1., 1., cfg, runner_step=499) == (0., 0.)
    assert effective_alphas(1., 1., cfg, runner_step=799) == (0., 0.)
    assert effective_alphas(1., 1., cfg, runner_step=249) == pytest.approx((250/499, 250/499))
    cfg['alpha_schedule'].pop('chunk')
    assert effective_alphas(1., .8, cfg, runner_step=499) == (0., .8)

@pytest.mark.parametrize("bad", [-.1, 1.1, float('nan'), True])
def test_invalid_probability_rejected(bad):
    with pytest.raises(ValueError):
        control(chunk_dropout=dict(enabled=True, probability=bad))

def test_disabled_exact_legacy_equivalence(worker_factory):
    batch, _ = _batch_and_expected_weights()
    plain = worker_factory(factor_mapping='exp_mean', temperature_local=2.5, temperature_chunk=2.5)
    disabled = worker_factory(factor_mapping='exp_mean', temperature_local=2.5, temperature_chunk=2.5,
        chunk_dropout=dict(enabled=False), alpha_schedule=dict(enabled=False))
    a, am = plain._prepare_global_batch(batch, train_actor=True)
    b, bm = disabled._prepare_global_batch(batch, train_actor=True)
    assert torch.equal(a['rlt_dvac_new_weights'], b['rlt_dvac_new_weights'])
    assert am == bm and disabled.rlt_dvac_controls == {}

@pytest.mark.parametrize('kind', ['dropout', 'anneal'])
def test_worker_clean_endpoint_and_microbatch_gradient(worker_factory, kind):
    opts = dict(chunk_dropout=dict(enabled=True, probability=1.)) if kind == 'dropout' else dict(alpha_schedule=schedule())
    batch, _ = _batch_and_expected_weights()
    results = []
    for splits in (1, 2, 4):
        worker = worker_factory(factor_mapping='exp_mean', temperature_local=2.5, temperature_chunk=2.5, **opts)
        worker.version = 499
        worker.update_step = 123456  # Must not be used as the annealing round.
        prepared, metrics = worker._prepare_global_batch(batch, train_actor=True)
        assert torch.equal(prepared['rlt_dvac_new_weights'], torch.ones(4, 10))
        assert metrics['rlt_dvac_new/applied_mean'] == 1.
        assert metrics['rlt_dvac_controls/runner_round'] == 500.
        for micro in split_dict_to_chunk(prepared, splits):
            loss, _, _ = _actor_loss(worker, micro)
            (loss / splits).backward()
        results.append(worker.model.actor_actions.grad.clone())
        pi = worker.model.actor_actions
        clean = 2.5 * ((pi.reshape(1,10,2) - batch['curr_obs']['ref_chunk'])**2).mean() - .45 * pi.mean() * worker.model.critic_bias.detach()
        expected, = torch.autograd.grad(clean, pi)
        torch.testing.assert_close(results[-1], expected)
    for grad in results[1:]:
        torch.testing.assert_close(grad, results[0])

def test_resume_preserves_dropout_counter_and_rejects_changed_controls(worker_factory):
    opts = dict(chunk_dropout=dict(enabled=True, probability=.2))
    worker = worker_factory(**opts)
    worker.update_step = 400
    state = worker._rlt_state_payload(runner_step=14)
    resumed = worker_factory(**opts)
    resumed._validate_rlt_state(state)
    resumed._restore_rlt_state(state)
    assert resumed.update_step == 400
    weights = torch.full((512,10), 2.)
    mask = torch.ones(512, dtype=torch.bool)
    a = apply_chunk_dropout(weights, mask, worker.rlt_dvac_controls, update_step=worker.update_step)[0]
    b = apply_chunk_dropout(weights, mask, resumed.rlt_dvac_controls, update_step=resumed.update_step)[0]
    assert torch.equal(a,b)
    changed = worker_factory(chunk_dropout=dict(enabled=True, probability=.1))
    with pytest.raises(ValueError, match='rlt_resume_contract'):
        changed._validate_rlt_state(state)
