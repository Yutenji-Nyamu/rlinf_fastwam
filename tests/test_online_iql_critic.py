"""Numerical, isolation, restore, and official-encoder parity checks.

Run on the project server, with JAX_PLATFORMS=cpu for the reference comparison.
No actor checkpoint, scorer, simulator, or GPU is needed for these checks.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from rlinf.algorithms.online_iql import advantage_weights, expectile_loss, iql_phase, shape_reward
from rlinf.models.embodiment import online_iql_critic as module


@pytest.mark.parametrize("number,expected", [(1, "bc"), (10, "bc"), (11, "iql"), (100, "iql")])
def test_phase(number, expected):
    assert iql_phase(number) == expected
    assert iql_phase(number, 0) == "iql"


@pytest.mark.parametrize("number", [0, -1, 1.5, True])
def test_phase_rejects_ambiguous_rounds(number):
    with pytest.raises(ValueError):
        iql_phase(number)


def test_reward_preserves_raw_terminal_prediction_and_absorbs():
    p = torch.tensor([-8.0, -2.0, -4.0], requires_grad=True)
    pn = torch.tensor([-5.0, -0.5, -3.0], requires_grad=True)
    r = shape_reward(p, pn, torch.tensor([0., 1., 0.]), torch.tensor([1., 0., 0.]))
    torch.testing.assert_close(r, torch.tensor([0.305, 1.2, 0.4]))
    assert not r.requires_grad
    torch.testing.assert_close(pn.detach(), torch.tensor([-5., -.5, -3.]))
    torch.testing.assert_close(shape_reward(p, pn, 0., 1., potential_scale=0.), torch.zeros(3))


def test_reward_telescopes_with_success_and_failure():
    gamma, scale = .99, .1
    potentials = [-9., -6., -4.]
    for success in [0., 1.]:
        r = [float(shape_reward(potentials[i], potentials[i+1] if i < 2 else -3.,
                                success if i == 2 else 0., float(i < 2))) for i in range(3)]
        ret = sum(gamma ** i * r[i] for i in range(3))
        assert ret == pytest.approx(gamma ** 2 * success - scale * potentials[0], abs=2e-7)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_finite_validation_before_reward_and_weight_clip(bad):
    with pytest.raises(ValueError):
        shape_reward(-1., bad, 0., 0.)
    with pytest.raises(ValueError):
        advantage_weights(torch.tensor([bad]))


def test_expectile_sign_and_weight_formula():
    d = torch.tensor([-2., 0., 2.], requires_grad=True)
    loss = expectile_loss(d)
    torch.testing.assert_close(loss, torch.tensor([.8, 0., 3.2]))
    loss.sum().backward()
    torch.testing.assert_close(d.grad, torch.tensor([-.8, 0., 3.2]))
    a = torch.tensor([-1., 0., .1, .5, 3.4e38], requires_grad=True)
    w = advantage_weights(a)
    torch.testing.assert_close(w[:4], torch.tensor([math.exp(-10), 1., math.e, 100.]))
    assert torch.isfinite(w).all() and w[-1] == 100 and not w.requires_grad
    assert w.mean() != 1


def test_crop_matches_numpy_edge_reference_and_preserves_cameras():
    pixels = torch.arange(2 * 9 * 7 * 11).remainder(251).to(torch.uint8).reshape(2, 9, 7, 11)
    offsets = torch.tensor([[0, 8], [7, 2]])
    expected = []
    for image, (dy, dx) in zip(pixels.numpy(), offsets.numpy()):
        padded = np.pad(image, ((0, 0), (4, 4), (4, 4)), mode="edge")
        expected.append(padded[:, dy:dy + 7, dx:dx + 11])
    generator = torch.Generator().manual_seed(456)
    before = generator.get_state().clone()
    actual = module.random_crop(pixels, generator, offsets)
    np.testing.assert_array_equal(actual.numpy(), np.stack(expected))
    assert torch.equal(before, generator.get_state())
    a = module.random_crop(pixels, generator)
    assert a.shape == pixels.shape and a.dtype == torch.uint8


def test_same_padding_and_spatial_xy_order():
    x = torch.arange(16.).reshape(1, 1, 4, 4)
    p = module.same_padding(x, 3, 2)
    assert p.shape == (1, 1, 5, 5)  # trailing padding only for this stride/shape
    torch.testing.assert_close(p[..., :4, :4], x)
    spatial = module.SpatialSoftmax(2, 2)
    features = torch.tensor([[[[0., 0.], [0., 20.]], [[20., 0.], [0., 0.]]]])
    result = spatial(features)
    torch.testing.assert_close(result, torch.tensor([[1., -1., 1., -1.]]), atol=1e-6, rtol=0)


def test_stack_pixels_uses_official_resize_and_uint8():
    from openpi_client import image_tools
    main = torch.arange(2 * 120 * 180 * 3).remainder(256).to(torch.uint8).reshape(2, 120, 180, 3)
    wrist = torch.stack((main.flip(1), main.flip(2)), dim=1)
    state = torch.get_rng_state().clone()
    result = module.stack_pixels(main, wrist)
    refs = [image_tools.resize_with_pad(x.numpy(), 224, 224) for x in (main, wrist[:, 0], wrist[:, 1])]
    expected = np.concatenate(refs, axis=-1).transpose(0, 3, 1, 2)
    np.testing.assert_array_equal(result.numpy(), expected)
    assert torch.equal(torch.get_rng_state(), state)
    assert result.dtype == torch.uint8 and result.shape == (2, 9, 224, 224)
    with pytest.raises(ValueError):
        module.stack_pixels(main.float(), wrist)


class TinyQ(nn.Module):
    def __init__(self):
        super().__init__()
        # A realistic constructor consumes RNG even before explicit init. The
        # learner must isolate that consumption from the actor's stream.
        self.weight = nn.Parameter(torch.rand(2) * 0 + torch.tensor([.8, 1.2]))
        self.bias = nn.Parameter(torch.tensor([.1, -.1]))

    def forward(self, pixels, actions):
        x = pixels.float().mean((1, 2, 3)) / 255 + actions[:, 0, 0]
        return self.weight[:, None] * x[None] + self.bias[:, None]


class TinyV(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(()) * 0 + .25)

    def forward(self, pixels):
        return self.weight * pixels.float().mean((1, 2, 3)) / 255


@pytest.fixture
def learner_factory(monkeypatch):
    # Isolate algorithm/order tests from costly convolutional backprop.  Actual
    # production encoder is independently compared to official Flax below.
    monkeypatch.setattr(module, "PixelQ", TinyQ)
    monkeypatch.setattr(module, "PixelV", TinyV)
    return lambda **kwargs: module.PixelIQLLearner({"seed": 74, **kwargs}, "cpu")


def _batch():
    p = torch.full((3, 9, 224, 224), 255, dtype=torch.uint8)
    pn = torch.full_like(p, 128)
    actions = torch.zeros(3, 50, 14)
    actions[:, 0, 0] = torch.tensor([.3, -.2, .1])
    return p, actions, pn, torch.tensor([0., 1., .2]), torch.tensor([1., 0., 1.])


def test_exact_v_q_target_order_and_head_mean(learner_factory):
    learner = learner_factory()
    p, actions, pn, rewards, masks = _batch()
    # Independent reference using the exact declared Adam and algebra.
    q, v, target = copy.deepcopy(learner.critic), copy.deepcopy(learner.value), copy.deepcopy(learner.target_critic)
    vo = torch.optim.Adam(v.parameters(), lr=3e-4, betas=(.9, .999), eps=1e-8)
    qo = torch.optim.Adam(q.parameters(), lr=3e-4, betas=(.9, .999), eps=1e-8)
    with torch.no_grad():
        qt = target(p, actions).min(0).values
    diff = qt - v(p)
    lv = (torch.where(diff > 0, .8, .2) * diff.square()).mean()
    lv.backward(); vo.step()
    with torch.no_grad():
        y = rewards + .99 * masks * v(pn)
    lq = (q(p, actions) - y).square().sum() / (2 * len(p))
    lq.backward(); qo.step()
    before = torch.get_rng_state().clone()
    metrics = learner.update(p, actions.requires_grad_(), pn, rewards, masks)
    assert torch.equal(before, torch.get_rng_state())
    assert actions.grad is None
    assert metrics["value_loss"] == pytest.approx(float(lv.detach()), rel=1e-6)
    assert metrics["critic_loss"] == pytest.approx(float(lq.detach()), rel=1e-6)
    for actual, expected in zip(learner.value.parameters(), v.parameters()):
        torch.testing.assert_close(actual, expected)
    for actual, expected, target_old, target_new in zip(learner.critic.parameters(), q.parameters(), target.parameters(), learner.target_critic.parameters()):
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(target_new, .995 * target_old + .005 * expected)
    assert all(p.grad is None for p in learner.target_critic.parameters())
    assert all(p.grad is None for p in learner.value.parameters())


def test_advantages_current_networks_batch_splits_rng_and_detach(learner_factory):
    learner = learner_factory()
    p, a, *_ = _batch()
    with torch.no_grad():
        learner.critic.bias.add_(.3)
        learner.target_critic.bias.sub_(20.)
    before_global = torch.get_rng_state().clone()
    before_crop = learner.crop_generator.get_state().clone()
    actual = learner.advantages(p, a, batch_size=1)
    with torch.no_grad():
        expected = learner.critic(p, a).min(0).values - learner.value(p)
    torch.testing.assert_close(actual["advantage"], expected)
    torch.testing.assert_close(actual["weights"], learner.advantages(p, a, batch_size=3)["weights"])
    assert torch.equal(before_global, torch.get_rng_state())
    assert torch.equal(before_crop, learner.crop_generator.get_state())
    assert all(not t.requires_grad for t in actual.values())


def test_checkpoint_restores_models_adam_target_rng_and_counter(learner_factory):
    before = torch.get_rng_state().clone()
    first = learner_factory()
    assert torch.equal(before, torch.get_rng_state())
    first.update(*_batch())
    snapshot = first.state_dict()
    second = learner_factory()
    second.load_state_dict(snapshot)
    assert second.update_count == 1
    first_metrics, second_metrics = first.update(*_batch()), second.update(*_batch())
    assert first_metrics == second_metrics
    assert torch.equal(first.crop_generator.get_state(), second.crop_generator.get_state())
    for key in first.critic.state_dict():
        torch.testing.assert_close(first.critic.state_dict()[key], second.critic.state_dict()[key], rtol=0, atol=0)
    incompatible = learner_factory(beta=1.)
    with pytest.raises(ValueError, match="identity"):
        incompatible.load_state_dict(snapshot)


def test_reject_model_padding_and_nonfinite_without_hidden_fallback(learner_factory):
    learner = learner_factory()
    p, actions, pn, r, mask = _batch()
    with pytest.raises(ValueError, match="50,14"):
        learner.update(p, torch.zeros(3, 50, 32), pn, r, mask)
    actions[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        learner.advantages(p, actions)
    assert learner.update_count == 0


# SHA-locked official sources already deployed on the test server. No upstream
# repository is included in the method branch. Text hashes normalize LF only.
UPSTREAM_SOURCE_MANIFEST = '{"jaxrl2.networks.constants":{"path":"networks/constants.py","sha256":"9b092d7046191af2cd25c7e49f93cc32f4757675027419d63ef45027d13d6970"},"jaxrl2.networks.encoders.spatial_softmax":{"path":"networks/encoders/spatial_softmax.py","sha256":"3ca9994d4aa6928114c7da9f71011481e63fae1a96694086d012f8086d426d75"},"jaxrl2.networks.encoders.resnet_encoderv1":{"path":"networks/encoders/resnet_encoderv1.py","sha256":"c3ac2fcc35b905bbb1dca096f015dc3dd6f244e21e48f656378996166c1a04a2"},"jaxrl2.networks.encoders.networks":{"path":"networks/encoders/networks.py","sha256":"6921af904545058d8b0851868faf78ee1fa4cbe52abda564da08c506dbb38a9d"}}'


def _official_encoder(monkeypatch):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    jax = pytest.importorskip("jax")
    flax = pytest.importorskip("flax.linen")
    payload = json.loads(UPSTREAM_SOURCE_MANIFEST)
    source_root = Path(os.environ.get("ONLINE_IQL_UPSTREAM",
        "/data/chenyiteng/projects/RynnValue-10e0d333/pi-rl/third_party/jaxrl2/jaxrl2"))
    for name in ("jaxrl2", "jaxrl2.networks", "jaxrl2.networks.encoders"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    # CrossNorm is only imported by the official encoder; norm='group' never
    # calls it. Keep this unused dependency explicit rather than editing source.
    cross = types.ModuleType("jaxrl2.networks.encoders.cross_norm")
    class UnusedCrossNorm:
        def __init__(self, *args, **kwargs):
            raise AssertionError("The locked IQL config must not use CrossNorm")
    cross.CrossNorm = UnusedCrossNorm
    monkeypatch.setitem(sys.modules, cross.__name__, cross)
    loaded = {}
    for name, fixture in payload.items():
        source = source_root / fixture["path"]
        assert source.is_file(), f"Set ONLINE_IQL_UPSTREAM to locked jaxrl2 source: missing {source}"
        text = source.read_text(encoding="utf-8").rstrip("\n") + "\n"
        assert hashlib.sha256(text.encode()).hexdigest() == fixture["sha256"]
        target = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, target)
        exec(compile(text, f"upstream@10e0d333/{name}", "exec"), target.__dict__)
        loaded[name] = target
    class Identity(flax.Module):
        @flax.compact
        def __call__(self, obs, actions=None, training=False):
            return obs["pixels"]
    encoder = loaded["jaxrl2.networks.encoders.resnet_encoderv1"].ResNet18(
        norm="group", use_spatial_softmax=True, softmax_temperature=1.)
    ref = loaded["jaxrl2.networks.encoders.networks"].PixelMultiplexer(
        encoder=encoder, network=Identity(), latent_dim=50, use_bottleneck=True)
    return jax, ref


def test_actual_encoder_against_locked_official_flax_forward(monkeypatch):
    jax, ref = _official_encoder(monkeypatch)
    # Independent initialization equality is not expected across backends. Map
    # exactly the production Torch parameters into the actual upstream module.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(913)
        encoder = module.PixelEncoder().eval()
    pixels = torch.arange(9 * 224 * 224).remainder(251).to(torch.uint8).reshape(1, 9, 224, 224)
    images = pixels.permute(0, 2, 3, 1).numpy()[..., None]
    with jax.default_device(jax.devices("cpu")[0]):
        variables = ref.init(jax.random.PRNGKey(47), {"pixels": images})
        from flax.core import unfreeze, freeze
        params = unfreeze(variables)
        enc = params["params"]["encoder"]
        def conv(dst, src):
            dst["kernel"] = src.weight.detach().numpy().transpose(2, 3, 1, 0)
        def norm(dst, src):
            dst["scale"] = src.weight.detach().numpy()
            dst["bias"] = src.bias.detach().numpy()
        conv(enc["conv_init"], encoder.stem)
        norm(enc["bn_init"], encoder.stem_norm)
        for i, block in enumerate(encoder.blocks):
            target = enc[f"ResNetBlock_{i}"]
            conv(target["Conv_0"], block.conv1)
            conv(target["Conv_1"], block.conv2)
            norm(target["MyGroupNorm_0"], block.norm1)
            norm(target["MyGroupNorm_1"], block.norm2)
            if block.projection is not None:
                conv(target["conv_proj"], block.projection)
                norm(target["norm_proj"], block.projection_norm)
        params["params"]["Dense_0"]["kernel"] = encoder.bottleneck.weight.detach().numpy().T
        params["params"]["Dense_0"]["bias"] = encoder.bottleneck.bias.detach().numpy()
        norm(params["params"]["LayerNorm_0"], encoder.bottleneck_norm)
        expected = np.asarray(ref.apply(freeze(params), {"pixels": images}))
    with torch.no_grad():
        actual = encoder(pixels).numpy()
    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)


def test_q_and_v_head_contract_without_model_padding():
    with torch.random.fork_rng(devices=[]):
        head = module.ValueMLP(750, True)
        value_head = module.ValueMLP(50, False)
    assert head.layers[0].in_features == 50 * 14 + 50
    assert len(head.norms) == 2 and value_head.norms is None
    # Independent NumPy Dense -> LN -> ReLU check, including LN epsilon.
    x = np.linspace(-1, 1, 750, dtype=np.float32)[None]
    expected = x.copy()
    for i, layer in enumerate(head.layers):
        expected = expected @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()
        if i < 2:
            expected = (expected - expected.mean(-1, keepdims=True)) / np.sqrt(expected.var(-1, keepdims=True) + 1e-6)
            expected = np.maximum(expected, 0)
    with torch.no_grad():
        np.testing.assert_allclose(head(torch.from_numpy(x)).numpy(), expected[:, 0], atol=2e-5, rtol=2e-5)
