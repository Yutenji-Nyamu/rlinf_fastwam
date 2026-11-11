"""Apply the reviewed optional critic loss to a private source export."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path("/data/chenyiteng/projects/rlinf-shenzhen/exports/cql-pour-20261111/dsrl")
p = root / "rlinf/workers/actor/fsdp_sac_policy_worker.py"
old = p.read_text()
s = old
needle = (
    '        return critic_loss, {"q_data": self._metric(all_data_q_values.mean())}'
)
new = """        metrics = {"q_data": self._metric(all_data_q_values.mean())}
        cql_weight = float(self.cfg.algorithm.get("dsrl_cql_weight", 0.0))
        if cql_weight:
            if not (
                self.use_dsrl_flat_replay
                and use_dsrl
                and not use_crossq
                and not getattr(self, "use_dsrl_u", False)
                and 0 < cql_weight < float("inf")
            ):
                raise ValueError("Conservative Q requires clean flat-replay DSRL and a positive finite weight")
            from rlinf.algorithms.conservative_q import conservative_q_loss

            with torch.no_grad():
                if self._dsrl_critic_warmup_active:
                    candidate_actions = torch.randn_like(actions, dtype=torch.float32)
                else:
                    candidate_actions, _, _ = self.model(
                        forward_type=ForwardType.SAC, obs=curr_obs, train=True
                    )
            candidate_q = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=candidate_actions.detach(),
                train=True,
            )
            penalty = cql_weight * conservative_q_loss(
                all_data_q_values,
                candidate_q,
                float(self.cfg.algorithm.get("dsrl_cql_temperature", 1.0)),
            )
            metrics.update(
                td_loss=self._metric(critic_loss),
                cql_loss=self._metric(penalty),
                cql_q_gap=self._metric((candidate_q - all_data_q_values).mean()),
                cql_policy_q=self._metric(candidate_q.mean()),
                target_q_mean=self._metric(target_q_values.mean()),
                target_q_max=self._metric(target_q_values.max()),
            )
            critic_loss = critic_loss + penalty
        return critic_loss, metrics"""
assert s.count(needle) == 1
s = s.replace(needle, new)
p.write_text(s)
shutil.copyfile(
    Path(__file__).with_name("conservative_q.py"),
    root / "rlinf/algorithms/conservative_q.py",
)
print(
    json.dumps(
        {
            "file": str(p),
            "before": hashlib.sha256(old.encode()).hexdigest(),
            "after": hashlib.sha256(s.encode()).hexdigest(),
        }
    )
)
