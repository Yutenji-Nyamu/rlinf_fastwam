"""Small opt-in delta over the accepted ff64209 + full-noise DSRL export.

The rollout U definition and SAC objective remain the October pi05 version.
Schema 4 distinguishes full-noise + U from old repeated-noise schema 2.
"""
import ast
from pathlib import Path
from tools.dojo_offpolicy.full_noise import replace

def apply(root):
    root=Path(root)
    path=root/'rlinf/models/embodiment/openpi/openpi_action_model.py'
    text=path.read_text()
    text=replace(text,
        '            if self.config.dsrl_action_noise_dim != self.config.action_dim:\n                raise ValueError("DSRL U requires the complete model-space latent")',
        '            expected_latent = self.config.action_dim * (self.config.action_horizon if self.config.dsrl_full_noise else 1)\n            if self.config.dsrl_action_noise_dim != expected_latent:\n                raise ValueError("DSRL U latent dimension does not match its noise mode")')
    ast.parse(text);path.write_text(text)
    path=root/'rlinf/data/storage/replay/dsrl_transition.py';text=path.read_text()
    text=replace(text,'            if self.schema_version != 2:\n                raise ValueError("DSRL U replay requires schema_version=2")\n        elif self.schema_version == 2:\n            raise ValueError("DSRL replay schema_version=2 requires a U signal spec")',
        '            if self.schema_version not in (2, 4):\n                raise ValueError("DSRL U replay requires repeated schema 2 or full schema 4")\n        elif self.schema_version in (2, 4):\n            raise ValueError("DSRL replay schema 2/4 requires a U signal spec")')
    ast.parse(text);path.write_text(text)
    path=root/'rlinf/workers/actor/fsdp_sac_policy_worker.py';text=path.read_text()
    text=replace(text,'        if replay_cfg.get("schema_version", 1) != 2:\n            raise ValueError("DSRL U requires replay schema_version=2")',
        '        expected_schema = 4 if model_cfg.get("dsrl_full_noise", False) else 2\n        if replay_cfg.get("schema_version", 1) != expected_schema:\n            raise ValueError(f"DSRL U replay requires schema_version={expected_schema}")')
    ast.parse(text);path.write_text(text)

if __name__=='__main__':
    import sys
    apply(sys.argv[1])
