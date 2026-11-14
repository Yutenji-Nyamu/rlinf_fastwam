"""Apply the scoped RLT critic/gradient repair to a prepared checkout."""
import argparse, difflib
from pathlib import Path

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one matching code block, got {text.count(old)}: {old[:70]}')
    return text.replace(old, new, 1)

def transform(rel, text):
    if rel.endswith('q_head.py'):
        text=replace_once(text, '        q_values = self.net(x)\n', '''        if getattr(self, "output_feature_normalization", False):
            # OFN: normalize the final critic feature, before the scalar head.
            # Keep Sequential indices/state_dict keys compatible with checkpoints.
            for layer in list(self.net.children())[:-1]:
                x = layer(x)
            x = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-8)
            q_values = self.net[-1](x)
        else:
            q_values = self.net(x)
''')
    elif rel.endswith('fsdp_sac_policy_worker.py'):
        text=replace_once(text, '        # Enable gradient checkpointing if configured\n', '''        if self.cfg.algorithm.get("q_output_feature_normalization", False):
            if self.cfg.actor.model.get("q_head_type", "default") != "default":
                raise ValueError("Q output feature normalization requires default QHead")
            for policy in ([module, target_module] if initialize_target else [module]):
                for head in policy.q_head.qs:
                    head.output_feature_normalization = True

        # Enable gradient checkpointing if configured
''')
        anchor='    @Worker.timer("update_one_epoch")\n'
        text=replace_once(text, anchor, '''    def _clear_grads_outside_optimizer(self, optimizer):
        """Restrict FSDP's global norm to the optimizer being stepped.

        Actor backprop still traverses Q to obtain dQ/da. Discard only parameter
        gradients owned by the other optimizer, after backward has completed.
        """
        owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for parameter in self.model.parameters():
            if id(parameter) not in owned:
                parameter.grad = None

'''+anchor)
        text=replace_once(text, '        qf_grad_norm = self.model.clip_grad_norm_(\n', '''        self._clear_grads_outside_optimizer(self.qf_optimizer)
        qf_grad_norm = self.model.clip_grad_norm_(
''')
        text=replace_once(text, '            actor_grad_norm = self.model.clip_grad_norm_(\n', '''            self._clear_grads_outside_optimizer(self.optimizer)
            actor_grad_norm = self.model.clip_grad_norm_(
''')
    elif rel.endswith('fsdp_rlt_ac_policy_worker.py'):
        text=replace_once(text, '''        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {"q_data": self._metric(all_data_q_values.mean())}
''', '''        target_min = self.cfg.algorithm.get("critic_target_min", None)
        target_max = self.cfg.algorithm.get("critic_target_max", None)
        raw_target = target_q_values
        if target_min is not None or target_max is not None:
            target_q_values = target_q_values.clamp(min=target_min, max=target_max)
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {
            "q_data": self._metric(all_data_q_values.mean()),
            "q_target": self._metric(target_q_values.mean()),
            "q_target_raw": self._metric(raw_target.mean()),
            "q_target_clipped_fraction": self._metric(
                (target_q_values != raw_target).float().mean()
            ),
        }
''')
    return text

FILES=['rlinf/models/embodiment/modules/q_head.py',
       'rlinf/workers/actor/fsdp_sac_policy_worker.py',
       'rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py']
def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--output-root');p.add_argument('--in-place',action='store_true');p.add_argument('--patch-output');a=p.parse_args()
    src=Path(a.source_root);dst=src if a.in_place else Path(a.output_root)
    edits=[]
    for rel in FILES:
        old=(src/rel).read_text(encoding='utf-8');new=transform(rel,old)
        compile(new,rel,'exec')
        edits.append((rel,old,new))
    diffs=[]
    for rel,old,new in edits:
        f=dst/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(new,encoding='utf-8')
        diffs.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    if a.patch_output:Path(a.patch_output).write_text(''.join(diffs),encoding='utf-8')
    print('Patched and syntax-checked:',*[x[0] for x in edits],sep='\n')
if __name__=='__main__':main()
