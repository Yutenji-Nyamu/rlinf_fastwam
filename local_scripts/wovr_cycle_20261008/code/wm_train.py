"""OpenDW official losses/trainer, with the live 14D/z-score/resize interface."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Dataset


class NativeChunks(Dataset):
    def __init__(self,root,split,stats,contexts):
        self.root=Path(root); self.stats=stats; self.contexts=contexts
        self.rows=[r for r in json.loads((self.root/'manifest.json').read_text())['rows'] if r['split']==split]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        from dexbotic.policy.dw05_policy import _zscore_normalize,ROBOTWIN_PROMPT_FORMAT
        r=self.rows[index]
        with np.load(self.root/r['file'],allow_pickle=False) as z:
            video=torch.from_numpy(z['video'].copy()).permute(3,0,1,2).float()*(2/255)-1
            action=torch.from_numpy(_zscore_normalize(z['action'],self.stats['action']))
            state=torch.from_numpy(_zscore_normalize(z['state'],self.stats['state']))
            pad=torch.from_numpy(z['image_is_pad'].copy())
        prompt=ROBOTWIN_PROMPT_FORMAT.format(task=r['prompt']);context,mask=self.contexts[prompt]
        return dict(video=video,action=action,proprio=state,prompt=prompt,context=context,context_mask=mask,
            image_is_pad=pad,action_is_pad=torch.zeros(32,dtype=torch.bool),
            proprio_is_pad=torch.zeros(1,dtype=torch.bool),action_dim_mask=torch.ones(14,dtype=torch.bool),
            has_action=torch.tensor(r['success'],dtype=torch.bool))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--steps',type=int);p.add_argument('--accum',type=int,default=1)
    p.add_argument('--lr',type=float,default=1e-5);a=p.parse_args()
    os.environ['DIFFSYNTH_MODEL_BASE_PATH']=a.bundle
    from accelerate import Accelerator,DistributedDataParallelKwargs
    from dexbotic.model.dw05 import DW05ModelConfig
    from dexbotic.exp.dw05_trainer import DW05Trainer
    from dexbotic.exp.base_dw_exp import DWTrainerConfig
    from dexbotic.policy.dw05_policy import _load_norm_stats,ROBOTWIN_PROMPT_FORMAT
    from torch.utils.data._utils.collate import default_collate
    acc=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=a.accum,
        kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=True,find_unused_parameters=True)])
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    recipe_path=output.parents[2]/'wm_training.json'
    fallback=bool(a.steps and recipe_path.exists())
    if recipe_path.exists():
        a.batch=int(json.loads(recipe_path.read_text())['microbatch'])
    torch.cuda.set_device(acc.device)
    cfg=DW05ModelConfig(load_text_encoder=True,skip_dit_load_from_pretrain=True,
        action_dim=14,proprio_dim=14,mot_checkpoint_mixed_attn=True)
    model=cfg.build_model(model_dtype=torch.bfloat16,device=str(acc.device))
    payload=model.load_checkpoint(a.checkpoint);del payload
    rows=json.loads((Path(a.dataset)/'manifest.json').read_text())['rows']
    prompts=sorted({ROBOTWIN_PROMPT_FORMAT.format(task=r['prompt']) for r in rows})
    contexts={}
    with torch.no_grad():
        for prompt in prompts:
            ctx,mask=model.encode_prompt([prompt])
            contexts[prompt]=(ctx[0].detach().cpu(),mask[0].detach().cpu())
    if getattr(model,'text_encoder',None) is not None:
        model.text_encoder.cpu();model.text_encoder=None
    gc.collect();torch.cuda.empty_cache()
    # DDP must call forward; unwrapping training_loss bypasses gradient sync.
    type(model).forward=lambda self,sample:self.training_loss(sample)
    stats=_load_norm_stats(Path(a.bundle)/'norm_stats.json')
    train=NativeChunks(a.dataset,'train',stats,contexts);val=NativeChunks(a.dataset,'val',stats,contexts)
    if not len(train):raise ValueError('No native training chunks')
    class Trainer(DW05Trainer):
        def save_checkpoint(self):
            # Each WM stage starts from weights; policy optimizer resume is separate.
            # Export the service-compatible weights once, without a duplicate model
            # plus optimizer dump that this pipeline never reloads.
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self._save_weights_checkpoint(f'step_{self.global_step:06d}')
            self.accelerator.wait_for_everyone()
        def compute_training_loss(self,sample): return self.model(sample)
        @torch.no_grad()
        def evaluate(self):
            if not len(val):return None
            raw=self.accelerator.unwrap_model(self.model);raw.eval()
            values=[]
            # Fixed cases and fixed flow noise allow an old/new comparison.
            with torch.random.fork_rng(devices=[self.accelerator.device.index]):
                torch.manual_seed(20261008)
                for i in np.linspace(0,len(val)-1,min(8,len(val)),dtype=int):
                    sample=default_collate([val[int(i)]])
                    with self.accelerator.autocast():loss,parts=raw.training_loss(sample)
                    values.append(dict(loss=float(loss),**parts))
            result={k:float(np.mean([r[k] for r in values])) for k in values[0]}
            if self.accelerator.is_main_process:
                with (output/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(step=self.global_step,**result))+'\n')
            self.set_train_mode();return result
        def _maybe_log_train_step(self,loss,metrics,grad_norm,lr):
            super()._maybe_log_train_step(loss,metrics,grad_norm,lr)
            if not np.isfinite(loss) or not np.isfinite(grad_norm):raise FloatingPointError('Nonfinite WM training')
            if self.accelerator.is_main_process:
                with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=self.global_step,loss=loss,
                    grad_norm=grad_norm,lr=lr,elapsed=time.perf_counter()-self.run_start_time,
                    peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),**metrics))+'\n')
    tc=DWTrainerConfig(output_dir=str(output),batch_size=a.batch,num_epochs=a.epochs,max_steps=a.steps,
        num_workers=2,persistent_workers=True,learning_rate=a.lr,gradient_accumulation_steps=a.accum,
        log_every=1,save_every=0,eval_every=0,save_final=not bool(a.steps),
        lr_scheduler_type='constant' if a.steps else 'cosine')
    trainer=Trainer(model=model,train_dataset=train,val_dataset=val,cfg=tc.to_trainer_cfg(),accelerator=acc)
    before=trainer.evaluate();torch.cuda.reset_peak_memory_stats();trainer.train()
    if a.steps:
        # Full Adam state and DDP buffers now exist. Estimate activation growth
        # from the measured peak, then validate ONE larger batch in this load.
        torch.cuda.synchronize();gc.collect()
        allocated=torch.cuda.memory_allocated();peak=torch.cuda.max_memory_allocated()
        free,total=torch.cuda.mem_get_info();external=max(0,total-free-torch.cuda.memory_reserved())
        transient=max(peak-allocated,512*1024**2)
        target=total*.90-external
        candidate=max(a.batch,int(a.batch*max(0,target-allocated)/transient))
        candidate=min(candidate,max(a.batch,len(train)//acc.num_processes))
        if fallback:candidate=a.batch
        sizes=acc.gather(torch.tensor([candidate],device=acc.device));candidate=int(sizes.min())
        baseline=dict(microbatch=a.batch,peak_allocated=peak,allocated_after_update=allocated,
            external_bytes=external,total_bytes=total,target_fraction=.90,candidate=candidate)
        if acc.is_main_process:
            (output/'batch_measurement.json').write_text(json.dumps(baseline,indent=2)+'\n')
            # A failed larger probe leaves this already passed size available.
            recipe_path.write_text(json.dumps(dict(microbatch=a.batch,validated=True,source=str(output),baseline=baseline),indent=2)+'\n')
        acc.wait_for_everyone()
        if candidate>a.batch:
            trainer.batch_size=candidate;trainer.cfg.batch_size=candidate
            loader=trainer.build_train_loader(train)
            trainer.train_loader=acc.prepare_data_loader(loader)
            trainer.max_steps=trainer.global_step+a.steps
            torch.cuda.reset_peak_memory_stats();trainer.train();a.batch=candidate
        trainer.save_checkpoint()
        if acc.is_main_process:
            recipe_path.write_text(json.dumps(dict(microbatch=a.batch,validated=True,source=str(output),baseline=baseline),indent=2)+'\n')
    after=trainer.evaluate()
    acc.wait_for_everyone()
    if acc.is_main_process:
        files=sorted((output/'checkpoints/weights').glob('*.pt'))
        assert files,'Official trainer did not export a checkpoint'
        (output/'result.json').write_text(json.dumps(dict(checkpoint=str(files[-1]),steps=trainer.global_step,
            world_size=acc.num_processes,microbatch=a.batch,global_batch=a.batch*acc.num_processes*a.accum,
            before=before,after=after,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()),indent=2)+'\n')

if __name__=='__main__':main()
