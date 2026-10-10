"""FP32 reconstruction and action-aligned E / Flow diagnostics from native Q/K."""
import torch
import numpy as np
GROUPS=('visual','instruction','state','action')
def distribution(a,mask):
    z=a*mask.unsqueeze(-2);mass=z.sum(-1);n=mask.sum(-1)
    p=z/mass.clamp_min(1e-30).unsqueeze(-1)
    entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)
    valid=(mass>0)&(n.unsqueeze(-1)>1)
    e=entropy/n.clamp_min(2).float().log().unsqueeze(-1)
    return p,mass,torch.where(valid,e,torch.nan),valid

def js(p,q):
    mid=(p+q)/2
    return ((p*(p.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum(-1)+(q*(q.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum(-1))/(2*np.log(2))

def score(a,source,key_valid):
    result={}
    for gi,name in enumerate(GROUPS):
        mask=(source==gi)&key_valid
        p,m,e,valid=distribution(a,mask)
        boundary=torch.full_like(e,torch.nan);boundary[:,1:]=js(p[:,1:],p[:,:-1])
        pair=valid[:,1:]&valid[:,:-1];boundary[:,1:]=torch.where(pair,boundary[:,1:],torch.nan)
        count=mask.sum(-1)[:,None].expand_as(e);hraw=-(p*p.clamp_min(1e-30).log()).sum(-1)
        result.update({f'H_{name}':torch.where((count>0)&(m>0),hraw,torch.nan),f'N_{name}':count,f'E_{name}':e,f'M_{name}':m,f'valid_{name}':valid,f'B_{name}':boundary,f'low_mass_{name}':m<1e-4})
    result['M_other']=(a*((source==4)&key_valid).unsqueeze(-2)).sum(-1)
    pm=key_valid.clone();pm[:,-50:]=False
    result['E_obs']=distribution(a,pm)[2]
    block=a[:,:,-50:];s=block/block.sum(-1,keepdim=True).clamp_min(1e-30);h=s.shape[-1]
    diag=s.diagonal(dim1=-2,dim2=-1);positions=torch.arange(h,device=a.device)
    span=(s*(positions[:,None]-positions[None,:]).abs()/(h-1)).sum(-1)
    b=torch.full_like(span,torch.nan);b[:,1:]=(span[:,1:]-span[:,:-1]).abs()
    anchor=(s.sum(-2)-diag)/(h-1);readers=torch.arange(h-1,-1,-1,device=a.device)
    future=s.tril(-1).sum(-2)/readers.clamp_min(1);future[:,-1]=torch.nan
    result.update(D_span=span,B_span=b,I_anchor=anchor,I_future=future,I_future_sum=s.tril(-1).sum(-2),I_future_readers=readers.expand_as(future),I_anchor_joint=(block.sum(-2)-block.diagonal(dim1=-2,dim2=-1))/(h-1),self_read=diag)
    return result

def reconstruct(raw,device='cpu'):
    source=raw['source'].to(device);b=source.shape[0];scores=[];layers=[];heads=[];steps=[]
    nvis=raw['prefix_valid'].shape[-1]-raw['token_ids'].shape[-1]
    prev=None;refs=[];aa=[];visual_maps=[]
    for k in range(10):
        per_layer=[];layer_e=[];head_e=[];head_j=[]
        for l in range(3):
            q=raw['q_action'][:,k,l].to(device=device,dtype=torch.float32)
            key=torch.cat([raw['k_prefix'][:,l],raw['k_action'][:,k,l]],dim=-2).to(device=device,dtype=torch.float32)
            mask=raw['attention_mask'][:,k].to(device=device,dtype=torch.float32)
            logits=(q@key.transpose(-2,-1))*raw['scaling'][l]+mask
            a=torch.softmax(logits,dim=-1)
            valid=mask[:,0,0]>-1e4
            means=a.mean(1);per_layer.append(means)
            es=[];jsheads=[]
            for gi in range(4):
                gm=((source==gi)&valid)[:,None,:].expand(-1,8,-1).reshape(b*8,-1)
                hp,_,he,_=distribution(a.reshape(b*8,50,-1),gm)
                es.append(he.reshape(b,8,50))
                hp=hp.reshape(b,8,50,-1);ph=hp.mean(1)
                jsheads.append(-(ph*ph.clamp_min(1e-30).log()).sum(-1)+(hp*hp.clamp_min(1e-30).log()).sum(-1).mean(1))
            head_e.append(torch.stack(es,dim=-1).cpu());head_j.append(torch.stack(jsheads,dim=-1).cpu())
            layer_scores=score(means,source,valid)
            layer_e.append(torch.stack([layer_scores['E_'+g] for g in GROUPS],dim=-1).cpu())
            if 'native_attention_reference' in raw:
                ref=raw['native_attention_reference'][0,k,l].to(device).float()
                native_q=raw['q_action'][:,k,l].to(device)
                native_k=torch.cat([raw['k_prefix'][:,l],raw['k_action'][:,k,l]],dim=-2).to(device).repeat_interleave(8,dim=1)
                native_logits=(native_q@native_k.transpose(-2,-1))*raw['scaling'][l]
                native_a=torch.softmax(native_logits+raw['attention_mask'][:,k].to(device),dim=-1,dtype=torch.float32).to(native_q.dtype)
                native_delta=(native_a[0].float()-ref).abs().max().item()
                delta=(a[0]-ref).abs()
                refs.append(dict(step=k,layer=l,native_precision_max_abs=native_delta,max_abs=delta.max().item(),max_row_l1=delta.sum(-1).max().item(),top1_agreement=(a[0].argmax(-1)==ref.argmax(-1)).float().mean().item(),entropy_mae=(-(a[0]*a[0].clamp_min(1e-30).log()).sum(-1)+(ref*ref.clamp_min(1e-30).log()).sum(-1)).abs().mean().item()))
        mean=torch.stack(per_layer).mean(0)
        if 'native_attention_reference' in raw:
            native_mean=raw['native_attention_reference'][0,k].float().mean(1).mean(0).to(device)
            md=(mean[0]-native_mean).abs()
            for refrow in refs[-3:]:refrow.update(main_max_abs=md.max().item(),main_max_row_l1=md.sum(-1).max().item())
        current=score(mean,source,valid)
        current['J_layer']=torch.stack([torch.stack([js(distribution(per_layer[l],(source==gi)&valid)[0],distribution(per_layer[l-1],(source==gi)&valid)[0]) for gi in range(4)],dim=-1) for l in range(1,3)],dim=1)
        changes=[]
        for gi in range(4):
            p=distribution(mean,(source==gi)&valid)[0]
            changes.append(torch.full_like(p[:,:,0],torch.nan) if prev is None else js(p,prev[gi]))
        current['J_step']=torch.stack(changes,dim=-1)
        prev=[distribution(mean,(source==gi)&valid)[0] for gi in range(4)]
        current['J_head']=torch.stack(head_j,dim=1).mean(1).to(device)
        scores.append({key:value.cpu() for key,value in current.items()})
        layers.append(torch.stack(layer_e,dim=1));heads.append(torch.stack(head_e,dim=1))
        if k in [0,4,9]:
            aa.append(mean[:,:,-50:].cpu());visual_maps.append(mean[:,:,:nvis].cpu())
    arrays={key:torch.stack([s[key] for s in scores],dim=1).numpy() for key in scores[0]}
    arrays['layer_entropy']=torch.stack(layers,dim=1).numpy()
    arrays['head_entropy']=torch.stack(heads,dim=1).numpy()
    arrays['action_attention_1_5_10']=torch.stack(aa,dim=1).numpy()
    arrays['visual_attention_1_5_10']=torch.stack(visual_maps,dim=1).numpy()
    # Native DV uses original endpoint arithmetic; retain exact accepted definition.
    arrays['dv']=raw['native_dv_telemetry']['z_endpoint'][:,-5:].float().numpy().astype(np.float64).var(axis=1,ddof=0).sum(-1)
    arrays['dt_from_actual_t']=(raw['timesteps']-torch.cat([raw['timesteps'][:,1:],torch.zeros_like(raw['timesteps'][:,:1])],dim=1)).numpy()
    arrays['action_amplitude']=raw['env_action'].float().square().mean(-1).sqrt().numpy()
    return arrays,refs
