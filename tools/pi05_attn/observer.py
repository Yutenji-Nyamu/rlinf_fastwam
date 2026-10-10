"""Read-only native pi0.5 attention observer. No solver/kernel/RNG changes."""
import importlib,inspect,time
import numpy as np
import torch

LAYERS=(9,13,17)
SCHEMA="pi05-attn-raw-v1"

def cpu(x):
    return x.detach().to("cpu",copy=True)

class AttentionObserver:
    def __init__(self,model):
        self.model=model;self.active=False;self.originals=[];self.handles=[]
        modules=model.paligemma_with_expert.gemma_expert.model.layers
        self.selected={id(modules[i].self_attn):i for i in LAYERS}
        gm=importlib.import_module(type(modules[0].self_attn).__module__)
        original=gm.eager_attention_forward
        self.kernel_source=dict(path=inspect.getsourcefile(original),function=original.__name__)
        def attention(module,query,key,value,attention_mask,scaling,*args,**kwargs):
            result=original(module,query,key,value,attention_mask,scaling,*args,**kwargs)
            if self.active and id(module) in self.selected:
                lid=self.selected[id(module)];k=self.step;copy_started=time.perf_counter()
                assert k>=0 and query.shape[1:]==(8,50,256) and key.shape[1]==1
                if k==0:
                    self.prefix[lid]=cpu(key[:,:,:-50])
                    self.scales[lid]=float(scaling)
                self.q[k,lid]=cpu(query);self.k[k,lid]=cpu(key[:,:,-50:])
                if lid==LAYERS[0]:
                    self.masks.append(cpu(attention_mask))
                if self.reference:self.refs[k,lid]=cpu(result[1][:1])
                self.qk_copy_sync_seconds+=time.perf_counter()-copy_started
            return result
        gm.eager_attention_forward=attention
        self.originals.append((gm,'eager_attention_forward',original))
        self.wrap('_preprocess_observation',self.preprocess)
        self.wrap('_build_prefix_cache',self.prefix_call)
        self.wrap('get_velocity',self.velocity)
        self.wrap('sample_noise',self.noise)
        expert=model.paligemma_with_expert.gemma_expert.model
        expert_forward=expert.forward
        def forward(*args,**kwargs):
            if self.active:self.position_ids.append(cpu(kwargs['position_ids']))
            return expert_forward(*args,**kwargs)
        expert.forward=forward;self.originals.append((expert,'forward',expert_forward))
        # Observe exact original tokenization; proto is used only for source offsets.
        tm=importlib.import_module('openpi.models.tokenizer')
        tok_orig=tm.PaligemmaTokenizer.tokenize
        def tokenize(tokenizer,prompt,state=None):
            result=tok_orig(tokenizer,prompt,state)
            if self.active and state is not None:
                text=prompt.strip().replace('_',' ').replace('\n',' ')
                discrete=np.digitize(state,bins=np.linspace(-1,1,257)[:-1])-1
                state_text=' '.join(map(str,discrete))
                full=f"Task: {text}, State: {state_text};\nAction: "
                proto=tokenizer._tokenizer.encode(full,return_type='proto',add_bos=False,add_eos=False,reverse=False,emit_unk_piece=False)
                pieces=list(proto.pieces);ids=[tokenizer._tokenizer.bos_id()]+[p.id for p in pieces]
                n=int(result[1].sum());assert ids[:n]==result[0][:n].tolist()
                ib,ie=6,6+len(text);sb=ie+9;se=sb+len(state_text)
                labels=['other'];offsets=[[0,0]]
                for p in pieces[:n-1]:
                    b,e=p.begin,p.end
                    label='instruction' if e>b and b>=ib and e<=ie else 'state' if e>b and b>=sb and e<=se else 'other'
                    labels.append(label);offsets.append([b,e])
                row=dict(full_prompt=full,labels=labels,offsets=offsets,normalized_state=np.asarray(state).tolist(),discretized_state=discrete.tolist(),instruction_span=[ib,ie],state_span=[sb,se])
                self.token_rows[tuple(result[0].tolist())]=row
            return result
        tm.PaligemmaTokenizer.tokenize=tokenize
        self.originals.append((tm.PaligemmaTokenizer,'tokenize',tok_orig))

    def wrap(self,name,handler):
        original=getattr(self.model,name)
        def call(*args,**kwargs):return handler(original,*args,**kwargs)
        setattr(self.model,name,call);self.originals.append((self.model,name,original))

    def begin(self,reference=False):
        self.active=True;self.reference=reference;self.step=-1;self.q={};self.k={};self.prefix={};self.scales={};self.refs={}
        self.masks=[];self.position_ids=[];self.x=[];self.v=[];self.t=[];self.token_rows={};self.initial_noise_draw=None;self.qk_copy_sync_seconds=0.0
        self.started=time.perf_counter()

    def preprocess(self,original,observation,*args,**kwargs):
        result=original(observation,*args,**kwargs)
        if self.active:
            images,img_masks,tokens,masks,state=result
            self.tokens=cpu(tokens);self.lang_masks=cpu(masks);self.state=cpu(state)
            self.image_keys=list(observation.images)
            self.image_shapes=[list(x.shape) for x in images]
            self.image_masks=[cpu(x) for x in img_masks]
        return result

    def prefix_call(self,original,*args,**kwargs):
        result=original(*args,**kwargs)
        if self.active:self.prefix_mask=cpu(result[1])
        return result

    def noise(self,original,*args,**kwargs):
        result=original(*args,**kwargs)
        if self.active and self.step==-1:
            assert self.initial_noise_draw is None
            self.initial_noise_draw=cpu(result)
        return result

    def velocity(self,original,state,x,t,*args,**kwargs):
        if self.active:
            self.step+=1;self.x.append(cpu(x));self.t.append(cpu(t))
        result=original(state,x,t,*args,**kwargs)
        if self.active:self.v.append(cpu(result[0]))
        return result

    def positions(self,module,args,kwargs):
        if self.active:self.position_ids.append(cpu(kwargs['position_ids']))

    def finish(self,result,actions):
        self.active=False
        assert self.step==9 and len(self.q)==30 and len(self.k)==30 and len(self.masks)==10 and len(self.position_ids)==10
        assert self.initial_noise_draw is not None and torch.equal(self.initial_noise_draw.to(self.x[0].dtype),self.x[0])
        chains=cpu(result['forward_inputs']['chains'])
        assert chains.shape[1:]==(11,50,32) and torch.equal(chains[:,:10],torch.stack(self.x,dim=1))
        tokens=self.tokens.numpy();nvis=self.prefix_mask.shape[-1]-tokens.shape[-1]
        assert nvis%len(self.image_keys)==0
        assert all(torch.equal(m,m[:,:,:1,:].expand_as(m)) for m in self.masks), 'Unexpected action-dependent key mask'
        per_camera=nvis//len(self.image_keys);side=int(per_camera**.5);assert side*side==per_camera
        source=torch.full((tokens.shape[0],self.prefix_mask.shape[-1]+50),4,dtype=torch.int64)
        source[:,:nvis]=0;source[:,-50:]=3;meta=[]
        for i,row in enumerate(tokens):
            item=dict(self.token_rows[tuple(row.tolist())]);item['normalized_state']=self.state[i].tolist();meta.append(item)
            for j,label in enumerate(item['labels']):source[i,nvis+j]={'instruction':1,'state':2,'other':4}[label]
        ts=torch.stack(self.t,dim=1);dt=ts-torch.cat([ts[:,1:],torch.zeros_like(ts[:,:1])],dim=1)
        raw=dict(schema=SCHEMA,q_action=torch.stack([torch.stack([self.q[k,l] for l in LAYERS],dim=1) for k in range(10)],dim=1),
            k_prefix=torch.stack([self.prefix[l] for l in LAYERS],dim=1),
            k_action=torch.stack([torch.stack([self.k[k,l] for l in LAYERS],dim=1) for k in range(10)],dim=1),
            attention_mask=torch.stack(self.masks,dim=1),position_ids=torch.stack(self.position_ids,dim=1),
            prefix_position_ids=self.prefix_mask.long().cumsum(-1)-1,prefix_valid=self.prefix_mask,
            source=source,token_ids=self.tokens,token_valid=self.lang_masks,normalized_state=self.state,
            image_masks=self.image_masks,scaling=[self.scales[l] for l in LAYERS],
            initial_noise_full=self.initial_noise_draw,solver_initial_noise=self.x[0],x_chain=chains,velocity=torch.stack(self.v,dim=1),timesteps=ts,dt=dt,final_model_action=chains[:,-1],key_valid=self.masks[0][:,0,0]>-1e4,
            native_dv_telemetry={k:cpu(v) for k,v in result['dvac_telemetry'].items()},
            env_action=cpu(actions) if torch.is_tensor(actions) else torch.as_tensor(actions).clone(),valid_action_dims=list(range(14)))
        if self.reference:raw['native_attention_reference']=torch.stack([torch.stack([self.refs[k,l] for l in LAYERS],dim=1) for k in range(10)],dim=1)
        metadata=dict(schema=SCHEMA,layers_1based=[10,14,18],heads=list(range(8)),denoise_calls=10,token_rows=meta,
            image_keys=self.image_keys,image_shapes=self.image_shapes,patch_grid=[side,side],patches_per_camera=per_camera,
            camera_columns=[dict(camera=key,start=i*per_camera,end=(i+1)*per_camera,positions=[[r,c] for r in range(side) for c in range(side)]) for i,key in enumerate(self.image_keys)],
            language_start=nvis,key_order='camera prefixes then original padded text then 50 action slots',
            capture_revision='v1.1',token_boundary_rule='Whole SentencePiece token must be within span; boundary-crossing token is other',qk_copy_sync_seconds=self.qk_copy_sync_seconds,noise_capture='Full original FP32 random draw and actual cast solver latent both retained',qk_capture='post-RoPE pre-scaling; original native eager call unchanged',kernel=self.kernel_source,
            dtypes={k:str(v.dtype) for k,v in raw.items() if torch.is_tensor(v)},observer_seconds=time.perf_counter()-self.started)
        return raw,metadata

    def close(self):
        for h in self.handles:h.remove()
        for obj,name,original in reversed(self.originals):setattr(obj,name,original)
