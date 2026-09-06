const fs=require('fs');
const base='docs/rlinf-shenzhen-multitask-pi05/evidence/';
const lines=fs.readFileSync(base+'PI05_PI0_COMPARISON_DATA_SOURCE_20260904.txt','utf8').split(/\r?\n/);
const read=k=>lines.filter(l=>l.startsWith(k+' ')).map(l=>JSON.parse(l.slice(k.length+1)));
const runs=read('RUN_JSON'),sources=read('SOURCE_JSON');
let latest=null;
const actorFile=base+'PI05_PI0_COMPARISON_ACTOR_20260904.txt';
if(fs.existsSync(actorFile)){
 const row=fs.readFileSync(actorFile,'utf8').split(/\r?\n/).find(l=>l.startsWith('LATEST_SCALARS_JSON '));
 if(row){latest=JSON.parse(row.slice(20));Object.assign(runs.find(r=>r.key==='sidney'),{scalars:latest.scalars,progress:latest.progress,errors:latest.errors})}
}
for(const file of ['PI05_PI0_COMPARISON_SUPPLEMENT_20260904.txt','PI05_PI0_COMPARISON_ACTOR_20260904.txt']){
 if(fs.existsSync(base+file))for(const l of fs.readFileSync(base+file,'utf8').split(/\r?\n/).filter(l=>l.startsWith('SOURCE_JSON '))){
  const s=JSON.parse(l.slice(12));sources.push({...s,root:s.path.slice(0,s.path.lastIndexOf('/')),rel:s.path.split('/').at(-1)});
 }
}
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
const median=a=>{a=[...a].sort((a,b)=>a-b);return a.length?(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2:null};
const flat=(v,p='',out={})=>{if(v&&typeof v==='object'&&!Array.isArray(v)){for(const[k,x]of Object.entries(v))flat(x,p?p+'.'+k:k,out)}else out[p]=v;return out};
const analysis={time:read('TIME_JSON'),latest_time:latest?.time,runs:{},series:{},diff:[],hashes:[]};
for(const r of runs){
 const metric=(tag,lo=1,hi=Infinity)=>r.scalars[tag]?.filter(p=>p.step>=lo&&p.step<=hi).map(p=>p.value)||[];
 const tr=r.scalars['env/success_once'],ev=r.scalars['eval/success_once'];
 const windows=[[1,5],[1,10],[6,10],[11,20],[21,30],[31,40],[41,50],[43,52],[44,53],[87,96]];
 const model=r.config.actor.model,env=r.config.env.train;
 analysis.series[r.key]=r.scalars;
 analysis.runs[r.key]={progress:r.progress,metric_step:tr.at(-1).step,train_last:tr.at(-1),
  windows:windows.map(([lo,hi])=>({lo,hi,n:metric('env/success_once',lo,hi).length,train:mean(metric('env/success_once',lo,hi))})),
  eval:ev.map(p=>({step:p.step,n:p.value*32})),
  time:Object.fromEntries(Object.keys(r.scalars).filter(t=>t.startsWith('time/')).map(t=>[t,{first50_mean:mean(metric(t,1,50)),first50_median:median(metric(t,1,50)),last10_mean:mean(metric(t,tr.at(-1).step-9))}])),
  diagnostics:Object.fromEntries(Object.keys(r.scalars).filter(t=>t.startsWith('train/')||t.includes('mask')||t.includes('length')||t.includes('reward')).map(t=>[t,{first50:mean(metric(t,1,50)),last10:mean(metric(t,tr.at(-1).step-9)),last:r.scalars[t].at(-1)?.value}])),
  wall_hours_to50:(tr.find(p=>p.step===50).wall_time-Date.parse(r.state['started_at.txt'])/1000)/3600,
  config:{algorithm:r.config.algorithm,model,optim:r.config.actor.optim,env,actor_offload:r.config.actor.enable_offload,env_offload:r.config.env.enable_offload,rollout:r.config.rollout},
  state:r.state,wrapper_alive:r.wrapper_alive,errors:r.errors};
}
const a=flat(runs.find(r=>r.key==='pi0').config),b=flat(runs.find(r=>r.key==='sidney').config);
analysis.diff=[...new Set([...Object.keys(a),...Object.keys(b)])].sort().filter(k=>JSON.stringify(a[k])!==JSON.stringify(b[k])).map(k=>({key:k,pi0:a[k],sidney:b[k]}));
for(const rel of [...new Set(sources.map(s=>s.rel))])analysis.hashes.push({rel,...Object.fromEntries(sources.filter(s=>s.rel===rel).map(s=>[s.root.includes('/worktrees/')?s.root.split('/worktrees/')[1].split('/')[0]:s.root,s.sha256]))});
fs.writeFileSync(base+'PI05_PI0_COMPARISON_ANALYSIS_20260904.json',JSON.stringify(analysis,null,2));
if(process.argv[2]==='source'){
 const root=process.argv[3]||'sidney-pi05-current-rlinf',rel=process.argv[4],pat=process.argv[5];
 for(const s of sources.filter(s=>(root==='*'||s.root.endsWith(root))&&(!rel||s.rel.includes(rel)))){
  const ls=s.text.split('\n'); console.log('\nFILE '+s.root+'/'+s.rel+' SHA '+s.sha256);
  if(!pat){ls.forEach((l,i)=>console.log((i+1)+': '+l));continue}
  const idx=new Set(),range=pat.match(/^:(\d+)-(\d+)$/);
  if(range){for(let j=+range[1]-1;j<Math.min(ls.length,+range[2]);j++)idx.add(j)}
  else ls.forEach((l,i)=>{if(new RegExp(pat).test(l))for(let j=Math.max(0,i-5);j<=Math.min(ls.length-1,i+15);j++)idx.add(j)});
  [...idx].sort((a,b)=>a-b).forEach(i=>console.log((i+1)+': '+ls[i]));
 }
}else{
 for(const[key,r]of Object.entries(analysis.runs)){const {config,...rest}=r;console.log(JSON.stringify({key,...rest},null,2))}
 console.log('LEAF_DIFF',JSON.stringify(analysis.diff,null,2));
 console.log('HASH_COMPARISON',JSON.stringify(analysis.hashes,null,2));
}
