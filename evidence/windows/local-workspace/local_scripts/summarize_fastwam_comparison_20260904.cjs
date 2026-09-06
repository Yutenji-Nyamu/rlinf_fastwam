const fs=require('fs');
const input='docs/fastwam-robotwin-rlinf-grpo/evidence/FASTWAM_LEARNING_AND_MINIMAL_FIX_REVIEW_20260904.data.txt';
const rows=fs.readFileSync(input,'utf8').replace(/^\uFEFF/,'').split(/\r?\n/).filter(l=>l.startsWith('RUN_JSON ')).map(l=>JSON.parse(l.slice(9)));
const avg=a=>a.length?a.reduce((s,p)=>s+p.value,0)/a.length:null;
for(const [i,r] of rows.entries()){
 const c=r.config||{},t=c.env?.train||{},m=c.actor?.model||{};
 const metric=tag=>{const a=r.scalars[tag]||[];return {first:avg(a.slice(0,5)),last5:avg(a.slice(-5)),last10:avg(a.slice(-10)),last:a.at(-1)?.value,max:Math.max(...a.map(p=>p.value))}};
 console.log(JSON.stringify({i,name:r.name,step:r.last_progress,alive:r.wrapper_alive,task:t.task_config?.task_name,batch:[c.actor?.global_batch_size,c.actor?.micro_batch_size],algorithm:c.algorithm,model:m,optim:c.actor?.optim,env:t,train:metric('env/success_once'),eval:r.scalars['eval/success_once']?.map(p=>[p.step,p.value*32]),kl:metric('train/actor/approx_kl'),clip:metric('train/actor/clip_fraction'),grad:metric('train/actor/grad_norm'),lr:metric('train/actor/lr'),ratio:metric('train/actor/ratio'),time:metric('time/step'),trainable:r.trainable_logs,fatal:r.fatal,checkpointCount:r.checkpoints.length,resourceHead:r.resource.slice(0,1)}));
}
