const fs=require('fs'),crypto=require('crypto');
const base='docs/fastwam-robotwin-rlinf-grpo/evidence/FASTWAM_LEARNING_AND_MINIMAL_FIX_REVIEW_20260904';
const all=['.source.txt','.actor.txt'].filter(s=>fs.existsSync(base+s)).flatMap(s=>fs.readFileSync(base+s,'utf8').split(/\r?\n/).filter(l=>l.startsWith('SOURCE_JSON ')).map(l=>JSON.parse(l.slice(12))));
if(!process.argv[2])for(const s of all){const p=`worktrees/${s.worktree}/${s.rel}`;const local=fs.existsSync(p)?crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'):null;console.log(JSON.stringify({worktree:s.worktree,rel:s.rel,sha:s.sha256,local_match:local===s.sha256,normalized_match:fs.existsSync(p)&&s.text!=null?fs.readFileSync(p,'utf8').replace(/\r\n/g,'\n')===s.text:false,lines:s.text?.split('\n').length}));}
const mode=process.argv[2];
for(const s of all){
 let ranges=[];
 if(mode==='openpi'&&s.rel.endsWith('openpi_action_model.py')&&s.worktree==='pi05-robotwin-rl'){
  const lines=s.text.split('\n');const hits=[];for(let i=0;i<lines.length;i++)if(/def (get_mean_std|sample_actions|freeze_|get_logprobs)|linspace|denoise_inds =|return.*logprob/.test(lines[i]))hits.push(i);const chosen=new Set;for(const i of hits)for(let j=Math.max(0,i-4);j<Math.min(lines.length,i+30);j++)chosen.add(j);console.log('EXCERPT',s.worktree,s.rel);console.log([...chosen].sort((a,b)=>a-b).map(i=>`${i+1}: ${lines[i]}`).join('\n'));
 }
 if(mode==='actor'&&s.rel.endsWith('embodied_fsdp_actor_worker.py')){const lines=s.text.split('\n');const chosen=new Set;for(let i=0;i<lines.length;i++)if(/masked_mean|np.mean|loss_mask_sum|gradient_accumulation|clip_ratio_c|filter_rewards/.test(lines[i]))for(let j=Math.max(0,i-3);j<Math.min(lines.length,i+5);j++)chosen.add(j);console.log('EXCERPT',s.worktree,s.rel);console.log([...chosen].sort((a,b)=>a-b).map(i=>`${i+1}: ${lines[i]}`).join('\n'));}
}
