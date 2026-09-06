// Fetch primary-source text only; never import or execute downloaded project code.
const fs=require('fs'),path=require('path');
const base='docs/rlinf-robotwin-pi0-online-bc/evidence/source-audit-20260904';
const headers={'User-Agent':'rl-online-bc-readonly-research','Accept':'application/vnd.github+json'};
async function get(url,json=true){const r=await fetch(url,{headers,signal:AbortSignal.timeout(30000)});if(!r.ok)throw Error(r.status+' '+url);return json?r.json():r.text();}
const key=r=>r.replaceAll('/','__');
function save(p,v){fs.mkdirSync(path.dirname(p),{recursive:true});fs.writeFileSync(p,typeof v==='string'?v:JSON.stringify(v,null,2));}
async function discover(repo){
 const root=base+'/'+key(repo),meta=await get('https://api.github.com/repos/'+repo),commit=await get('https://api.github.com/repos/'+repo+'/commits/'+meta.default_branch);
 const lock={repo,sha:commit.sha,commit_time:commit.commit.committer.date,checked:new Date().toISOString(),stars:meta.stargazers_count,license:meta.license?.spdx_id,default_branch:meta.default_branch};
 save(root+'/lock.json',lock);
 const tree=await get('https://api.github.com/repos/'+repo+'/git/trees/'+commit.sha+'?recursive=1');save(root+'/tree.json',tree);
 console.log(JSON.stringify({...lock,truncated:tree.truncated,paths:tree.tree.filter(x=>x.type==='blob'&&/readme|rabc|weighting|dagger|sft_forward|record\.py|config.*ya?ml$|train.*\.py$|online.*\.py$|episode_filter|replay_buffer|sample.*weight|train.*\.sh$|collect.*\.py$/i.test(x.path)).map(x=>x.path).slice(0,100)}));
}
async function files(repo,paths){const root=base+'/'+key(repo),lock=JSON.parse(fs.readFileSync(root+'/lock.json'));for(const p of paths){const text=await get('https://raw.githubusercontent.com/'+repo+'/'+lock.sha+'/'+p,false);save(root+'/source/'+p,text);console.log(JSON.stringify({repo,sha:lock.sha,path:p,bytes:Buffer.byteLength(text)}));}}
async function main(){const[mode,...args]=process.argv.slice(2);if(mode==='discover'){for(let i=0;i<args.length;i+=3)await Promise.all(args.slice(i,i+3).map(discover));}else if(mode==='files')await files(args[0],args.slice(1));else throw Error('discover REPOS... | files REPO PATHS...');}
main().catch(e=>{console.error(e);process.exitCode=1});
