"""Append fresh return status and publish the read-only proc-race correction."""
import ast,base64,hashlib,json
from pathlib import Path
W=Path(__file__).resolve().parents[2];H=Path(__file__).resolve().parent
D=W/'docs/world-model/task_reward_plan_20261006'
O=Path('E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003')
s=json.loads((O/'rlt45-audit1007-final2.out').read_text())
p=D/'rlt45_return_20261007.json';r=json.loads(p.read_text());r['last_verified_at']=s['time'];r['gpu']=s['gpu']
for k,v in s['runs'].items():
    r['runs'][k]['phase']='loading checkpoint; first full resumed round not yet verified'
    r['runs'][k]['resume_log']=[x for l in v['logs'].values() for x in l['tail'].splitlines() if 'Resuming training from checkpoint' in x]
p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
note='\n01:37:47追加：两路日志均明确进入各自CP225/250加载；GPU4/5各约10.86GiB，actor load_checkpoint，rollout模型已加载。6/7身份保持，0–3无上下文；首个恢复完整轮尚未验收。只读状态脚本修复/proc短命进程退出时的枚举竞争，不影响训练。首次归还实现/文档已推5fa4ab1abc31b4c5c272bc8470b394dade444117。\n'
with (D/'rlt45_return_20261007.md').open('a',encoding='utf-8') as f:f.write(note)
text=(H/'rlt45_publication_remote.py').read_text();tree=ast.parse(text);P=ast.literal_eval(tree.body[0].value)
P['prior']='5fa4ab1abc31b4c5c272bc8470b394dade444117';P['release_name']='rlt45-return-final-20261007-v1'
names=['docs/world-model/task_reward_plan_20261006/'+n for n in ['rlt45_return_20261007.md','rlt45_return_20261007.json','rlt45_return_published.json']]+['local_scripts/lift_two_gpu_20261006/rlt45_status_remote.py','local_scripts/lift_two_gpu_20261006/finalize_rlt45_record.py']
P['files']={}
for n in names:
    raw=(W/n).read_bytes().replace(b'\r\n',b'\n')
    if n.endswith('.py'):ast.parse(raw)
    P['files'][n]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
remote=text.split('\n',1)[1].replace("assert P['release_name']=='rlt45-return-20261007-v1'","assert P['release_name']=='rlt45-return-final-20261007-v1'")
(H/'rlt45_final_publication_remote.py').write_text('P = '+repr(P)+'\n'+remote,encoding='utf-8',newline='\n')
