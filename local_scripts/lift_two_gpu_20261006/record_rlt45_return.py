"""Record scoped return evidence and prepare a narrow publication package."""
import ast,base64,hashlib,json
from pathlib import Path
from make_result_publication import REMOTE
W=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
D=W/'docs/world-model/task_reward_plan_20261006'
O=Path('E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003')
s=json.loads((O/'rlt45-audit1007-status2.out').read_text())
light={k:s[k] for k in ['time','dispatch','gpu']}
light['protected']=[{k:r[k] for k in ['pid','alive','uid','start']} for r in s['protected']]
light['runs']={}
for k,r in s['runs'].items():
    light['runs'][k]={q:r[q] for q in ['run','namespace','checkpoint','receipts']}
    for v in light['runs'][k]['receipts'].values():
        if 'current' in v:v['current'].pop('argv',None)
    light['runs'][k]['phase']='initializing; no completed post-resume round verified'
light['wm']={'state':'failed_then_deferred_by_user','failure_time':'2026-10-07T00:45:01+08:00','failure':'CUDA OOM in VAE single_decode/F.pad on physical GPU5','requested_GiB':5.58,'free_GiB':2.05,'completed_rounds':0,'new_checkpoints':0,'restarted':False}
(D/'rlt45_return_20261007.json').write_text(json.dumps(light,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
note='''# 2026-10-07：WM暂缓，4/5恢复RLT

用户最后指示“放RLT，WM明天再说”。本轮只恢复深圳3 GPU4/5；WM未重启。

01:33:31/43，两路RLT分别从完整CP225（clean）/CP250（combo）派发，训练参数保持。01:35两路driver身份和worker均存活，正在初始化；尚未验证恢复后的完整一轮。GPU6/7的Norm实验原driver3182779/3182804及owner3651109保持。现场0–3无本任务计算或图形上下文，未清理其他用户。

| 卡 | driver PID / start | namespace | checkpoint |
|---|---|---|---|
| 4 | 1046451 / 727903513 | rlt-opendw-audit1007-gpu4 | 225 |
| 5 | 1047081 / 727904666 | rlt-opendw-audit1007-gpu5 | 250 |

唯一归还路由：`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rlt45-return-20261007-v1/cycle`。启动源码为各`return-g4/g5/rlt_returned_cycle.py --cycle-dir <该子目录> driver --key gpu4/gpu5`，详细输出位置在[轻量回执](rlt45_return_20261007.json)。旧归还cycle已写superseded指向新路由，请勿重放旧入口。

归还阻塞原因：新增Norm6/7的NVIDIA DSO匹配规则不在旧scope冻结清单。新scope只追加这两份规则的精确内容/hash审计，原profile、GPU mask、运行源码及Norm环境不变；RLT不加载Norm marker。未重启共享Ray。

WM事实：from0-v2于00:45在GPU5视频VAE解码OOM（申请5.58GiB、剩2.05GiB；allocated56.14/reserved-unallocated19.62GiB），00:46清理完成。首轮只到采样2/8，无完整更新、无新checkpoint；旧四卡CP10及历史日志保留。短B32通过不能证明长采样稳定。VAE拆批等候选仅本地草稿，未部署、未GPU验证，按用户要求留到之后处理。

已有profile解析器会报Norm两个匿名profile重名警告；本轮没有更改其文件。后续以实际进程计算/图形卡位为准。RLT启动后沿原错误退出行为，不新增WM重试或自动切回WM。
'''
(D/'rlt45_return_20261007.md').write_text(note,encoding='utf-8',newline='\n')
route='2026-10-07 01:35：**按用户要求暂停WM，GPU4/5恢复RLT。** CP225/250已派发，driver1046451/1047081及worker存活、正在初始化；6/7 Norm原实验保持。WM from0-v2视频VAE解码OOM退出，未完成首轮、无新CP。新归还cycle为`rlt45-return-20261007-v1/cycle`，请勿重放旧WM/归还入口。详见[归还记录](rlt45_return_20261007.md)。'
for name in ['README.md','two_gpu_b32_execution.md']:
    p=D/name;t=p.read_text(encoding='utf-8');head,body=t.split('\n',1)
    p.write_text(head+'\n\n'+route+'\n'+body,encoding='utf-8',newline='\n')
p=W/'HANDOFF.md';t=p.read_text(encoding='utf-8');head,body=t.split('\n',1)
p.write_text(head+'\n\n'+route.replace('(rlt45_return_20261007.md)','(docs/world-model/task_reward_plan_20261006/rlt45_return_20261007.md)')+'\n'+body,encoding='utf-8',newline='\n')
names=['docs/world-model/task_reward_plan_20261006/'+n for n in ['README.md','two_gpu_b32_execution.md','rlt45_return_20261007.md','rlt45_return_20261007.json']]
names+=['local_scripts/lift_two_gpu_20261006/'+n for n in ['restore_rlt45_remote.py','rlt45_status_remote.py','record_rlt45_return.py']]
files={}
for n in names:
    raw=(W/n).read_bytes().replace(b'\r\n',b'\n');assert len(raw)<262144
    if n.endswith('.py'):ast.parse(raw)
    files[n]={'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
P={'prior':'25e2b9a791956d47d761fcb08d82d583b909f00f','branch':'codex/wmrl-bell-reward-20261005','files':files,'runtime_head':'9ce50c602c5e773e87e3306218fdde0cf167e8a8','release_name':'rlt45-return-20261007-v1'}
remote=REMOTE.replace("assert P['release_name'] in ['lift-two-gpu-b32-result-20261006-v1','lift-two-gpu-from0-started-20261007-v1']","assert P['release_name']=='rlt45-return-20261007-v1'")
a=remote.index("F=S/'lift-two-gpu-from0-v2'");b=remote.index('assert not E.exists()',a);remote=remote[:a]+remote[b:]
a=remote.index("message='Record fresh");b=remote.index("git(W,'commit'",a);remote=remote[:a]+"message='Restore GPU4/5 RLT and defer WMRL after VAE OOM'\n"+remote[b:]
source='P = '+repr(P)+'\n'+remote;ast.parse(source)
(HERE/'rlt45_publication_remote.py').write_text(source,encoding='utf-8',newline='\n')
print(json.dumps({'files':len(files),'prior':P['prior']}))
