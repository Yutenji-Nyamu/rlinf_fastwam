import json,html,re
from pathlib import Path
W=Path('C:/Users/86136/Documents/rl');E=Path('E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a');O=E/'dojo_u_1020'
m=json.loads((O/'manifest.json').read_text())
tasks=[line.split('|')[1]for line in(W/'docs/dojo_u_video/05_TASK_BUDGETS.md').read_text(encoding='utf8').splitlines()if re.match(r'^\|[a-zA-Z_]+\|\d',line)]
assert len(tasks)==54
rows=[]
for task in tasks:
 s=m['tasks'].get(task)
 history=[r for r in m['runs']if r.get('result')and r['result'].get('task')==task]
 if s:state='有已封装回合'
 elif history:state='已尝试/跳过，尚无完整回合；部分已补排'
 else:state='尚无完整回合/等待队列'
 rows.append({'task':task,'episodes':s['episodes']if s else 0,'successes':s['successes']if s else 0,'status':state,'has_gallery':bool(s)})
(O/'all-task-coverage.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
detail='<details><summary>全部54任务覆盖情况：10项有图，44项暂无完整回合</summary><table><tr><th>任务</th><th>已封装</th><th>成功</th><th>状态</th></tr>'
for r in rows:detail+='<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>'for k in ['task','episodes','successes','status'])+'</tr>'
detail+='</table></details>'
p=O/'index.html';s=p.read_text(encoding='utf8')
s=s.replace('<input id="q"',detail+'<input id="q"')
p.write_text(s,encoding='utf8')
print('All 54 task coverage states added.')
