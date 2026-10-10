"""Generate the compact static atlas index from batch receipts, without raw reload."""
import json,html,time,os
from pathlib import Path
O=Path('/data/chenyiteng/results/pi05-attn/20261010-v1')
def update():
 m=json.loads((O/'manifest.json').read_text());rows=[];summary=dict(validated_batches=0,recorded_episodes=0,successes=0,failed_batches=0,duplicate_seed_batches=0)
 unique_episodes=set()
 for p in m['configs']:
  c=json.loads(Path(p).read_text());d=Path(c['output']);state='排队';n=s=0
  if (d/'started.json').exists():state='采集中'
  if (d/'done.json').exists():state='已记录，待离线'
  if (d/'validation.json').exists():
   v=json.loads((d/'validation.json').read_text());state='已验收';n=v['episodes'];s=v['successes'];summary['validated_batches']+=1;summary['recorded_episodes']+=n;summary['successes']+=s
   unique_episodes.update((e['task'],e['actual']) for e in json.loads((d/'episodes.json').read_text()))
   if v['seed_duplicate_warning']:state='记录完整；实际seed重复';summary['duplicate_seed_batches']+=1
  if (d/'error.json').exists():state='失败';summary['failed_batches']+=1
  target=d.relative_to(O)
  link=f'<a href="{target}/index.html">{html.escape(c["task"])} / b{c["batch"]}</a>' if (d/'index.html').exists() else html.escape(c['task'])+' / b'+str(c['batch'])
  rows.append(f'<tr><td>{link}</td><td>{state}</td><td>{n}</td><td>{s}</td></tr>')
 summary['unique_task_seed_episodes']=len(unique_episodes)
 text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>π0.5 Attn E / Flow</title><style>body{font:16px system-ui;margin:32px auto;max-width:1150px;padding:0 18px;color:#172333}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#0057b8}</style><h1>π0.5 · Attn E / Flow</h1><p>B16/H50/M10 · 10/14/18层 × 8头 × 10轮 · 深圳3 GPU6/7 · 推理与离线分析</p><p>目标50任务×32条；已记录验收 '+str(summary['recorded_episodes'])+' 条（不同任务/实际seed组合 '+str(len(unique_episodes))+'），任务成功 '+str(summary['successes'])+' 条。实际seed重复批 '+str(summary['duplicate_seed_batches'])+'；失败批 '+str(summary['failed_batches'])+'。当前记录不构成训练收益结论。</p><table><tr><th>任务/批次</th><th>状态</th><th>完整轨迹</th><th>成功</th></tr>'+''.join(rows)+'</table><p>终止chunk物理执行前缀unknown；原raw保留可复算，不将失败/重复条目当作独立完成量。</p></html>'
 for name,value in [('index.html',text),('START.html',text),('summary.json',json.dumps(dict(time=time.time(),**summary),indent=2))]:
  tmp=O/(name+f'.tmp-{os.getpid()}');tmp.write_text(value);tmp.replace(O/name)
 print(json.dumps(summary))
if __name__=='__main__':update()
