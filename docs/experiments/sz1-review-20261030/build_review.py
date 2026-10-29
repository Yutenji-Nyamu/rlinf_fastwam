import csv,json,re,statistics,shutil,difflib
from pathlib import Path
from datetime import datetime,timezone,timedelta
L=Path(__file__).parent;V=L.parent;W=Path('C:/Users/86136/Documents/rl/worktrees/pi05-distill-20261021')
e=json.loads((L/'evidence/experiments-final.out').read_text());h=json.loads((L/'evidence/health-final.out').read_text());s=json.loads((L/'evidence/storage_queue.out').read_text());eta=json.loads((L/'evidence/eta-storage.out').read_text())
stamp=datetime.fromtimestamp(e['time'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
rows=json.loads((V/'review1029e/points.json').read_text());points={(r['protocol'],r['method'],r['step']):r for r in rows}
names={'endpoint':'Direct','cp':'CP','onedp':'DMD new','mfd':'MFD','cp_v2':'CP v2','dmd_v2':'DMD v2','mfd_v2':'MFD v2','fql_fixed_teacher':'FQL'}
for n,f in e['families'].items():
 if not n.startswith(('single-formal-','single-extend200-','dmd-extend200-','mfd-extend200-','physical-v2-formal-','single-extend400-')):continue
 for a in f['summaries'].values():
  if not a.get('checkpoint') or a.get('method') not in names:continue
  r=dict(protocol='Native single task',method=names[a['method']],step=a['checkpoint']['step'],successes=a['successes'],episodes=a['episodes'],rate=a['successes']/a['episodes']*100 if a['episodes'] else 0,time=a['time'],complete=a['complete'])
  points[(r['protocol'],r['method'],r['step'])]=r
rows=sorted(points.values(),key=lambda x:(x['protocol'],x['method'],x['step']))
review=V/'review1030a';single=V/'single1030a'
for d in [review,single]:
 d.mkdir(exist_ok=True)
 (d/'points.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 with (d/'points.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['protocol','method','step','successes','episodes','rate','time','complete']);w.writeheader();w.writerows(rows)
t=(V/'single1029e/index.html').read_text(encoding='utf-8')
t=re.sub(r'(<script id="source-data" type="application/json">).*?(</script>)',lambda m:m[1]+json.dumps(rows,ensure_ascii=False)+m[2],t,flags=re.S)
t=t.replace('2026-10-29 17:46',stamp).replace('../review1029e/','../review1030a/')
t=t.replace('DMD v2 / MFD v2尚无完整评估点时不画曲线。','DMD v2已纳入；MFD v2尚未启动。')
(single/'index.html').write_text(t,encoding='utf-8')
proof=json.loads(Path('C:/Users/86136/Documents/rl/docs/u_policy_switch/PROTECTED_CHECKPOINTS.json').read_text(encoding='utf-8'))
protected=json.dumps(proof);dependency=json.dumps(eta['fallbacks'])+json.dumps(s['queue_plan'].get('initial_receipts',{}))
candidates=[]
criteria=[('Home：旧单任务100轮点',lambda p:'/home/chenyiteng/results/pi05-distill/single-formal-sz1-g67-20261027/' in p and any('/'+m+'/' in p for m in ['endpoint','onedp','mfd']),'保留各200轮末代/最佳与400轮续训依赖；放弃原100轮点续训。'),('Data：闲置Dojo / OpenWAM下载权重',lambda p:p.startswith('/data/chenyiteng/projects/robodojo-openwam-sz1/checkpoints/'),'以后运行Dojo需重新下载，当前π0.5蒸馏与RLT不依赖这些文件。'),('Data：Direct C10早期评估点',lambda p:'/endpoint-multitask-c10-sz1-g45-20261025/endpoint/' in p and any('-'+z+'.pt' in p for z in ['005565','022260']),'留当前完整续训点及最佳38955，早期两点评估曲线保留；删除会失去这两点续训。')]
for title,pred,cost in criteria:
 rr=[r for r in s['records'] if pred(r['path']) and r['path'] not in s['open_files'] and r['nlink']==1]
 assert all(r['path'] not in protected and r['path'] not in dependency and '/protected-eval/' not in r['path'] for r in rr)
 candidates.append({'title':title,'gib':sum(r['allocated'] for r in rr)/2**30,'cost':cost,'files':rr})
manifest={'time':s['time'],'executed':False,'candidates':candidates,'excluded':'current complete checkpoints, queued resume points, protected-eval and registry, RLT dependencies, open files, shared hardlinks'}
(L/'cleanup-candidates.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
queue=[('DMD v2 → 200轮','运行中','1–2 h'),('MFD v2 → 200轮','待运行','11–14 h'),('Direct 200→400轮','待运行','5–6.5 h'),('CP v2 200→400轮','待运行','5–6.5 h'),('DMD new 200→400轮','待运行','6–8 h'),('MFD 200→400轮','待运行','9–11 h'),('FQL-fix → 200轮','待运行','9–12 h'),('逐卡RLT','候补','无固定截止预算')]
latest={}
for m in names.values():
 rr=[r for r in rows if r['protocol']=='Native single task' and r['method']==m and r['complete']]
 if rr:latest[m]={'latest':max(rr,key=lambda x:x['step']),'best':max(rr,key=lambda x:x['rate'])}
summary={'stamp':stamp,'health_time':h['time'],'disk':h['disk'],'memory_gib':h['memory_gib'],'load':h['load'],'gpus':h['gpus']['out'],'pressure':h['pressure'],'newly_completed':'CP v2 200 epochs, final 114/128, best119/128 at100epochs','native_methods':latest,'queue':queue,'rlt':{'cards':[4,5,6,7],'resume_steps':[225,225,25,300],'owner_alive':True,'status':'WAIT_EXTERNAL'},'cleanup_candidates':[{k:v for k,v in a.items() if k!='files'} for a in candidates]}
(review/'status.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(L/'plot_input.json').write_text(json.dumps({'rows':rows,'stamp':stamp}),encoding='utf-8')
native_rows=''.join(f"<tr><td>{m}</td><td>{a['latest']['rate']:.1f}% ({a['latest']['successes']}/128)</td><td>{a['best']['rate']:.1f}%</td></tr>" for m,a in latest.items() if m!='FQL')
qrows=''.join(f'<tr><td>{m}</td><td>{state}</td><td>{hours}</td></tr>' for m,state,hours in queue)
crows=''.join(f"<tr><td>{a['title']}</td><td>{a['gib']:.1f} GiB</td><td>{a['cost']}</td></tr>" for a in candidates)
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>深圳1 · {stamp}</title><style>body{{max-width:1160px;margin:26px auto;padding:0 18px;background:#f5f7fa;color:#213447;font:16px/1.65 system-ui,"Microsoft YaHei"}}h1{{font-size:26px}}h2{{font-size:21px}}section{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:20px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #e4e9ee}}a{{color:#006daf}}img{{width:100%;height:auto}}small{{color:#63758a}}nav{{display:flex;gap:18px;flex-wrap:wrap}}.pill{{display:inline-block;padding:6px 12px;background:#e9f2fa;border-radius:6px}}</style>
<h1>深圳1 · 实验、排队与容量</h1><p>{stamp} CST · 健康快照 {h['time']} · 本轮未新增删除</p>
<nav><a href="../single1030a/">① 单任务：勾选同族曲线</a><a href="#multi">② 多任务同族</a><a href="#queue">③ 完整队列</a><a href="#storage">磁盘清理建议</a><a href="../shadow1029f/#aggregate">九种信号统计</a><a href="../slot25_1029g/">slot25视频与12张图</a></nav>
<section><h2>当前运行与新完成</h2><table><tr><th>卡</th><th>工作</th><th>当前进度</th><th>剩余估计</th></tr><tr><td>4/5</td><td>50任务 Direct，C10</td><td>48,260 / 55,650（43.36 / 50轮）</td><td>15–17 h → RLT</td></tr><tr><td>6/7</td><td>单任务 DMD v2</td><td>2,110 / 2,600（162.3 / 200轮）</td><td>1–2 h → 下表队列</td></tr></table><p><b>新完成：CP v2 200轮，末代89.1%，最佳93.0%。</b>旧CP末代82.8%，这次末代高6.25个百分点；v2与旧版的训练预算/学习率历程也不同，暂不把全部差异归因于维度mask。DMD v2最新78.9%、最好85.2%，曲线有波动。</p><p>原生单任务参考：原始N5为86.7%，原始N1为76.6%。每点128回合。网页横轴均为训练轮数。</p><a href="../single1030a/"><img src="native.png" alt="单任务全部同族"></a><table><tr><th>方法</th><th>最新完整点</th><th>最好完整点</th></tr>{native_rows}</table></section>
<section id="multi"><h2>多任务同族</h2><p>当前Direct最新完整评估36.1%（133/368）；同协议原始N10为41.6%，N1为20.4%。当前后续训练尚未产生新评估点，不能把最新训练步误当成新成功率。</p><img src="c10.png" alt="多任务C10完整同族"><details><summary>历史C50同族（协议不同，单列）</summary><img src="c50.png" alt="历史C50同族"></details><p><a href="points.csv">全部精确点表CSV</a> · <a href="../review1028d/">历史GRPO / EXPO / 接管图册</a></p></section>
<section id="queue"><h2>6/7卡顺序与预计耗时</h2><table><tr><th>实验</th><th>状态</th><th>本段耗时</th></tr>{qrows}</table><p>全队列粗估48–60小时。按近期更新时间＋实际128回合评估约12分钟/点＋重载时间估算；不含异常重试和容量不足跳过。CP v2续400已替换旧CP续400。</p><p><b>RLT仅4–7卡：</b>四路等待正常；完整恢复点225 / 225 / 25 / 300，配置和环境文件存在。唯一CPU调度器已重载并能读取新登记；需当前v2、后续队列对应owner全部退出且目标卡空闲才接管。</p></section>
<section><h2>服务器</h2><p>0–3卡本次计算和图形进程快照均空；4–7卡为我方训练。其他用户只读查看，未见其GPU训练。活动卡约67–75 GiB显存/卡，最高温度约67℃。</p><p>RAM可用 {h['memory_gib']['MemAvailable']:.1f} / {h['memory_gib']['MemTotal']:.1f} GiB；load {h['load'].split()[0]}。CPU、内存、IO压力未见持续等待。根盘余 {h['disk']['/']['free_gib']:.1f} GiB。</p><p>网卡错误/丢包未增加；既有17890代理访问GitHub、HF成功，外网直连仍不稳定。最近6小时内核筛查未见OOM、GPU Xid、磁盘I/O错误；常见挖矿进程名/端口筛查无命中，这只是有限范围检查。</p></section>
<section id="storage"><h2>空间与推荐清理</h2><p><b>Home余 {h['disk']['/home']['free_gib']:.1f} GiB；Data余 {h['disk']['/data']['free_gib']:.1f} GiB。</b>未来400轮队列按“最近2个＋本段最佳”最多约130 GiB新权重，Home仍偏紧，另需一次保存的临时余量。</p><table><tr><th>推荐顺序</th><th>可释放</th><th>保留 / 代价</th></tr>{crows}</table><p>上述精确候选合计约 {sum(a['gib'] for a in candidates):.1f} GiB，尚未删除。另有Home旧Dojo真实视频约83 GiB可单独讨论；它们是原始素材，不能按可重建缓存直接删。UV共享硬链接和大量仍打开的Ray日志未计入推荐。</p><p><b>写满时会怎样：</b>后续400轮/FQL-fix队列新增保存空间预检；不足时记失败、清理本阶段自己的进程、继续下一方法。失败回执可落根盘临时目录，避免Home/Data写满连回执都写不了。若后续方法也没空间，会继续跳过，最后按卡接RLT；这不等于凭空增加空间。RLT自身仍需要能保存的空间。</p><p>13项针对性CPU测试通过。当前健康训练未重启；保护权重、最新完整续训点、RLT依赖与日志曲线均保留。<a href="cleanup-candidates.json">精确候选清单</a></p></section>
<section><h2>绘图材料</h2><p>本地8194网页服务已恢复。<a href="../shadow1029f/#aggregate">九种统计图及解释</a> · <a href="../slot25_1029g/">slot25完整视频、12阶段图与双曲线</a>。</p><p><a href="https://github.com/Yutenji-Nyamu/uaca_cv27_draw_bkp/tree/main/materials/pillbottle_slot25_shadow_20261030">bkp材料目录</a>包含可编辑图、信号表、选定轨迹的原始动作/观测/控制数据和短视频，以及总体关系对照。</p></section></html>'''
html=html.replace('最高温度约67℃','最高温度72℃，未触发热降频').replace('外网直连仍不稳定','GitHub直连本次成功，HF直连仍失败')
html=html.replace(h['time'],datetime.fromisoformat(h['time']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M CST'))
(review/'index.html').write_text(html,encoding='utf-8');shutil.copy2(L/'cleanup-candidates.json',review/'cleanup-candidates.json')
print(json.dumps({'stamp':stamp,'points':len(rows),'candidates':summary['cleanup_candidates'],'health':{k:summary[k] for k in ['disk','memory_gib','load']},'gpu':h['gpus']['out'],'http_codes':{k:h[k]['out'].splitlines()[:1] for k in ['http','hf_http']}},ensure_ascii=False))
