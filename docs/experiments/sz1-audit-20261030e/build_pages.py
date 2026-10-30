import json,re,csv,shutil,tarfile
from pathlib import Path
from datetime import datetime,timezone,timedelta
L=Path(__file__).parent;V=L.parent
W=Path('C:/Users/86136/Documents/rl/worktrees/pi05-distill-20261021')
e=json.loads((L/'evidence/experiments-refresh1030f.json').read_text())
h=json.loads((L/'evidence/health-latest1030f.out').read_text())
stamp=datetime.fromtimestamp(e['time'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
rows=json.loads((V/'review1030a/points.json').read_text())
points={(r['protocol'],r['method'],r['step']):r for r in rows}
names={'endpoint':'Direct','cp':'CP','onedp':'DMD new','mfd':'MFD','cp_v2':'CP v2','dmd_v2':'DMD v2','mfd_v2':'MFD v2','fql_fixed_teacher':'FQL'}
for n,f in e['families'].items():
    native=n.startswith(('single-formal-','single-extend200-','dmd-extend200-','mfd-extend200-','physical-v2-formal-','single-extend400-','v2-fqlfix400-'))
    multi=n=='endpoint-multitask-c10-sz1-g45-20261025'
    if not(native or multi):continue
    for a in f['summaries'].values():
        if native:
            if not a.get('checkpoint') or a.get('method') not in names:continue
            method=names[a['method']];step=a['checkpoint']['step'];protocol='Native single task';complete=a['complete']
        else:
            match=re.fullmatch(r'endpoint-s(\d+)',a.get('condition',''))
            if not match:continue
            method='Direct';step=int(match[1]);protocol='C10 multitask';complete=a.get('attempted_tasks')==50
        r=dict(protocol=protocol,method=method,step=step,successes=a['successes'],episodes=a['episodes'],rate=100*a['successes']/a['episodes'] if a['episodes'] else 0,time=a['time'],complete=complete)
        points[(protocol,method,step)]=r
rows=sorted(points.values(),key=lambda r:(r['protocol'],r['method'],r['step']))
for name in ['review1030e','single1030e']:
    d=V/name;d.mkdir(exist_ok=True)
    (d/'points.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    with (d/'points.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
t=(V/'single1030a/index.html').read_text(encoding='utf-8')
t=re.sub(r'(<script id="source-data" type="application/json">).*?(</script>)',lambda m:m[1]+json.dumps(rows,ensure_ascii=False)+m[2],t,flags=re.S)
t=t.replace('2026-10-30 00:15',stamp).replace('../review1030a/','../review1030e/').replace('DMD v2已纳入；MFD v2尚未启动。','Direct已完成400轮；CP v2与DMD v2续400推进中，MFD v2已完成200轮。')
(V/'single1030e/index.html').write_text(t,encoding='utf-8')
(L/'plot_input.json').write_text(json.dumps(dict(rows=rows,stamp=stamp)),encoding='utf-8')
latest={}
for m in names.values():
    rr=[r for r in rows if r['protocol']=='Native single task' and r['method']==m and r['complete']]
    if rr:latest[m]={'latest':max(rr,key=lambda r:r['step']),'best':max(rr,key=lambda r:r['rate'])}
queue45=[('DMD v2，225→400轮','5–7 h'),('MFD v2，200→400轮','9–11 h'),('FQL-fix，从基座→400轮','15–20 h'),('逐卡RLT','候补')]
queue67=[('CP v2，277→400轮','3.5–4.5 h'),('DMD new，200→400轮','6–8 h'),('MFD，200→400轮','9–11 h'),('逐卡RLT','候补')]
table=''.join(f"<tr><td>{m}</td><td>{v['latest']['rate']:.1f}%（{v['latest']['successes']}/{v['latest']['episodes']}）</td><td>{v['best']['rate']:.1f}%</td></tr>" for m,v in latest.items() if m!='FQL')
def qtable(q):return '<table><tr><th>顺序</th><th>该段剩余</th></tr>'+''.join(f'<tr><td>{m}</td><td>{eta}</td></tr>' for m,eta in q)+'</table>'
style='body{font:16px/1.65 system-ui,"Microsoft YaHei";max-width:1160px;margin:30px auto;padding:15px;color:#234;background:#f4f7fa}section{background:white;padding:22px;margin:18px 0;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}img{max-width:100%;height:auto}a{color:#0674ad}.muted{color:#576774}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:700px){.cols{grid-template-columns:1fr}}'
body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>深圳1 · {stamp}</title><style>{style}</style>
<h1>深圳1 · {stamp} CST</h1><nav><a href="../single1030e/">勾选同族曲线</a> · <a href="#queue">完整队列</a> · <a href="../signal1030e/">分散度与师生距离</a> · <a href="../review1028d/">历史GRPO / EXPO / 接管</a></nav>
<section><h2>正在运行与新完成</h2><table><tr><th>卡</th><th>任务</th><th>现场进度</th><th>本段剩余</th></tr><tr><td>45</td><td>DMD v2 续400</td><td>2930/5200；225/400轮</td><td>5–7小时</td></tr><tr><td>67</td><td>CP v2 续400</td><td>8310/12000；277/400轮</td><td>3.5–4.5小时</td></tr></table>
<p><b>45卡交接故障已修复：</b>220轮评估后，67卡评估进程在4卡的图形上下文触发旧队列“卡未清空”断言，RLT临时接管。现在改为有界等待、保护无关进程；从2860完整点已推进至2930步，保留原配置和评估历史。34项CPU检查通过。</p>
<p><b>单任务Direct已完成400轮：</b>105/128＝82.0%；此前最佳108/128＝84.4%。后半程没有继续明显提高。<b>CP v2最新完整260轮116/128＝90.6%</b>，仍低于100轮最佳119/128＝93.0%。DMD v2最新220轮104/128＝81.3%。</p>
<a href="../single1030e/"><img src="native.png" alt="单任务蒸馏同族成功率"></a><p><a href="../single1030e/">打开可勾选曲线</a>；横轴为训练轮数，每点评128回合；原始N5 86.7%、原始N1 76.6%。失败FQL保留点表，主图隐藏。</p>
<details><summary>各组最新与最好成绩</summary><table><tr><th>单任务方法</th><th>最新完整评估</th><th>最好</th></tr>{table}</table></details></section>
<section><h2>多任务同族</h2><p>多任务Direct C10已完成50预算轮（55650更新），末评<b>141/368＝38.3%</b>；50任务全部尝试，46有效、4仿真错误缺测。原始N10 41.6%、N1 20.4%；比原始N10低3.3个百分点。与原生单任务分开看。</p><img src="c10.png" alt="C10多任务同族"><details><summary>C50历史同族（另一协议）</summary><img src="c50.png"></details><a href="points.csv">全部精确点CSV</a></section>
<section id="queue"><h2>全部排队任务</h2><div class="cols"><div><h3>45卡</h3>{qtable(queue45)}<p>合计约29–38小时。</p></div><div><h3>67卡</h3>{qtable(queue67)}<p>合计约19–24小时。</p></div></div><p class="muted">含训练、原定128回合评估和重载；依据近期实测步时及评估周期。跨卡图形上下文等待、仿真异常和容量不足会延长时间。旧CP400未添加；FQL-fix只在45卡执行。</p><p><b>四路RLT正常等待：</b>4/5/6/7完整恢复点250/250/25/300；唯一调度器存活，心跳新鲜，检查点、实配、环境文件均存在。45本轮让位前保存到250。实验方法失败后清理自己的进程再接下一项，最后逐卡RLT。</p></section>
<section><h2>服务器</h2><p>19:14快照：0/1为其他用户Python作业，约15.9/16.0 GiB，仅只读检查；2/3空闲。4/5我方DMD v2约75.3 GiB/卡，6/7 CP v2约77.2 GiB/卡；训练/保存阶段利用率会波动。最高72℃，无热降频。</p><p>内存可用<b>{h['memory_gib']['MemAvailable']:.0f}/{h['memory_gib']['MemTotal']:.0f} GiB</b>，load 9.29，CPU/内存/IO压力接近0。Home余<b>{h['disk']['/home']['free_gib']:.1f} GiB</b>，Data余<b>{h['disk']['/data']['free_gib']:.1f} GiB</b>，根盘214.5 GiB；本轮未清理用户产物。</p><p>近6小时内核检查未见OOM、Xid、磁盘错误；常见挖矿进程名/端口筛查无命中，这不是全面安全审计。网卡短采样无新增错误/丢包；网络访问有波动，既有17890代理GitHub/HF均可达。其他用户任务与共享Ray未修改。</p><p>后续新点仍用最近2个完整点＋本段最佳，原200轮保留。容量预检不够时，该方法留失败记录并转后续；同盘均不足时，后续也可能无法保存，不能把“能切换”理解为解决容量。</p></section>
<section><h2>公开归档核对</h2><p>核对SZ1 10月21日起56个运行/部署目录；补齐训练指标、逐次评估成绩、失败日志、配置、部署源码与交接回执。共43432份轻量文件，压缩19.5 MiB。权重、回放、图像、完整视频与逐物理步控制大包排除。正在运行的目录是本次快照。</p><p><a href="https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/pi05-fql-fix-shadow-20261029/docs/experiments/sz1-audit-20261030e">归档与执行说明</a> · 2/3机不可达时期产物不作本轮已核实保证。</p></section></html>'''
(V/'review1030e/index.html').write_text(body,encoding='utf-8')
summary=dict(stamp=stamp,queue45=queue45,queue67=queue67,health={k:h[k] for k in ['disk','memory_gib','load','pressure']},native_methods=latest,rlt=e['rlt']['checks'],recovery=dict(from_step=2860,verified_step=2930,tests_passed=34))
(V/'review1030e/status.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
S=V/'signal1030e';S.mkdir(exist_ok=True)
signal=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>分散度与师生距离 · 复看</title><style>{style}</style><h1>先分清：学生内部不一致，还是学生与老师不一致</h1><p><a href="../shadow1029f/">原图册：12条成功轨迹＋9种统计图</a> · <a href="../slot25_1029g/">slot25视频、12帧与可编辑图</a> · <a href="../review1030e/">服务器与蒸馏</a></p>
<section><h2>1 · 局部确实能看到，但不是处处重合</h2><p><b>蓝线</b>：学生用4种噪声推理，动作有多分散。<b>橙线</b>：同一观察、同一主噪声下，学生与老师动作差多少。蓝线并非严格的概率熵，橙线也不是动作对错。</p><a href="../slot25_1029g/"><img src="../slot25_1029g/signal-and-12-frames.png" alt="slot25信号和真实画面"></a><p>slot25成功，70–80附近两线同升，ρ＝0.430；10–18附近蓝线升高，橙线变化小。原32条中12条成功，单看好看的轨迹不能代表整体。</p></section>
<section><h2>2 · 为什么混起来反而不清楚</h2><img src="../shadow1029f/phase-and-trajectories.png" alt="逐轨迹和阶段关系"><p>12条成功轨迹各自都正相关（0.09–0.55），合并后却只有0.014。原因之一是每条轨迹的基线不同：某条一直很分散，但与老师一直很接近，混合后会掩盖各自的同向变化。</p><p>动作块内越靠后，两量也可能共同变化。分块看相关约0.435，扣除块内位置趋势后约0.161；所以时间位置解释了部分关联。</p></section>
<section><h2>3 · 指标怎么定义，影响很大</h2><img src="../shadow1029f/relative-vs-absolute.png" alt="相对与绝对分散度"><p>原蓝线会除以动作本身的大小：动作很小时，小差异也可能被放大。橙线是绝对距离。补充改成<b>绝对分散度</b>后，整体相关从0.089到0.369，更清楚，但这是另一指标，不能悄悄替换原定义。</p><p>仅用未执行的另3个噪声，绝对相关0.368，因此共享主噪声不是全部原因。原指标整轨迹bootstrap 95%区间−0.038～0.209，本批尚不能确认稳定的总体正相关。</p></section>
<section><h2>是否重采：先把问题缩小</h2><table><tr><th>因素</th><th>优先处理</th></tr><tr><td>归一化与动作维度</td><td>相对/绝对同时报告；关节和夹爪分开，查是否少数坐标主导。</td></tr><tr><td>动作位置与场景混合</td><td>按轨迹、前3块/后段分组；扣除块内位置趋势。</td></tr><tr><td>实际执行与提交前缀</td><td>原5820是提交动作位置；提前成功后的未到达末帧应排除。slot25已从100截到97，ρ仍0.434。</td></tr><tr><td>样本量</td><td>只有32局、12成功；4噪声估分散度也有波动。更多独立种子能检验是否复现。</td></tr><tr><td>强模型的含义</td><td>老师成功率更高，也不意味着每个动作都正确；距离衡量分歧，不能直接解释成错误。</td></tr></table>
<p><b>建议：</b>先用保存的数组/映射做前三项，不改推理；若补采，保持Clean CP140→Uτ3 CP70、N10/H50/C20，补96条新种子到总128（32并行×3批），成败均纳入。不要为了图更明显改任务、挑成功或改变老师。需要比较4/8噪声时另作同轨迹重算，避免同时改变执行。</p><p class="muted">本轮复看现有图与统计，未启动新采集。上述分项及补采是后续方案；未把尚未完成的分项分析写成结果。</p></section></html>'''
(S/'index.html').write_text(signal,encoding='utf-8')
A=W/'docs/experiments/sz1-audit-20261030e';A.mkdir(exist_ok=True)
for n in ['distillation-reviewed-light-20261030.tar.gz','distillation-reviewed-summary-20261030.json']:
    shutil.copy2(L/'evidence'/n,A/n)
for src,dst in [('repair-validate1030e.out','repair-tests.txt'),('repair-activate1030e.out','repair-receipt.json'),('experiments-refresh1030f.json','experiment-snapshot.json')]:
    shutil.copy2(L/'evidence'/src,A/dst)
for n in ['points.json','points.csv','status.json']:shutil.copy2(V/'review1030e'/n,A/n)
(A/'README.md').write_text(f'''# 深圳1蒸馏归档与续训恢复 · {stamp} CST

## 核对与补归档

公开目标：Yutenji-Nyamu/rlinf_fastwam，既有codex/pi05-fql-fix-shadow-20261029分支。
覆盖SZ1 2026-10-21日起56个运行/部署目录（含smoke和当前部分运行），43432份轻量文件，tar.gz约19.5 MiB。
既有Git中已保存方法源码、历史点表和快照；本次补齐现场训练指标、每次评估、失败日志、实配和部署源码快照。包内manifest给出原路径、时间、文件大小及排除口径，没有额外逐项SHA核验。
不含模型/恢复权重、回放、图像、完整视频、逐物理步控制大包；6336份逐物理步控制记录约2.9 GiB保留在服务器。运行中日志为采集时刻快照，不承诺覆盖未来输出。2/3机历史产物本轮未联网核实。

## 修复与现场验收

g45 DMDv2完成220轮/2860更新评估104/128后，旧队列因g67评估的跨卡G上下文触发GPU-empty断言，清理同样断言使队列退出。RLT临时运行至250/250。
修复仅改交接等待及同配置中断恢复；不杀其他工作，不改变损失/预算/评估。新部署v2-fqlfix400-sz1-g45-recover1复用旧输出，完整恢复2860，已见2930更新、有限损失和5.78秒/更新；34项CPU检查通过。保留已有220轮评估及最佳选择。
45队列：DMDv2→400、MFDv2→400、FQL-fix从原始基座→400、RLT。67队列：Direct400已完成、CPv2→400、DMDnew→400、MFD→400、RLT。旧CP400取消。RLT250/250/25/300检查点/环境/实配齐全，唯一owner和四个WAIT_EXTERNAL均正常。

## 新完成和曲线

单任务Direct400：105/128=82.03%，最高108/128=84.375%；CPv2最新260轮116/128=90.63%，历史最好119/128=92.97%。
多任务Direct C10完成50轮：141/368=38.315%，50任务尝试、46有效、4仿真错误；原始N10为153/368=41.576%，N1为75/368=20.38%。
本目录点表含所有已记录同族完整点，原生单任务/C10多任务/C50历史分开。当前动态见status.json；图表横轴按训练轮数。

## 本地网页

- http://127.0.0.1:8194/review1030e/ 全状态与队列
- http://127.0.0.1:8194/single1030e/ 可勾选同族曲线
- http://127.0.0.1:8194/signal1030e/ 信号三图复看

信号仍是原32局12成功；本轮未新增采集。详细解释见docs/student_teacher_signal/08_REVIEW_AND_COLLECTION_DECISION_20261030.md。
''',encoding='utf-8')
print(json.dumps({'stamp':stamp,'points':len(rows),'archive_mib':(A/'distillation-reviewed-light-20261030.tar.gz').stat().st_size/2**20,'output':str(A)}))
