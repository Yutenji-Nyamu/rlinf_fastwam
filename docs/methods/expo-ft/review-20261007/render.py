import html,json,os,statistics,sys
from pathlib import Path
P=Path('E:/Codex/home/visualizations/2026/10/02/01a0fc88-6d9d-7131-932a-4b2852e69b58/expo-review-20261007')
sys.path.insert(0,'E:/Codex/home/visualizations/2026/09/14/01a09e67-ffad-7650-8664-f90f50a1e1ec/experiment-refresh-20260918-1420/plot-deps-0919')
os.environ['MPLCONFIGDIR']=str(P/'mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties,fontManager
fontManager.addfont('C:/Windows/Fonts/msyh.ttc');font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
s=json.loads((P/'summary.json').read_text());eps=json.loads((P/'episodes.json').read_text());learn=json.loads((P/'learning.json').read_text());frames=json.loads((P/'frame-index.json').read_text())
T='#087f8c';O='#d57922';G='#71808b';L='#d8e0e4';D='#233c46'
fig,axs=plt.subplots(2,2,figsize=(14,9));fig.subplots_adjust(left=.07,right=.96,top=.83,bottom=.12,hspace=.53,wspace=.23)
fig.text(.07,.952,'2机 EXPO｜两卡运行正常，续训收益仍待确认',fontsize=21,weight='bold',color=D)
fig.text(.07,.90,'10月7日 11:52  ·  28,227 / 60,000 动作（47.0%） · 196回合 · 655次学习已保存 · GPU4–5',fontsize=12,color=G)
for ax in axs.flat:ax.grid(axis='y',color=L,lw=.7);ax.set_axisbelow(True)
a,b,c,e=axs.flat
evals=[x for x in s['evaluations'] if x['episode']>0]
a.axhline(45,color=O,ls='--',lw=1.4,label='原始单候选基座 9/20')
a.plot([x['episode'] for x in evals],[100*x['rate'] for x in evals],color=T,marker='o',lw=2,label='完整EXPO · 固定20场')
for x in evals:
 if x['episode'] in (25,50,110,128,160,190):a.annotate(f"{x['successes']}/20",(x['episode'],100*x['rate']),xytext=(0,9 if x['episode']!=128 else -20),textcoords='offset points',ha='center',fontsize=10,color=T)
for x in (128,174):a.axvline(x,color=G,ls=':',lw=1)
a.set(title='固定评估：80% → 65% → 60% → 65%',xlabel='已完成在线回合',ylabel='成功率（%）',ylim=(0,105),xlim=(0,202));a.legend(loc='lower right',frameon=False,fontsize=9)
x=[r['episode'] for r in eps]
for n,color,lw,alpha in [(5,'#aebbc1',1,.75),(15,'#86bdc4',1,.8),(10,O,1.7,1),(20,T,2,1)]:
 b.plot(x,[100*r['ma'+str(n)] if r['ma'+str(n)] is not None else float('nan') for r in eps],color=color,lw=lw,alpha=alpha,label=f"滑动{n}：{100*eps[-1]['ma'+str(n)]:.0f}%")
b.set(title='在线采集：最近20回合12成功 / 20',xlabel='在线回合',ylabel='成功率（%）',ylim=(0,105),xlim=(0,202));b.legend(loc='lower right',frameon=False,ncol=2,fontsize=9)
times=[r['learner_seconds']/60 for r in learn];calls=[r['call'] for r in learn]
c.plot(calls,times,color=L,lw=.8)
c.plot(calls,[statistics.mean(times[max(0,i-19):i+1]) for i in range(len(times))],color=T,lw=2,label='最近20次平均')
c.axvline(595.5,color=O,ls='--',lw=1.5,label='4卡→2卡');c.annotate('当前9.2分钟',xy=(655,9.19),xytext=(470,11.8),arrowprops=dict(arrowstyle='-',color=T),color=T)
c.set(title='学习耗时：两卡连续完成60次更新',xlabel='完整学习调用',ylabel='分钟 / 次',xlim=(0,680),ylim=(0,14));c.legend(loc='lower right',frameon=False,fontsize=9)
last=eps[-50:]
for success,col,label in [(0,G,'失败'),(1,T,'成功')]:
 z=[r for r in last if r['success']==success];e.scatter([r['episode'] for r in z],[r['actions'] for r in z],color=col,s=24,label=label,zorder=3)
e.axhspan(0,49,color='#fff1e4');e.axhline(50,color=O,ls='--',lw=1.2,label='FM真实窗口 H50')
e.set(title='短成功覆盖：近20回合7条成功短于H50',xlabel='在线回合（最近50）',ylabel='真实动作数',xlim=(145,198),ylim=(0,215));e.legend(loc='center left',frameon=False,fontsize=9)
fig.text(.07,.053,'固定评估竖线：128回合续至60k；174回合改两卡。在线采集种子与固定评估不同。',fontsize=10,color=G)
fig.text(.07,.025,'原始基座与EXPO的候选机制不同；20个固定种子是小样本，曲线不能单独证明训练或编辑器的贡献。',fontsize=10,color=G)
fig.savefig(P/'overview.png',dpi=145);fig.savefig(P/'overview.svg');plt.close(fig)
(P/'overview.svg').write_bytes(('\n'.join(line.rstrip() for line in (P/'overview.svg').read_text(encoding='utf-8').splitlines())+'\n').encode())

ev=s['evaluations'];seeds=ev[0]['contract']['seeds'];matrix=[[int(next(r['success_once'] for r in v['rows'] if r['seed']==seed)) for v in ev] for seed in seeds]
fig,ax=plt.subplots(figsize=(12,8));fig.subplots_adjust(left=.17,right=.96,bottom=.13,top=.86)
ax.imshow(matrix,cmap=ListedColormap(['#f0d4ba',T]),vmin=0,vmax=1,aspect='auto',interpolation='nearest')
ax.set_xticks(range(len(ev)),['原始' if v['episode']==0 else str(v['episode']) for v in ev],rotation=0)
ax.set_yticks(range(20),[str(x) for x in seeds]);ax.set(xlabel='固定评估对应的在线回合（128为20k终评）',ylabel='场景种子')
fig.text(.17,.955,'固定场景逐项结果：通过与失败仍在互换',fontsize=19,weight='bold',color=D)
fig.text(.17,.906,'深青=成功，浅橙=失败；第160→190回合新增2个成功，同时失去5个原成功。',fontsize=11,color=G)
for line in range(21):ax.axhline(line-.5,color='white',lw=.7)
for line in range(len(ev)+1):ax.axvline(line-.5,color='white',lw=.7)
fig.savefig(P/'seed-grid.png',dpi=140);plt.close(fig)

fig,axs=plt.subplots(2,4,figsize=(13,6.2));fig.subplots_adjust(left=.025,right=.98,top=.82,bottom=.09,hspace=.36,wspace=.04)
for row,axes in zip(frames,axs):
 for item,ax in zip(row['frames'],axes):
  ax.imshow(plt.imread(P/item['file']));ax.set_title(f"回合{row['episode']} · 动作{item['step']}",fontsize=10);ax.axis('off')
fig.text(.025,.95,'真实在线回放：成功39步，失败200步',fontsize=19,weight='bold',color=D)
fig.text(.025,.884,'上：第194回合，环境判成功；下：第196回合，200动作超时。不是同场景配对实验。',fontsize=11,color=G)
fig.text(.025,.028,'原始回放的文件SHA及逐帧哈希已核对；关键帧只展示观察，不足以单独定位失败根因。',fontsize=10,color=G)
fig.savefig(P/'behavior.png',dpi=140);plt.close(fig)

tr=lambda a,b:f'<tr><th>{a}</th><td>{b}</td></tr>'
evrows=''.join(f'<tr><td>{v["episode"] or "原始基座"}</td><td>{v["successes"]}/20</td><td>{v["seconds"]/60:.1f}分钟</td></tr>' for v in ev)
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EXPO · 10月7日现场与产物</title>
<style>body{{max-width:1100px;margin:32px auto;padding:0 22px;font:16px/1.7 "Microsoft YaHei",sans-serif;color:#233c46;background:#fafcfc}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #d8e0e4;vertical-align:top}}th{{width:22%}}.small{{color:#687c85;font-size:13px}}.card{{background:white;border:1px solid #d8e0e4;border-radius:10px;padding:18px;margin:16px 0}}code{{overflow-wrap:anywhere}}a{{color:#087f8c}}li{{margin:8px 0}}@media(max-width:600px){{body{{padding:0 12px}}h1{{font-size:23px}}td,th{{padding:6px;font-size:12px}}}}</style>
<h1>2机 EXPO：两卡运行正常，继续训练的增益尚未体现</h1><p class="small">现场：2026-10-07 11:52，北京时间。只读审计，未启停任务、调参或新增GPU评估。</p>
<div class="card"><b>196回合 · 28,227/60,000动作（47.0%） · 655次学习已保存，第656次进行中。</b><p>GPU4–5运行，owner3008973与driver3503441身份匹配，心跳21秒，无新失败回执，17份运行源码逐文件核同。两卡已连续完成60次更新，以及180/190回合各20场完整固定评估。</p></div>
<img src="overview.png" alt="固定评估、在线滑动成功率、学习耗时与轨迹长度">
<h2>如何理解目前效果</h2><ul><li>原始单候选基座9/20；20k结束完整EXPO为15/20。续训后最高16/20，最近160/170/180/190回合分别16/13/12/13（/20）。目前没有继续超越此前最好成绩。</li><li>第170回合已降至65%，早于174回合减卡，不能把下降归因于两卡。硬件切换可能改变后续随机数及浮点归约，但不是现有曲线的因果解释。</li><li>最新对20k终评：2个种子失败→成功，4个成功→失败；对160回合：2个改善、5个退步。最近三次一直失败的仅2个种子，其余失败在变化；更像成功覆盖尚不稳定，原因仍需模块评估。</li><li>在线累计100/196，近5/10/15/20回合分别40%/60%/53.3%/60%。采集种子、停止规则和固定评估不同，不能直接把两条曲线相减。</li></ul>
<img src="seed-grid.png" alt="20个固定种子逐场成败热图">
<h2>算力、时间与配置</h2><table>{tr('显存 / 内存','GPU4约76.2GiB、GPU5约65.3GiB；主卡余量约3GiB。进程RSS35.8GiB、峰值38.8GiB、swap0；主机可用约1.72TiB。驻留/缓存不能直接等同泄漏。')}{tr('实际GPU范围','EXPO的计算与图形仅4–5；0–3未见进程上下文，底层读数各4MiB。6/7已由Q-U/Q-Norm使用。')}{tr('吞吐','最近20次551.4秒/学习调用；切换前26次561.6秒。两卡刚切换20次603.6秒；负载/缓存变化下的现场比较，不是严格性能基准。按占卡数×耗时算，该学习阶段GPU秒约减半。')}{tr('时间结构','取完整第176–195回合区间：学习78.8%，额外固定评估9.9%，采集动作阶段3.0%，保存/reset等8.2%。只加速采集的收益有限。')}{tr('剩余粗估','近期2285动作/11.84小时，约193动作/小时；若速度保持，剩31,773动作约165小时（约7天）。成功轨迹长短、评估和后续变速会明显改变估计。')}{tr('原配置保持','N1采集；N4×5评估20场，每10回合；B64；8原+8编辑候选；Q20、FM/editor/温度各1；每40动作积学习额度；C10/H50/ODE10；编辑幅度0.2；latest/last1滚动保存。')}{tr('磁盘','根盘约66.5%已用、余30.1GiB；数据盘余6.06TiB，本轮未清理。')}</table>
<h2>产物已核对到什么程度</h2><table>{tr('Checkpoint','latest/last1各8.79GB，ZIP目录均可读，latest回执655次更新、finite=true；没有额外加载8.8GB模型重做恢复。减卡前595次学习的两份备份也保留。滚动保存并不保留每10回合历史最优权重。')}{tr('在线回放','196条、28,227动作，约19.69GB；392个数据/manifest文件身份全部匹配。全部内容未重复读盘哈希，另抽取194/196两条做完整文件SHA与逐帧校验。')}{tr('示范与训练窗口','50条既有成功示范、4863帧；在线Q窗口26,367、FM窗口4,449。100条在线成功中22条不足H50，不能进入当前完整H50的FM；近20回合12条成功中7条被排除。它们仍进入Q回放。')}{tr('评估与日志','16份固定评估（含初始、20k终评及历次评估），种子hash一致；655条学习记录与196条在线成败可重绘。所有记录的数值指标有限，冻结prefix没有梯度。')}{tr('行为素材','当前run没有mp4；真实回放保留三相机图像/动作/终止标志。本页展示两条主相机关键帧；图像不是新跑的评估。')}{tr('云端与本地','执行源码和此前切换证据已发布；本次图、统计、产物审计作为轻量证据补同步。checkpoint/回放及完整原始日志仍留服务器。')}</table>
<img src="behavior.png" alt="成功与失败轨迹关键帧">
<p class="small">第194回合指令：{html.escape(frames[0]['prompt'])}。第196回合：{html.escape(frames[1]['prompt'])}。两条场景/指令不同，不能从截图判定整体失败类型比例。</p>
<h2>我建议讨论的三个方向</h2><ol><li><b>先搞清收益来自谁。</b>同固定种子分开测原始基座、训练后基座、基座+Q、完整EXPO；解释编辑器贡献时加入相同候选总数的控制。最近20回合58.6%的chunk选中编辑候选，只说明经常使用，不能证明带来收益。</li><li><b>检查快成功样本没有进入FM的问题。</b>近20回合7/12成功短于H50，当前FM更多从长成功窗口学习；最近20次FM约63.9%样本来自在线成功。优先报告按轨迹长度的覆盖，再单独讨论只对真实动作监督的方案，不能伪造成功后的动作或顺手裁短基座H50。</li><li><b>效率先查学习内部。</b>学习占79%，代码回放缓存仍仅1条轨迹，且同观测的8个候选重复处理前缀。优先分段计时/缓存命中、显存峰值，再做保持采样和更新预算的缓存或分块实验；先不同时改LR、Q20和环境数。</li></ol>
<p>α已从1升至1.227，实际编辑熵约−129.69，接近当前±0.2范围的上限−128.28；目标仍−70。这个尺度不匹配仍存在，适合作独立对照，尚无证据证明它导致近期成功率下降。Q/FM/编辑器梯度均非零有限，低Q loss也不等于任务已学好。</p>
<h2>资源接续的新边界</h2><p>当前唯一协调器为<code>/data/chenyiteng/deployment-20261006/rlt-q-signals-g67-v1/coexist/current.json</code>。旧RLT现在等待<b>EXPO和Q-U/Q-Norm全部释放</b>；代码还要求<code>rlt-return-validation.json</code>的计算/图形绑定验证，本次现场该回执尚不存在。因此不能再表述成“EXPO一结束RLT立即顶上”。本轮仅指出这个待办，未抢占其他任务或修改归还链。</p>
<details><summary>全部固定评估数字</summary><table><tr><th>回合</th><th>成功</th><th>评估时间</th></tr>{evrows}</table></details>
<p class="small">证据：<a href="summary.json">统计与产物索引</a> · <a href="episodes.json">在线逐回合数据</a> · <a href="learning.json">学习标量</a> · <a href="overview.svg">矢量总览</a>。完整本地现场snapshot/detail不作为训练修改。</p></html>'''
(P/'review.html').write_bytes(body.encode())
print(json.dumps({'plots':['overview.png','seed-grid.png','behavior.png'],'report':str(P/'review.html')}))
