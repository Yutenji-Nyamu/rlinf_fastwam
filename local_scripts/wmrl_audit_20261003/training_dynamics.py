"""Create an offline audit from saved WMRL scalar and native-evaluation receipts.

No SSH, CUDA, experiment import, or experiment mutation is performed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path("E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077")
OUT = ROOT / "wmrl-audit-20261003"
os.environ["MPLCONFIGDIR"] = str(OUT / "_mpl_cache")
sys.path.insert(0, str(OUT / "_plot_deps"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALARS = OUT / "sz3/deep-v2.out"
EVAL = Path("E:/Codex/home/visualizations/2026/10/03/01a0ffb6-b309-7fa1-9ae0-b20e6d63a11c/three-server-refresh/sz3/wm-official-eval.out")
NAMES = Path("E:/Codex/home/visualizations/2026/10/01/01a0f6dd-5f6f-7462-b7df-25b90ed941d1/wm-rlt-review-20261002-1155/libero-task-names.json")
DATASET = OUT / "sz3/dataset.out"
DOC = Path(__file__).resolve().parents[2] / "docs/world-model/audit_20261003/training_dynamics.md"
deep = json.loads(SCALARS.read_text(encoding="utf-8"))
ev = json.loads(EVAL.read_text(encoding="utf-8"))
names = json.loads(NAMES.read_text(encoding="utf-8"))["tasks"]
dataset = json.loads(DATASET.read_text(encoding="utf-8"))
raw = deep["scalars"]
steps = np.arange(1, 140)
values = {}
for tag, rows in raw.items():
    assert [r["step"] for r in rows] == list(range(139)), (tag, "missing/duplicate steps")
    values[tag] = np.array([float(r["value"]) for r in rows])

def stats(v):
    finite = np.isfinite(v)
    return dict(count=len(v), finite_count=int(finite.sum()),
                mean=float(np.nanmean(v)), first10=float(np.nanmean(v[:10])),
                last10=float(np.nanmean(v[-10:])),
                minimum=float(np.nanmin(v)), maximum=float(np.nanmax(v)),
                nonfinite_rounds=steps[~finite].tolist())

def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [100*(center-radius), 100*(center+radius)]

analysis = dict(source_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for p in [SCALARS, EVAL, NAMES, DATASET]],
                snapshot_time=deep["time"], completed_runner_rounds=139,
                raw_step_to_round="round = raw TensorBoard step + 1",
                scalar_tags=len(raw), metrics={k: stats(v) for k,v in values.items()},
                checkpoints=[40,80,120], native_evaluated_checkpoints=[40,80],
                filtered_rounds=steps[values["rollout/loss_mask_fraction"] == 0].tolist(),
                effective_rounds=int(np.count_nonzero(values["rollout/loss_mask_fraction"])),
                nominal_trajectory_slots=139*512,
                nominal_chunk_slots=139*512*40,
                loss_enabled_chunk_slots_equivalent=float(values["rollout/loss_mask_fraction"].sum()*512*40),
                last10_loss_enabled_chunk_slots_per_round=float(values["rollout/loss_mask_fraction"][-10:].mean()*512*40),
                nominal_action_slots_upper_bound=139*512*320,
                measured_rollout_share_last10=float(values["time/generate_rollouts"][-10:].sum()/values["time/step"][-10:].sum()),
                evaluation={}, tasks=[], reset_dataset_inventory=dataset, limitations=[
                    "No continuous GPU or host memory time series in the 39 scalar tags; timing is not a memory-leak test.",
                    "Native evaluation uses 500 unique task/initial-state slots and policy seeds [42,43], not multiple training seeds.",
                    "Per-trial paired outcomes are absent in this saved receipt; no paired hypothesis test is claimed.",
                    "Wilson intervals are descriptive conditional binomial intervals, not across-seed uncertainty.",
                    "Loss-mask fraction combines post-done masking and reward-group filtering; it is not kept-group fraction.",
                    "env/return spans the fixed-length rollout; actor optimization masks after first done. These returns differ.",
                    "Task-5-excluded totals are a post-hoc failure localization and do not replace the full-suite score."])

for phase in ["original","cp40","cp80"]:
    p = ev["phases"][phase]
    r = p["evaluation-result.json"]
    assert r["ok"] and p["completed_unique"] == 500 and not p["conflicts"]
    assert r["successes"] == p["successes"]
    assert r["policy_seeds_by_rank"] == [42,43]
    analysis["evaluation"][phase] = dict(successes=p["successes"], trials=500,
        rate_percent=p["successes"]/5, wilson95_percent=wilson(p["successes"],500),
        success_at_end_percent=100*r["metrics"]["success_at_end"],
        ever_success_but_not_at_end_count=p["successes"]-round(500*r["metrics"]["success_at_end"]),
        policy_seeds=r["policy_seeds_by_rank"], environment_seed_base=r["environment_seed_base"],
        evaluation_contract=r["model_contract"])
for t in names:
    i = str(t["id"])
    a = dict(id=t["id"], language=t["language"])
    for phase in analysis["evaluation"]:
        a[phase] = ev["phases"][phase]["task_counts"][i]
        assert a[phase]["completed"] == 50
    a["cp40_delta_successes"] = a["cp40"]["successes"] - a["original"]["successes"]
    a["cp80_delta_successes"] = a["cp80"]["successes"] - a["original"]["successes"]
    analysis["tasks"].append(a)
analysis["without_task5"] = {p:sum(t[p]["successes"] for t in analysis["tasks"] if t["id"]!=5)
                             for p in analysis["evaluation"]}
(OUT / "metrics-analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

TEAL = "#12616b"
ORANGE = "#dc792b"
GRAY = "#7f8991"
LIGHT = "#e4e9e9"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
                     "axes.spines.top":False, "axes.spines.right":False,
                     "axes.titleweight":"bold", "axes.titlesize":13,
                     "axes.labelcolor":"#253f45", "text.color":"#253f45",
                     "xtick.color":"#42545a", "ytick.color":"#42545a",
                     "axes.edgecolor":"#aab9bd", "figure.facecolor":"white"})
fig, axes = plt.subplots(3,2,figsize=(16.4,13.8))
fig.subplots_adjust(left=.075,right=.975,bottom=.112,top=.895,hspace=.48,wspace=.29)
fig.suptitle("WMRL audit | 139 completed rounds; CUDA OOM in round 140", x=.075,y=.972,ha="left",fontsize=22,fontweight="bold")
fig.text(.075,.939,"LIBERO Goal / pi0.5 + frozen Wan     ·     Training proxy and native evaluation are different measurements",fontsize=12,color="#566b71")

def style(a, title, ylabel, train=False):
    a.set_title(title,loc="left",pad=12)
    a.set_ylabel(ylabel)
    a.grid(axis="y",color=LIGHT,linewidth=.7,zorder=0)
    a.set_axisbelow(True)
    if train:
        a.set_xlim(1,139)
        a.set_xticks([1,40,80,120,139])
        a.set_xlabel("Completed runner round (raw step + 1)")
        for x in [40,80,120]:a.axvline(x,color="#c3cccc",ls=":",lw=1,zorder=0)

def rolling(v): return np.convolve(v,np.ones(10)/10,mode="valid")
def line(a,tag,color,scale=1,label="10-round mean"):
    v=values[tag]*scale
    a.plot(steps,v,color=color,alpha=.24,lw=1)
    a.plot(steps[9:],rolling(v),color=color,lw=2.5,label=label)

a=axes[0,0]
style(a,"A  Proxy success rises; final reward stays low","WM proxy success once (%)",True)
line(a,"env/success_once",TEAL,100)
a.set_ylim(0,14)
a.text(.015,.96,"First 10: 0.86%   →   Last 10: 4.82%",transform=a.transAxes,va="top",fontsize=11)
a.text(.015,.845,"Mean full-rollout return: 0.00313 → 0.00352",transform=a.transAxes,va="top",fontsize=10,color="#61777c")
a.legend(loc="lower right",frameon=False,fontsize=10)

a=axes[0,1]
style(a,"B  Most collected slots carry no policy loss","Loss-enabled chunk slots (%)",True)
line(a,"rollout/loss_mask_fraction",ORANGE,100)
a.set_ylim(0,21)
a.text(.015,.96,"First 10: 2.82%   →   Last 10: 6.51%",transform=a.transAxes,va="top",fontsize=11)
a.text(.015,.845,"Combines post-done masking and group filtering",transform=a.transAxes,va="top",fontsize=10,color="#61777c")
a.scatter([4,5,6],[0,0,0],s=25,color=ORANGE,zorder=4)

a=axes[1,0]
style(a,"C  No native full-suite improvement","True LIBERO success (%)")
phases=["original","cp40","cp80"]
pos=np.arange(3)
once=np.array([analysis["evaluation"][p]["rate_percent"] for p in phases])
end=np.array([analysis["evaluation"][p]["success_at_end_percent"] for p in phases])
a.bar(pos-.18,once,width=.34,color=TEAL,label="Success once",zorder=3)
a.bar(pos+.18,end,width=.34,color=ORANGE,label="Success at end",zorder=3)
for x,y in zip(pos-.18,once):a.text(x,y+2,f"{y:.1f}",ha="center",fontsize=10)
for x,y in zip(pos+.18,end):a.text(x,y+2,f"{y:.1f}",ha="center",fontsize=10)
a.set_xticks(pos,["Original","CP40","CP80"])
a.set_ylim(0,110)
a.set_xlabel("500 trials each; same evaluation contract")
a.legend(loc="upper right",frameon=False,ncols=2,fontsize=9)

a=axes[1,1]
style(a,"D  Regression is concentrated in pushing the plate","Task ID / instruction")
y=np.arange(10)
d40=np.array([t["cp40_delta_successes"]*2 for t in analysis["tasks"]])
d80=np.array([t["cp80_delta_successes"]*2 for t in analysis["tasks"]])
a.barh(y-.17,d40,height=.32,color=ORANGE,label="CP40",zorder=3)
a.barh(y+.17,d80,height=.32,color=TEAL,label="CP80",zorder=3)
a.set_yticks(y,["0 Open middle drawer","1 Bowl → stove","2 Wine → cabinet","3 Bowl → top drawer","4 Bowl → cabinet","5 Push plate forward","6 Cheese → bowl","7 Turn on stove","8 Bowl → plate","9 Wine → rack"],fontsize=9)
a.set_ylabel("")
a.set_xlabel("Change from original (percentage points; 50 trials/task)")
a.set_xlim(-105,30)
a.axvline(0,color=GRAY,lw=1)
a.invert_yaxis()
a.legend(loc="lower left",frameon=False,ncols=2,fontsize=9)
a.text(-100,5,"−96 / −84 pp",va="bottom",fontsize=9,fontweight="bold",bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=1))
a.grid(False,axis="y");a.grid(axis="x",color=LIGHT,linewidth=.7)

a=axes[2,0]
style(a,"E  Rollout generation dominates elapsed time","Time per round (minutes)",True)
line(a,"time/generate_rollouts",TEAL,1/60,"Rollouts · 10-round mean")
line(a,"time/actor_training",ORANGE,1/60,"Actor update · 10-round mean")
a.set_ylim(0,25)
a.text(.015,.96,"Last 10: 90.5% of round time in rollout generation",transform=a.transAxes,va="top",fontsize=10)
a.legend(loc="center left",frameon=False,fontsize=9)

a=axes[2,1]
style(a,"F  Gradients are finite; no scalar explosion seen","Logged actor gradient norm",True)
line(a,"train/actor/grad_norm",TEAL)
a.set_ylim(0,3.45)
a.text(.015,.96,"Last 10: grad 1.710 · approx KL 0.00915 · clip 5.29%",transform=a.transAxes,va="top",fontsize=10)
a.text(.015,.84,"Learning rate = 5e−6 throughout; max logged grad = 2.764",transform=a.transAxes,va="top",fontsize=10,color="#61777c")
fig.text(.075,.035,"Light lines: individual rounds. Dotted lines: saved CP40 / CP80 / CP120. CP120 has no native evaluation yet.\nNo continuous VRAM/RAM series was recorded in these scalar logs; this figure cannot establish a memory-growth trend.",fontsize=10,color="#536c72",linespacing=1.6)
fig.savefig(OUT/"training-audit.png",dpi=150)
fig.savefig(OUT/"training-audit.pdf")
plt.close(fig)

m=analysis["metrics"]
selected=["env/success_once","env/return","rollout/rewards","rollout/loss_mask_fraction","train/actor/grad_norm","train/actor/approx_kl","train/actor/clip_fraction","train/actor/ratio","train/actor/lr","time/generate_rollouts","time/actor_training","time/step"]
metric_rows="\n".join(f"| `{k}` | {m[k]['first10']:.7g} | {m[k]['last10']:.7g} | {m[k]['minimum']:.7g}–{m[k]['maximum']:.7g} |" for k in selected)
task_rows="\n".join(f"| {t['id']} · {t['language']} | {t['original']['successes']}/50 | {t['cp40']['successes']}/50 | {t['cp80']['successes']}/50 | {t['cp40_delta_successes']*2:+d} / {t['cp80_delta_successes']*2:+d} |" for t in analysis['tasks'])
data_rows="\n".join(f"| {t['id']} · {t['language']} | {dataset['tasks'][t['language']]['initial']} | {dataset['tasks'][t['language']]['kir']} | {dataset['tasks'][t['language']]['initial']+dataset['tasks'][t['language']]['kir']} |" for t in analysis['tasks'])
doc=f"""# WMRL 训练动态与真实评估审计 · 2026-10-03

仅分析已有日志与只读刷新，不改变实验。完整输入为39个TensorBoard tag、每tag 139轮；真实轮次=`raw step+1`。最新完整训练轮139，第140轮采集已完、actor训练发生CUDA OOM；CP40/80已完成原生LIBERO评估，CP120尚未评估。图中不补造显存或内存曲线。

![训练审计]({OUT.as_posix()}/training-audit.png)

## 核心结果

- 原生500回合总体：原模型428/500=85.6%，CP40 386/500=77.2%，CP80 404/500=80.8%；两检查点均未超过原模型。这只覆盖已评40/80，不代表尚未评的CP120。
- **退化主要集中于推盘子任务**：48/50→0/50→6/50；分别损失48与42个成功，超过全套净损失42与24。其余9任务合计380/450→386/450→398/450；这是事后定位，不是剔除失败任务后重算主成绩。CP40额外退化的“开上层抽屉放碗”为11→2，CP80回到10。
- 真实`success_at_end`为76.8%→57.2%→69.2%；“曾成功但最后不成功”的回合数44→100→58，表明训练后保持最终成功状态也更弱。该指标属于相同固定320步评估，不是另一套试验。
- WM代理成功率前10→后10轮0.859%→4.824%；**全长rollout的平均相对回报仅0.003125→0.003516，几乎持平**。需区分任一时刻成功与最终帧/完整回报；不可把proxy上涨直接当作真实策略改进。
- 有效mask前10→后10轮2.823%→6.509%；后10平均每轮仅{analysis['last10_loss_enabled_chunk_slots_per_round']:.1f}/20,480个chunk槽位参与loss。它同时包含首done后的截断与整组奖励过滤，不能说“93.5%的组被过滤”。139轮中136轮正mask，仅第4/5/6轮整轮全过滤。
- 139轮有71,168个名义轨迹槽位、2,846,720个chunk槽位，mask折算累计{analysis['loss_enabled_chunk_slots_equivalent']:.1f}个有效chunk槽位（{100*analysis['loss_enabled_chunk_slots_equivalent']/analysis['nominal_chunk_slots']:.2f}%）。不是同数目的独立物理回合或有效优化更新。
- loss、grad、approx KL等训练标量均有限；最大记录轮汇总grad2.764。末10近似KL0.00915、clip5.29%、LR恒5e-6，没有日志层面的梯度数值爆炸证据。只有第4/5/6轮空mask的奖励/优势统计非有限；不能据此认定模型权重NaN。
- 末10轮采集占总时长{analysis['measured_rollout_share_last10']*100:.1f}%；actor训练102.8→99.1秒/轮，总轮长1003.8→1094.7秒，增量主要来自采集895.8→990.5秒。耗时不是显存/内存的测量，不能用它证明或排除泄漏。

## 真实任务分解

| 任务 | 原模型 | CP40 | CP80 | 变化pp：CP40 / CP80 |
|---|---:|---:|---:|---:|
{task_rows}

三个阶段均500唯一task/trial、每任务50，policy rank seed为42/43，environment seed base=0，双相机/H10/C5/5步去噪、LIBERO Goal320步的同一原生评估合同。本地汇总回执不含逐trial结果，**不计算配对检验**；同种子也不保证各策略变长调用之后继续共用相同随机噪声。没有多训练seed，不据这一次评估声称跨seed统计显著。JSON中的Wilson区间仅为给定试验的二项描述区间。

## 全量标量摘要

数值维持日志原单位，时间为秒；`success_once`和mask为0–1比例。前10含第4/5/6轮空mask，优势非有限项从均值中排除；所有非有限位置在JSON中保留。KL为采样近似可出现负值。源码中ratio先把无效位置置0再masked_mean，actor按microbatch及rank求平均；空mask微批的0也参与日志聚合，故ratio/clip/近似KL受无效微批比例影响，不能把ratio 0.826解读为所有有效动作概率都降到原来的82.6%。

| tag | 前10均值 | 后10均值 | 全程最小–最大 |
|---|---:|---:|---:|
{metric_rows}

真正靠近学习数据的`rollout/rewards`为：首done截断、整组过滤后的保留**动作槽位**平均相对差分奖励，跨rank按有效sum/count聚合（固定`metric_utils.py:473–533`），不是chunk成功率或轨迹总回报。原始rewards为`[40,128,8]`，chunk mask广播到8个动作。前10的非空轮均值0.0009913→后10 0.0014895；它确实增加，但有效mask及保留样本分布也在变，不能单独解释为策略整体改善。此指标全rank无有效数据时NaN；它与actor ratio按微批等权平均并补0的口径不同。

`env/episode_len`始终320、`env/num_trajectories`始终512；它们反映固定槽位预算，不是证明每个episode到320才第一次done。env可在首次done后继续生成固定长度画面，`env/success_once`累计任意时刻命中；相对奖励的`env/return`跨完整长度望远镜累加为最终分数。actor在首done后mask掉，优化的是截断回报，不能把全长`env/return`直接当作actor训练目标。尤其首done chunk内曾1后0时，成功日志为真而该chunk末差分回报可为0，需专门计数才能估其规模。

## 数据动态目前可见与不可见

可见：每轮全局success、回报、有效mask、优势范围、梯度、loss、近似KL、clip、耗时，以及静态reset资料库存。不可见：实际各任务训练采样次数/KIR起点分布、各任务有效mask、奖励原始概率与阈值来回翻转、真实与生成帧的一致性、首done后成功保持、逐rank/逐进程连续资源。**当前数据支持优先定位推盘任务与成功保持性；不支持已经坐实reward hacking、任务采样偏置或显存随轮数单调增长。**

现场完整枚举742份NPY，496个initial、246个KIR，bad=[]。每任务51–97份；推盘为49+30=79份，占库存10.65%，不存在该任务缺数据。初态样例1帧、KIR样例25帧；包含image/reward/instruction/delta_action，不含state keys。以下为文件库存，不能当作实际训练采样次数或概率。

| 任务 | Initial文件 | KIR文件 | 合计 |
|---|---:|---:|---:|
{data_rows}

轻量后续记录应在现有rollout路径按任务汇总reset/KIR起点计数、成功/首done chunk末分数、有效组/有效步，保留少量固定种子视频对照；资源记录需独立按时间采样GPU总量/进程显存、主机MemAvailable、各自有进程RSS，并标训练阶段。相同阶段内比较基线与峰值，才能判断增长。

## 证据与复现

- [完整标量与只读现场]({SCALARS.as_posix()})
- [原生真实评估回执]({EVAL.as_posix()})
- [原生任务枚举]({NAMES.as_posix()})
- [reset数据库存枚举]({DATASET.as_posix()})
- [可机读计算与全部39 tag统计]({(OUT/'metrics-analysis.json').as_posix()})
- [PDF图]({(OUT/'training-audit.pdf').as_posix()})
- [离线分析脚本]({Path(__file__).resolve().as_posix()})

源文件SHA256保存在分析JSON，统计直接取完整139轮而非历史快照拼接。本离线分析仅消费父审计提供的只读数据库存，不自行扫描原始训练数据、访问SSH或运行服务器实验；没有参数/调度改动。
"""
DOC.parent.mkdir(parents=True, exist_ok=True)
DOC.write_text(doc, encoding="utf-8")
print(json.dumps(dict(figure=str(OUT/"training-audit.png"), document=str(DOC),
    scalar_tags=len(raw), rounds=139, effective_rounds=analysis["effective_rounds"],
    effective_chunk_slots=analysis["loss_enabled_chunk_slots_equivalent"]),ensure_ascii=False))
