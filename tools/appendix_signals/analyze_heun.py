"""CPU-only paired DV statistics, reproducible figures, and an offline explorer."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import write_json


def main(args):
    d=Path(args.data)
    meta=json.loads((d/'complete.json').read_text())
    q=json.loads((d/'queries.json').read_text())
    with np.load(d/'inputs.npz',allow_pickle=False) as z: mask=z['executed_mask'].copy()
    arrays={}
    for solver in ['euler','heun']:
        with np.load(d/f'{solver}.npz',allow_pickle=False) as z:
            arrays[solver]={k:z[k].copy() for k in z.files}
    e,h=(arrays[s]['dv'].reshape(32,8,50) for s in ['euler','heun'])
    rho=np.asarray([[spearmanr(e[i,k],h[i,k]).statistic for k in range(8)] for i in range(32)])
    valid=np.asarray([[spearmanr(e[i,k,mask[i]],h[i,k,mask[i]]).statistic
                       if mask[i].sum()>2 else np.nan for k in range(8)] for i in range(32)])
    overlap=np.asarray([[len(set(np.argsort(e[i,k])[-10:]) & set(np.argsort(h[i,k])[-10:]))/10
                         for k in range(8)] for i in range(32)])
    ratio=h/(e+1e-12)
    episode=[]
    for seed in dict.fromkeys(x['seed'] for x in q):
        ix=[i for i,x in enumerate(q) if x['seed']==seed]
        episode.append(dict(seed=seed,original_success=q[ix[0]]['success'],observations=len(ix),
                            rho_mean=float(np.nanmean(rho[ix])),
                            mean_dv_ratio=float(h[ix].mean()/e[ix].mean())))
    result=dict(task=meta['task'],observations=32,episodes=8,paired_chunks=256,
        settings=dict(N=10,L=5,H=50,D=14),
        rho_mean=float(np.nanmean(rho)),rho_median=float(np.nanmedian(rho)),
        rho_p10_p90=np.nanpercentile(rho,[10,90]).tolist(),undefined_rho=int(np.isnan(rho).sum()),
        rho_original_executed_positions=float(np.nanmean(valid)),
        original_executed_pairs=int(np.isfinite(valid).sum()),
        high_dv_top20_overlap=float(overlap.mean()),
        euler_mean_dv=float(e.mean()),heun_mean_dv=float(h.mean()),
        ratio_of_mean_dv=float(h.mean()/e.mean()),median_position_dv_ratio=float(np.median(ratio)),
        normalized_action_rmse=float(np.sqrt(np.mean((arrays['euler']['final_model_action']-arrays['heun']['final_model_action'])**2))),
        inference_time_ratio=meta['timings']['heun']['seconds']/meta['timings']['euler']['seconds'],
        timings=meta['timings'],episode_summary=episode,
        interpretation='One-task offline solver sensitivity. Noise repeats are paired robustness samples, not independent episodes. No Heun execution success rates or training gains measured.')
    write_json(d/'summary.json',result)
    np.savez_compressed(d/'statistics.npz',rho=rho,executed_rho=valid,top20_overlap=overlap,dv_ratio=ratio)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    axs[0].scatter(e.ravel(),h.ravel(),s=3,alpha=.15,color='#167d9a',rasterized=True)
    lim=[float(min(e.min(),h.min())),float(max(e.max(),h.max()))]
    axs[0].plot(lim,lim,'--',color='#8d98a4',lw=1)
    axs[0].set(xscale='log',yscale='log',xlabel='Euler DV',ylabel='Heun DV',title='Same observation + same noise')
    axs[1].hist(rho.ravel(),bins=np.linspace(-1,1,31),color='#167d9a')
    axs[1].axvline(result['rho_mean'],color='#d56b33',ls='--')
    axs[1].set(xlabel='Within-chunk Spearman',ylabel='Paired chunks',title=f"Mean rank correlation = {result['rho_mean']:.3f}")
    axs[2].plot(np.arange(1,51),e.mean((0,1)),label='Euler',color='#167d9a')
    axs[2].plot(np.arange(1,51),h.mean((0,1)),label='Heun',color='#d56b33')
    axs[2].set(xlabel='Action position in chunk',ylabel='Mean DV',title='Mean over 32 observations x 8 noises')
    axs[2].legend(frameon=False)
    fig.savefig(d/'overview.png',dpi=180);fig.savefig(d/'overview.svg');plt.close(fig)
    chart=[dict(seed=x['seed'],query=x['query'],success=x['success'],executed=int(mask[i].sum()),
                rho=rho[i].tolist(),e=e[i].tolist(),h=h[i].tolist()) for i,x in enumerate(q)]
    payload=json.dumps(chart,ensure_ascii=False).replace('</','<\\/')
    report=f'''# Euler → Heun：单任务配对 DV 实测

- turn_switch，冻结同一 π0.5；8 条原轨迹、32 个观测，8 份配对噪声，共 256 对动作块。
- N=10、L=5、H=50、D=14；两边都用完整积分步起点的 x−t·v1，取最后 5 份估计的总体方差，再对 14 维求和。
- 逐 chunk 排序相关性平均 **{result['rho_mean']:.4f}**，中位数 **{result['rho_median']:.4f}**；10%～90% 分位 **{result['rho_p10_p90'][0]:.4f}～{result['rho_p10_p90'][1]:.4f}**。
- 最高 20% DV 位置重合 **{result['high_dv_top20_overlap']:.1%}**；平均 DV 为 Euler 的 **{result['ratio_of_mean_dv']:.3f} 倍**。
- 仅比较原轨迹实际执行的位置：排序相关性 **{result['rho_original_executed_positions']:.4f}**，有效配对 {result['original_executed_pairs']}。
- 原生 Euler 与旧 B16 记录对齐；两种采样器首个 clean-action estimate 逐元素相同。每个动作块分别 10／20 次速度网络调用；批量推理耗时比 **{result['inference_time_ratio']:.3f}**（单轮诊断计时，含数据转换，不是严格速度榜单）。

这只能回答同输入换积分器后 DV 的数值与排序变化；没有执行 Heun 动作，不能据此报告成功率提高。噪声重复不等于新增独立轨迹。所有 32 个观测均可在网页切换查看，未按结果挑选。

原始文件：inputs.npz、euler.npz、heun.npz、queries.json；统计：summary.json、statistics.npz；图：overview.png／overview.svg。

实现参考 [EDM predictor/corrector](https://github.com/NVlabs/edm/blob/main/generate.py)；本项目采用无 sigma 除法的 flow velocity，在终点 t=0 也计算修正，因此是 20 次求值。DV 使用本项目原生 compute_endpoint_variance，Euler 路径不改。
'''
    (d/'REPORT.md').write_text(report,encoding='utf8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Euler / Heun · DV 配对分析</title>
<style>body{font:16px/1.7 system-ui;background:#f3f5f7;color:#223045;max-width:1120px;margin:auto;padding:30px}h1{font-size:29px}section{background:white;padding:24px;margin:18px 0;border-radius:14px}select{padding:9px;margin:5px;font-size:15px}img,svg{width:100%;height:auto}p{margin:8px 0}.muted{color:#657389}.kpi{display:flex;gap:24px;flex-wrap:wrap}.kpi b{font-size:25px;display:block;color:#167d9a}a{color:#167d9a}#stats{min-height:32px}</style>
<h1>Euler → Heun：DV 怎么变？</h1><p class="muted">turn_switch · 同模型 / 观测 / 初始噪声 · N=10，L=5 · 2026-11-12</p>
<section><div class="kpi">__KPIS__</div><p>32 个观测来自 8 条原轨迹；每个观测配对 8 份噪声，共 256 对。统计各动作块内 50 个动作位置的 DV 排序。</p><p class="muted">这是离线信号敏感性诊断；没有新执行成功率，也未测试 Filter IL 训练收益。</p></section>
<section><h2>全部配对结果</h2><img src="overview.png" alt="DV大小、排序相关性及动作位置平均曲线"></section>
<section><h2>逐个查看，不只展示好看的例子</h2><label>原轨迹观测 <select id="obs"></select></label><label>配对噪声 <select id="noise"></select></label><p id="stats"></p><svg id="plot" viewBox="0 0 1000 340" role="img" aria-label="Euler和Heun逐动作DV曲线"></svg><p class="muted">蓝色 Euler；橙色 Heun。虚线标出原轨迹实际执行位置的末端，其余是模型预测的后续位置。</p></section>
<section><h2>口径与原始数据</h2><p>每个完整积分步记录一次 x−t·v1；Heun 的第二次求值只修正动作更新。两者各取最后 5 份 clean-action estimate，按原生代码求总体方差并对 14 维求和。</p><p><a href="REPORT.md">简要报告</a> · <a href="summary.json">统计 JSON</a> · <a href="overview.svg">可编辑矢量图</a> · <a href="https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/appendix-signals-20261111">GitHub 原始数据</a></p></section>
<script>const rows=__DATA__;const $=x=>document.getElementById(x);rows.forEach((r,i)=>$('obs').add(new Option(`${i+1}. seed ${r.seed} · query ${r.query} · 原轨迹${r.success?'成功':'失败'}`,i)));for(let k=0;k<8;k++)$('noise').add(new Option(k+1,k));
function draw(){let r=rows[+$('obs').value],k=+$('noise').value,e=r.e[k],h=r.h[k],mx=Math.max(...e,...h)*1.05,x=i=>65+i*900/49,y=v=>280-v/mx*240;
let path=a=>a.map((v,i)=>(i?'L':'M')+x(i).toFixed(2)+','+y(v).toFixed(2)).join(' ');let s=`<path d="M65,40V280H965" fill="none" stroke="#bac4d0"/>`;
for(let j=0;j<5;j++){let v=mx*j/4;s+=`<text x="57" y="${y(v)+4}" text-anchor="end" font-size="12">${v.toPrecision(2)}</text><path d="M65,${y(v)}H965" stroke="#eef1f4"/>`;}for(let i of [0,9,19,29,39,49])s+=`<text x="${x(i)}" y="304" font-size="13" text-anchor="middle">${i+1}</text>`;
s+=`<path d="${path(e)}" fill="none" stroke="#167d9a" stroke-width="2.5"/><path d="${path(h)}" fill="none" stroke="#d56b33" stroke-width="2.5"/>`;if(r.executed>0&&r.executed<50)s+=`<path d="M${x(r.executed-1)},40V280" stroke="#8d98a4" stroke-dasharray="5 5"/>`;
$('plot').innerHTML=s;$('stats').textContent=`这对动作块的排序相关性：${r.rho[k].toFixed(3)}；原轨迹执行 ${r.executed}/50 个位置。`;}
$('obs').onchange=draw;$('noise').onchange=draw;draw();</script></html>'''
    kpis=f"<div><b>{result['rho_mean']:.3f}</b>平均排序相关性</div><div><b>{result['ratio_of_mean_dv']:.2f}×</b>平均 DV 数值</div><div><b>{result['high_dv_top20_overlap']:.1%}</b>高 DV 位置重合</div><div><b>10 → 20</b>每条去噪链求值次数</div>"
    (d/'index.html').write_text(page.replace('__KPIS__',kpis).replace('__DATA__',payload),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);main(p.parse_args())
