const fs = require('fs');
const path = require('path');

const outDir = __dirname;
const snapshot = JSON.parse(fs.readFileSync(path.join(outDir, 'metrics_step1_21.json'), 'utf8'));
const latestLiveRow = {
  step: 22,
  train_success_once: 0.9277344,
  eval_success_once: null,
  actor_approx_kl: 0.014,
  actor_clip_fraction: 0.053,
  actor_grad_norm: 28.012,
  critic_explained_variance: 0.419,
  critic_value_loss: 0.020,
  current_step_time_s: null,
};
const rows = snapshot.rows.some(row => row.step === latestLiveRow.step)
  ? snapshot.rows
  : [...snapshot.rows, latestLiveRow];

const resource = {
  auditCst: '2026-08-22 10:08 CST',
  gpuTotalMiB: 81559,
  gpus: [
    { id: 4, memMiB: 63590 },
    { id: 5, memMiB: 62874 },
    { id: 6, memMiB: 62157 },
    { id: 7, memMiB: 62464 },
  ],
  hostTotalGiB: 2048,
  cgroupGiB: 1430.073,
  availableGiB: 597.0,
  envWorkerRssGiB: 1324.07,
  swapUsedGiB: 0,
  swapTotalGiB: 6,
  memoryEvents: { high: 0, max: 0, oom: 0, oomKill: 0 },
  runGiB: 34.50,
  checkpointsGiB: 34.42,
  videosGiB: 0.0827,
  dataAvailableGiB: 3174.4,
  dataUsedPct: 6,
  fileCount: 372,
  mp4Count: 352,
};

const esc = (s) => String(s).replace(/[&<>\"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));

const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>深圳 π0 PPO formal-100 · Step 22 训练仪表盘</title>
<style>
  :root {
    --ink:#152238; --muted:#637083; --subtle:#8490a3; --paper:#ffffff;
    --canvas:#eef2f7; --grid:#dce3ec; --line:#cbd5e1;
    --blue:#2563eb; --cyan:#0891b2; --green:#15803d; --orange:#ea580c;
    --purple:#7c3aed; --red:#dc2626; --amber:#b45309; --amber-bg:#fff7df;
    --green-bg:#eaf8ef; --shadow:0 8px 28px rgba(37,52,75,.09);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--canvas);color:var(--ink);font-family:"Microsoft YaHei","Noto Sans SC","PingFang SC",Arial,sans-serif}
  .page{max-width:1160px;margin:0 auto;padding:28px 20px 44px}
  .page-head{display:flex;justify-content:space-between;align-items:end;gap:18px;margin:0 0 18px}
  .page-head h1{margin:0;font-size:27px;letter-spacing:-.5px}.page-head p{margin:5px 0 0;color:var(--muted);font-size:14px}
  .desktop-note{font-size:13px;color:var(--muted);text-align:right}
  .layout{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}
  .sheet{width:540px;background:var(--paper);border-radius:22px;box-shadow:var(--shadow);overflow:hidden}
  .sheet-inner{padding:27px 28px 25px}
  .eyebrow{font-weight:800;letter-spacing:1.2px;text-transform:uppercase;color:var(--blue);font-size:11px}
  .sheet h2{margin:7px 0 6px;font-size:25px;line-height:1.2;letter-spacing:-.4px}
  .subtitle{margin:0;color:var(--muted);font-size:13px;line-height:1.6}
  .badges{display:flex;flex-wrap:wrap;gap:7px;margin:14px 0 16px}
  .badge{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:6px 10px;font-size:12px;font-weight:700}
  .badge.ok{background:var(--green-bg);color:var(--green)}.badge.warn{background:var(--amber-bg);color:var(--amber)}
  .dot{width:7px;height:7px;border-radius:50%;background:currentColor}
  .metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:0 0 15px}
  .metric{background:#f6f8fb;border:1px solid #e7ebf1;border-radius:13px;padding:11px 12px;min-height:76px}
  .metric .label{color:var(--muted);font-size:11px;font-weight:700}.metric .value{font-size:23px;line-height:1.2;font-weight:800;margin:3px 0 1px;letter-spacing:-.4px}
  .metric .detail{color:var(--subtle);font-size:10.5px;line-height:1.35}
  .section{border-top:1px solid #e8edf3;padding-top:17px;margin-top:17px}
  .section-head{display:flex;justify-content:space-between;align-items:start;gap:12px;margin-bottom:8px}
  .section-head h3{font-size:16px;margin:0;line-height:1.35}.section-head span{color:var(--muted);font-size:10.5px;text-align:right;line-height:1.4}
  .legend{display:flex;flex-wrap:wrap;gap:11px;margin:3px 0 5px;color:var(--muted);font-size:10.5px}
  .legend button{appearance:none;border:0;background:none;color:inherit;padding:0;cursor:pointer;font:inherit;display:flex;align-items:center;gap:5px}
  .legend button.off{opacity:.32;text-decoration:line-through}.swatch{width:15px;height:3px;border-radius:2px;background:var(--c)}
  .chart{width:100%;height:205px;display:block;overflow:visible}.chart.compact{height:152px}.chart text{font-family:inherit;fill:var(--muted);font-size:10px}
  .chart .grid{stroke:var(--grid);stroke-width:1}.chart .axis{stroke:#b7c1cf;stroke-width:1}.chart .eval-band{fill:#fff3e8}.chart .median{stroke:#8b98aa;stroke-dasharray:4 4;stroke-width:1}
  .chart .series{fill:none;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}.chart .point{stroke:white;stroke-width:1.4;cursor:crosshair}.chart .hit{fill:transparent;cursor:crosshair}
  .chart .focus-line{stroke:#8b98aa;stroke-dasharray:3 3;display:none}.chart .focus-point{stroke:#fff;stroke-width:2;display:none}
  .chart-note{font-size:10.5px;color:var(--muted);line-height:1.5;margin-top:4px}
  .takeaway{border-radius:14px;padding:12px 13px;font-size:11.5px;line-height:1.55;margin-top:16px;background:#eef6ff;border:1px solid #d8e9ff;color:#294567}
  .takeaway strong{color:#17395f}
  .warnbox{border-radius:14px;padding:13px 14px;background:var(--amber-bg);border:1px solid #f5d78e;color:#6f4700;font-size:11.5px;line-height:1.55;margin:14px 0}
  .warnbox strong{display:block;font-size:13px;margin-bottom:2px}
  .gpu-list{display:grid;gap:10px}.gpu-row{display:grid;grid-template-columns:48px 1fr 74px;gap:9px;align-items:center;font-size:11px}.gpu-id{font-weight:800;font-size:13px}
  .bar{height:13px;background:#e8edf3;border-radius:8px;overflow:hidden}.bar>i{display:block;height:100%;border-radius:8px;background:linear-gradient(90deg,#3b82f6,#06b6d4)}
  .bar.amber>i{background:linear-gradient(90deg,#f59e0b,#ea580c)}.bar.green>i{background:linear-gradient(90deg,#22c55e,#15803d)}
  .bar-label{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
  .resource-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px}.resource-card{background:#f6f8fb;border:1px solid #e7ebf1;border-radius:13px;padding:12px}
  .resource-card .k{font-size:10.5px;color:var(--muted);font-weight:700}.resource-card .v{font-size:20px;font-weight:800;margin:3px 0}.resource-card .s{font-size:10px;color:var(--subtle);line-height:1.45}
  .memory-line{margin:10px 0 13px}.memory-line .top{display:flex;justify-content:space-between;gap:8px;font-size:11px;margin-bottom:5px}.memory-line .top b{font-size:11.5px}.memory-line .foot{font-size:9.8px;color:var(--subtle);margin-top:4px}
  .artifact-table{width:100%;border-collapse:collapse;font-size:11px;margin-top:7px}.artifact-table th,.artifact-table td{padding:7px 4px;border-bottom:1px solid #e8edf3;text-align:left}.artifact-table th{color:var(--muted);font-size:10px}.artifact-table td:last-child,.artifact-table th:last-child{text-align:right;font-variant-numeric:tabular-nums}
  .footer{display:flex;justify-content:space-between;gap:10px;color:var(--subtle);font-size:9.5px;border-top:1px solid #e8edf3;padding-top:12px;margin-top:16px;line-height:1.4}
  #tooltip{position:fixed;z-index:99;pointer-events:none;display:none;background:#16243a;color:#fff;border-radius:9px;padding:8px 10px;font-size:11px;line-height:1.45;box-shadow:0 8px 25px rgba(0,0,0,.2);max-width:210px}
  @media(max-width:1120px){.layout{grid-template-columns:1fr}.sheet{margin:0 auto}.page-head{max-width:540px;margin:0 auto 18px}.desktop-note{display:none}}
  @media(max-width:580px){.page{padding:0}.page-head{display:none}.sheet{width:100%;border-radius:0;box-shadow:none}.layout{gap:8px}.sheet-inner{padding:22px 18px}.chart{height:210px}}
</style>
</head>
<body>
<main class="page">
  <header class="page-head">
    <div><h1>π0 PPO formal-100 · 在线训练快照</h1><p>深圳 H100 × latest RLinf × RoboTwin adjust_bottle</p></div>
    <div class="desktop-note">只读现场 · ${resource.auditCst}<br>图表可悬停；点击图例可隐藏曲线</div>
  </header>
  <div class="layout">
    <section id="training-export" class="sheet">
      <div class="sheet-inner">
        <div class="eyebrow">TRAINING · COMPLETE STEP 22</div>
        <h2>训练主链正常，效果指标在改善</h2>
        <p class="subtitle">Global Step 22/100 已完整写入，Step 23 已开始。Step 1–22 均来自完整日志解析；已见标量均 finite。</p>
        <div class="badges"><span class="badge ok"><i class="dot"></i>训练链路正常</span><span class="badge ok"><i class="dot"></i>2 次 eval / checkpoint 落盘</span></div>
        <div class="metric-grid" id="summary-metrics"></div>

        <div class="section">
          <div class="section-head"><h3>成功率</h3><span>train：每步 512 trajectories<br>eval：仅 Step 10 / 20，fixed-64</span></div>
          <div class="legend" id="success-legend"></div>
          <svg id="success-chart" class="chart" viewBox="0 0 470 205" role="img" aria-label="训练和评估成功率趋势"></svg>
          <div class="chart-note">eval 从 58/64（90.63%）升至 62/64（96.88%）；只有两个点，不能据此断言单调收敛。</div>
        </div>

        <div class="section">
          <div class="section-head"><h3>Actor 更新幅度</h3><span>KL 与 clipping 保持有界<br>Step 1 后快速回落</span></div>
          <div class="legend" id="actor-legend"></div>
          <svg id="actor-chart" class="chart compact" viewBox="0 0 470 152" role="img" aria-label="actor KL 和 clip fraction 趋势"></svg>
        </div>

        <div class="section">
          <div class="section-head"><h3>Critic 拟合</h3><span>EV 越高越好；value loss 越低越好</span></div>
          <div class="legend" id="critic-legend"></div>
          <svg id="critic-chart" class="chart" viewBox="0 0 470 205" role="img" aria-label="critic explained variance 和 value loss 趋势"></svg>
          <div class="chart-note">Explained variance：−0.240 → 0.419；value loss：0.119 → 0.020，方向一致且未出现数值发散。</div>
        </div>

        <div class="section">
          <div class="section-head"><h3>每步墙钟时间</h3><span>橙色阴影：含 periodic eval/save</span></div>
          <div class="legend" id="time-legend"></div>
          <svg id="time-chart" class="chart compact" viewBox="0 0 470 152" role="img" aria-label="每个训练步墙钟时间"></svg>
          <div class="chart-note">耗时数据覆盖 Step 1–22；中位数约 1,568 s（26.1 min）。Step 10 / 20 分别约 30.1 / 32.5 min，主要是 eval/save 额外开销。</div>
        </div>

        <div class="takeaway"><strong>判读：</strong>截至 Step 22，success、critic 与 actor 稳定性共同支持“训练功能正常”。这不是最终效果结论；正式比较仍应等更多 fixed-64/128 eval。</div>
        <div class="footer"><span>完整日志解析<br>Global Step 1–22</span><span>快照 ${resource.auditCst}<br>非最终报告</span></div>
      </div>
    </section>

    <section id="resource-export" class="sheet">
      <div class="sheet-inner">
        <div class="eyebrow">RESOURCE · POINT-IN-TIME</div>
        <h2>GPU 有余量，主机内存需黄色关注</h2>
        <p class="subtitle">这是 rollout epoch 间隙的单点快照，不是资源时间序列。瞬时 GPU util 低不代表空转；显存与进程归属均完整。</p>
        <div class="badges"><span class="badge warn"><i class="dot"></i>内存高水位</span><span class="badge ok"><i class="dot"></i>OOM / swap / fatal 均为 0</span></div>

        <div class="section">
          <div class="section-head"><h3>GPU 4–7 显存驻留</h3><span>每卡总显存 81,559 MiB</span></div>
          <div id="gpu-list" class="gpu-list"></div>
          <div class="chart-note">四卡约 60.7–62.1 GiB（76.2%–78.0%）；仍留约 17.5–18.9 GiB/卡。本时点未把另一时刻的 util 混入。</div>
        </div>

        <div class="warnbox"><strong>为什么是“黄色”而不是“异常”</strong>训练 cgroup 当前约 1.397 TiB，主机仅余约 597 GiB available；但 memory events 全 0、swap 未使用、训练仍持续完成 step。单点快照不能证明内存泄漏，需要后续同口径时间点判断斜率。</div>

        <div class="section">
          <div class="section-head"><h3>主机内存（两条独立口径）</h3><span>总 RAM 约 2.0 TiB</span></div>
          <div class="memory-line"><div class="top"><b>训练 cgroup current</b><span>${resource.cgroupGiB.toFixed(0)} GiB · ${(100*resource.cgroupGiB/resource.hostTotalGiB).toFixed(1)}%</span></div><div class="bar amber"><i style="width:${100*resource.cgroupGiB/resource.hostTotalGiB}%"></i></div><div class="foot">包含该训练 cgroup 的当前内存占用；memory.high/max 未设硬上限。</div></div>
          <div class="memory-line"><div class="top"><b>整机 MemAvailable</b><span>${resource.availableGiB.toFixed(0)} GiB · ${(100*resource.availableGiB/resource.hostTotalGiB).toFixed(1)}%</span></div><div class="bar green"><i style="width:${100*resource.availableGiB/resource.hostTotalGiB}%"></i></div><div class="foot">系统估算无需回收即可使用的内存，不与上条简单相加作精确分解。</div></div>
          <div class="resource-grid">
            <div class="resource-card"><div class="k">EnvWorker RSS 求和</div><div class="v">1.293 TiB</div><div class="s">RSS 会重复计算共享映射；判断以 cgroup / MemAvailable 为主。</div></div>
            <div class="resource-card"><div class="k">Memory events</div><div class="v">0 / 0</div><div class="s">oom / oom_kill；high、max 事件也均为 0。</div></div>
          </div>
        </div>

        <div class="section">
          <div class="section-head"><h3>产物与磁盘</h3><span>训练仍在增长</span></div>
          <table class="artifact-table"><thead><tr><th>产物</th><th>现场数量</th><th>大小</th></tr></thead><tbody>
            <tr><td>checkpoint</td><td>Step 10 / 20</td><td>34.42 GiB</td></tr>
            <tr><td>MP4</td><td>352</td><td>84.65 MiB</td></tr>
            <tr><td>TensorBoard event</td><td>1</td><td>60.9 KiB</td></tr>
            <tr><td>run 总计</td><td>372 files</td><td>34.50 GiB</td></tr>
          </tbody></table>
          <div class="memory-line"><div class="top"><b>/data 文件系统</b><span>已用 6% · 可用约 3.1 TiB</span></div><div class="bar green"><i style="width:${resource.dataUsedPct}%"></i></div><div class="foot">当前 run 约占可用空间 1.1%；按每 10 step 约 17.21 GiB checkpoint 粗线性增长，空间充足。</div></div>
        </div>

        <div class="takeaway"><strong>资源结论：</strong>GPU 和磁盘不是当前瓶颈；主机 RAM 是唯一需要继续观察的资源。现在没有 OOM、swap 或训练停滞证据，因此不支持中断训练。</div>
        <div class="footer"><span>单点现场：${resource.auditCst}<br>不代表峰值或趋势</span><span>只读审计<br>未改服务器</span></div>
      </div>
    </section>
  </div>
</main>
<div id="tooltip"></div>
<script>
const rows = ${JSON.stringify(rows)};
const resource = ${JSON.stringify(resource)};

const fmtPct = v => (v*100).toFixed(1)+'%';
const mean = a => a.reduce((x,y)=>x+y,0)/a.length;
const last = rows[rows.length-1];
const first5 = mean(rows.slice(0,5).map(d=>d.train_success_once));
const last5 = mean(rows.slice(-5).map(d=>d.train_success_once));
document.getElementById('summary-metrics').innerHTML = [
  ['进度','22 / 100','Step 23 已开始'],
  ['Train success',fmtPct(last.train_success_once),'近 5 步均值 '+fmtPct(last5)+'；首 5 步 '+fmtPct(first5)],
  ['Fixed-64 eval','96.88%','Step 20：62/64；Step 10：58/64'],
  ['Actor / Critic','KL '+last.actor_approx_kl.toFixed(3),'clip '+fmtPct(last.actor_clip_fraction)+'；EV '+last.critic_explained_variance.toFixed(3)]
].map(x=>'<div class="metric"><div class="label">'+x[0]+'</div><div class="value">'+x[1]+'</div><div class="detail">'+x[2]+'</div></div>').join('');

document.getElementById('gpu-list').innerHTML = resource.gpus.map(g=>{
  const pct=100*g.memMiB/resource.gpuTotalMiB;
  return '<div class="gpu-row"><div class="gpu-id">GPU '+g.id+'</div><div class="bar"><i style="width:'+pct+'%"></i></div><div class="bar-label">'+(g.memMiB/1024).toFixed(1)+' GiB<br>'+pct.toFixed(1)+'%</div></div>';
}).join('');

const NS='http://www.w3.org/2000/svg';
function el(tag, attrs={}, text=''){
  const n=document.createElementNS(NS,tag); Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v)); if(text)n.textContent=text; return n;
}
function niceTicks(min,max,n=4){ const out=[]; for(let i=0;i<=n;i++)out.push(min+(max-min)*i/n); return out; }
function drawLineChart(svgId, legendId, cfg){
  const svg=document.getElementById(svgId), W=470, H=Number(svg.getAttribute('viewBox').split(' ')[3]);
  const m={l:42,r:12,t:10,b:27}, iw=W-m.l-m.r, ih=H-m.t-m.b;
  const xs=rows.map(d=>d.step), xmin=Math.min(...xs),xmax=Math.max(...xs);
  const x=v=>m.l+(v-xmin)/(xmax-xmin)*iw, y=v=>m.t+(cfg.max-v)/(cfg.max-cfg.min)*ih;
  (cfg.bands||[]).forEach(b=>svg.appendChild(el('rect',{x:x(b-.45),y:m.t,width:x(b+.45)-x(b-.45),height:ih,class:'eval-band'})));
  niceTicks(cfg.min,cfg.max,cfg.yTicks||4).forEach(v=>{
    svg.appendChild(el('line',{x1:m.l,y1:y(v),x2:W-m.r,y2:y(v),class:'grid'}));
    svg.appendChild(el('text',{x:m.l-7,y:y(v)+3,'text-anchor':'end'},cfg.formatY(v)));
  });
  [...new Set([1,5,10,15,20,xmax])].forEach(v=>{ svg.appendChild(el('text',{x:x(v),y:H-7,'text-anchor':'middle'},v)); });
  svg.appendChild(el('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}));
  svg.appendChild(el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}));
  if(cfg.median!=null){svg.appendChild(el('line',{x1:m.l,y1:y(cfg.median),x2:W-m.r,y2:y(cfg.median),class:'median'}));}
  const legend=document.getElementById(legendId);
  cfg.series.forEach((s,si)=>{
    const vals=rows.filter(d=>d[s.key]!=null).map(d=>({step:d.step,value:d[s.key]}));
    const group=el('g',{'data-series':s.key});
    const path=vals.map((p,i)=>(i?'L':'M')+x(p.step).toFixed(2)+','+y(p.value).toFixed(2)).join(' ');
    group.appendChild(el('path',{d:path,class:'series',stroke:s.color}));
    vals.forEach(p=>group.appendChild(el('circle',{cx:x(p.step),cy:y(p.value),r:s.sparse?4.5:2.5,class:'point',fill:s.color})));
    svg.appendChild(group);
    const btn=document.createElement('button'); btn.innerHTML='<i class="swatch" style="--c:'+s.color+'"></i>'+s.label;
    btn.onclick=()=>{ const off=group.style.display==='none'; group.style.display=off?'':'none'; btn.classList.toggle('off',!off); };
    legend.appendChild(btn);
  });
  const focus=el('line',{x1:0,y1:m.t,x2:0,y2:H-m.b,class:'focus-line'}); svg.appendChild(focus);
  const hit=el('rect',{x:m.l,y:m.t,width:iw,height:ih,class:'hit'}); svg.appendChild(hit);
  const tip=document.getElementById('tooltip');
  hit.addEventListener('mousemove',e=>{
    const r=svg.getBoundingClientRect(); const sx=(e.clientX-r.left)/r.width*W; const step=Math.max(xmin,Math.min(xmax,Math.round(xmin+(sx-m.l)/iw*(xmax-xmin)))); const d=rows.find(v=>v.step===step);
    focus.style.display='block'; focus.setAttribute('x1',x(step));focus.setAttribute('x2',x(step));
    const parts=cfg.series.filter(s=>d[s.key]!=null).map(s=>'<span style="color:'+s.color+'">●</span> '+s.label+': <b>'+s.tooltip(d[s.key])+'</b>');
    tip.innerHTML='<b>Global Step '+step+'</b><br>'+parts.join('<br>');tip.style.display='block';tip.style.left=Math.min(innerWidth-225,e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';
  });
  hit.addEventListener('mouseleave',()=>{focus.style.display='none';tip.style.display='none'});
}

drawLineChart('success-chart','success-legend',{
  min:.70,max:1.00,formatY:v=>Math.round(v*100)+'%',bands:[10,20],
  series:[
    {key:'train_success_once',label:'Train success',color:'#2563eb',tooltip:fmtPct},
    {key:'eval_success_once',label:'Fixed-64 eval（仅 10/20）',color:'#ea580c',tooltip:fmtPct,sparse:true}
  ]
});
drawLineChart('actor-chart','actor-legend',{
  min:0,max:.10,formatY:v=>(v*100).toFixed(0)+'%',bands:[],
  series:[
    {key:'actor_approx_kl',label:'Approx KL',color:'#7c3aed',tooltip:v=>v.toFixed(4)},
    {key:'actor_clip_fraction',label:'Clip fraction',color:'#0891b2',tooltip:fmtPct}
  ]
});
drawLineChart('critic-chart','critic-legend',{
  min:-.25,max:.45,formatY:v=>v.toFixed(2),bands:[],
  series:[
    {key:'critic_explained_variance',label:'Explained variance',color:'#15803d',tooltip:v=>v.toFixed(3)},
    {key:'critic_value_loss',label:'Value loss（同轴）',color:'#ea580c',tooltip:v=>v.toFixed(3)}
  ]
});
const times=rows.map(d=>d.current_step_time_s).filter(v=>v!=null).slice().sort((a,b)=>a-b); const med=times[Math.floor(times.length/2)];
drawLineChart('time-chart','time-legend',{
  min:1300,max:2050,formatY:v=>Math.round(v/60)+'m',bands:[10,20],median:med,
  series:[{key:'current_step_time_s',label:'Time / step（虚线为中位数）',color:'#2563eb',tooltip:v=>(v/60).toFixed(1)+' min'}]
});
</script>
</body></html>`;

fs.writeFileSync(path.join(outDir, 'ppo_step22_dashboard.html'), html, 'utf8');
console.log(path.join(outDir, 'ppo_step22_dashboard.html'));
