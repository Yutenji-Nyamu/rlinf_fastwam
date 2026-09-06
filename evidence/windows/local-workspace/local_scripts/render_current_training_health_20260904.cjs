const fs=require('fs'),path=require('path');
const sharp=require('C:/Users/86136/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const src='docs/fastwam-robotwin-rlinf-grpo/evidence/CURRENT_TRAINING_HEALTH_20260904_1417.data.txt';
const data=JSON.parse(fs.readFileSync(src,'utf8').split(/\r?\n/).find(l=>l.startsWith('SNAPSHOT_JSON ')).slice(14));
const out='docs/fastwam-robotwin-rlinf-grpo/evidence/current-training-health-20260904-1419';
const W=1100,ink='#172f43',muted='#617181',orange='#d87814',teal='#087f8c',red='#b44857',grid='#dce5ec';
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const median=a=>{a=[...a].sort((a,b)=>a-b);return a.length%2?a[(a.length-1)/2]:(a[a.length/2-1]+a[a.length/2])/2;};
const runs=data.runs,labels={fastwam:'Fast-WAM / stapler / OIDN off',sidney:'Sidney pi0.5 / pillbottle'},colors={fastwam:orange,sidney:teal};
// TensorBoard is zero-based in this runner; align to one-based completed-step display.
const points=(r,tag,m=1)=>(runs[r].scalars[tag]||[]).map(p=>({x:p.step+1,y:p.value*m}));
for(const r of Object.keys(runs)){
 const a=points(r,'env/success_once');
 if(a.length&&a.at(-1).x!==runs[r].state.completed_step)throw Error('Step alignment mismatch: '+r);
}
const ma=(a,n)=>a.flatMap((p,i)=>i<n-1?[]:[{x:p.x,y:mean(a.slice(i-n+1,i+1).map(v=>v.y))}]);
const series=(name,color,points,extras={})=>({name,color,points,...extras});
const specs={};
const summary={snapshot:data.time,runs:{}};
for(const r of Object.keys(runs)){
 const run=runs[r],train=points(r,'env/success_once',100),ev=points(r,'eval/success_once',100),step=run.state.completed_step;
 const xmax=r==='fastwam'?5:Math.ceil(step/5)*5;
 specs[r]={id:r,title:labels[r],unit:'Success %',xmax,ymin:0,ymax:80,yticks:[0,20,40,60,80],xticks:r==='fastwam'?[0,1,2,3,4,5]:[0,10,20,30,40,50],xlabel:'Completed outer step (TB step + 1; no invented Step 0)',series:[series('Train / 256 episodes',colors[r],train,{opacity:.45,markers:true}),series('Train MA5',colors[r],ma(train,5),{width:3.5}),series('Fixed eval / 32',r==='fastwam'?teal:orange,ev,{markers:true,dash:true,labels:ev.map(p=>Math.round(p.y/100*32)+'/32')})],note:r==='fastwam'?'Step 1: 50/256. No fixed evaluation yet. Step 2: suspected stall.':'Step 46: 154/256. Fixed eval Step 45: 14/32; peak remains 19/32.'};
 const vals=(run.scalars['env/success_once']||[]).map(p=>p.value);
 const res=run.resource;
 const gpuKeys=Object.keys(res[0]||{}).filter(k=>k.endsWith('_used_mib'));
 const peaks=gpuKeys.map(k=>({gpu:k,peak_mib:Math.max(...res.map(row=>Number(row[k]||0)))}));
 const time=run.scalars['time/step']||[];
 summary.runs[r]={completed_step:step,last_train:vals.at(-1),ma5:vals.length>=5?mean(vals.slice(-5)):null,ma10:vals.length>=10?mean(vals.slice(-10)):null,last_eval:ev.at(-1)||null,first10_mean:vals.length>=10?mean(vals.slice(0,10)):null,last10_mean:vals.length>=10?mean(vals.slice(-10)):null,peaks,step_seconds_last:time.at(-1)?.value,step_seconds_median_last5:time.length?median(time.slice(-5).map(p=>p.value)):null,checkpoint_steps:run.checkpoints.map(c=>c.step),errors:run.state.errors};
 const grad=points(r,'train/actor/grad_norm'),clip=points(r,'train/actor/clip_fraction',100),kl=points(r,'train/actor/approx_kl');
 const ym=Math.ceil(Math.max(1,...grad.map(p=>p.y))*1.1/5)*5;
 specs[r+'_grad']={id:r+'_grad',title:labels[r]+' — optimizer diagnostics',unit:'Pre-clip gradient norm',xmax,ymin:0,ymax:ym,yticks:[0,ym/4,ym/2,ym*3/4,ym],xticks:specs[r].xticks,xlabel:'Completed outer step',series:[series('Gradient norm',colors[r],grad,{markers:step<3})],note:`Latest KL=${kl.at(-1)?.y.toFixed(4)}; clip=${clip.at(-1)?.y.toFixed(2)}%. Logged averages, not a causal comparison.`};
}
const fastres=runs.fastwam.resource,t0=Date.parse(fastres[0].timestamp),xend=(Date.parse(data.time)-t0)/36e5;
const gpuSeries=[];
for(const r of ['fastwam','sidney']){
 const rr=runs[r].resource.filter(x=>Date.parse(x.timestamp)>=t0);
 for(const k of Object.keys(rr[0]||{}).filter(k=>k.endsWith('_used_mib'))){
  const index=Number(k.match(/gpu(\d+)/)[1]);
  gpuSeries.push(series('GPU '+index,index<6?teal:orange,rr.map(p=>({x:(Date.parse(p.timestamp)-t0)/36e5,y:Number(p[k])/1024})),{width:2,dash:index%2===1}));
 }
}
specs.gpu={id:'gpu',title:'GPU memory since Fast-WAM launch',unit:'GiB / GPU',xmax:Math.ceil(xend*2)/2,ymin:0,ymax:80,yticks:[0,20,40,60,80],xticks:[0,.5,1,1.5,2],xlabel:'Hours since 12:43 CST | solid: 4/6; dashed: 5/7',series:gpuSeries,note:'Fast-WAM GPU 6/7: 0% utilization in consecutive samples since ~14:05.'};
specs.ram={id:'ram',title:'Host available RAM — includes all users and jobs',unit:'TiB available',xmax:specs.gpu.xmax,ymin:0,ymax:1.5,yticks:[0,.5,1,1.5],xticks:specs.gpu.xticks,xlabel:'Hours since Fast-WAM launch',series:[series('MemAvailable',teal,fastres.map(p=>({x:(Date.parse(p.timestamp)-t0)/36e5,y:Number(p.host_mem_available_kib)/2**30})))],note:'Availability is falling, but current memory / I/O pressure averages are zero.'};
function plot(s,y,h){
 const l=92,r=40,t=y+110,b=y+h-84,pw=W-l-r,ph=b-t,fx=x=>l+x/s.xmax*pw,fy=v=>b-(v-s.ymin)/(s.ymax-s.ymin)*ph;
 let a=`<g><text x="48" y="${y+29}" font-size="24" font-weight="600">${esc(s.title)}</text><text x="${l}" y="${y+93}" font-size="16" fill="${muted}">${esc(s.unit)}</text>`;
 for(const v of s.yticks)a+=`<line x1="${l}" x2="${W-r}" y1="${fy(v)}" y2="${fy(v)}" stroke="${grid}"/><text x="${l-14}" y="${fy(v)+6}" text-anchor="end" font-size="17" fill="${muted}">${Number(v.toFixed(2))}</text>`;
 for(const x of s.xticks.filter(x=>x<=s.xmax))a+=`<text x="${fx(x)}" y="${b+25}" text-anchor="middle" font-size="17" fill="${muted}">${x}</text>`;
 let lx=l;
 for(const v of s.series.filter(v=>v.points.length)){const pp=v.points.filter(p=>Number.isFinite(p.y));a+=`<path d="${pp.map((p,i)=>(i?'L':'M')+fx(p.x).toFixed(2)+','+fy(p.y).toFixed(2)).join(' ')}" fill="none" stroke="${v.color}" stroke-width="${v.width||2}" opacity="${v.opacity||1}" ${v.dash?'stroke-dasharray="8 5"':''}/>`;
 if(v.markers)pp.forEach((p,i)=>{a+=`<circle cx="${fx(p.x)}" cy="${fy(p.y)}" r="5" fill="white" stroke="${v.color}" stroke-width="2.5"/>`;if(v.labels)a+=`<text x="${fx(p.x)}" y="${fy(p.y)+25}" text-anchor="middle" font-size="16" fill="${v.color}">${v.labels[i]}</text>`;});
 a+=`<line x1="${lx}" x2="${lx+24}" y1="${y+56}" y2="${y+56}" stroke="${v.color}" stroke-width="3" ${v.dash?'stroke-dasharray="5 3"':''}/><text x="${lx+31}" y="${y+62}" font-size="16" fill="${muted}">${esc(v.name)}</text>`;lx+=v.name.length*8+62;}
 a+=`<text x="${l+pw/2}" y="${b+49}" text-anchor="middle" font-size="17" fill="${muted}">${esc(s.xlabel)}</text><text x="48" y="${y+h-8}" font-size="18" fill="${s.id==='fastwam'?red:muted}">${esc(s.note)}</text><line class="hoverline" x1="0" x2="0" y1="${t}" y2="${b}" stroke="${ink}" visibility="hidden"/><rect class="hoverarea" x="${l}" y="${t}" width="${pw}" height="${ph}" fill="transparent" data-id="${s.id}" data-left="${l}" data-width="${pw}"/></g>`;
 return a;
}
function svg(title,subtitle,content,h){return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${h}" viewBox="0 0 ${W} ${h}"><rect width="100%" height="100%" fill="#f7fafc"/><g font-family="Segoe UI,Arial,sans-serif" fill="${ink}"><text x="48" y="45" font-size="29" font-weight="700">${esc(title)}</text><text x="48" y="78" font-size="18" fill="${muted}">${esc(subtitle)}</text>${content}</g></svg>`;}
const st='Read-only server snapshot | 2026-09-04 14:19 CST';
const svgs={success:svg('Training: Sidney progressing; Fast-WAM needs attention',st+' | Different tasks; no direct model ranking',plot(specs.fastwam,100,360)+plot(specs.sidney,490,370),890),optimization:svg('Optimization diagnostics',st+' | Fast-WAM has one completed update cycle',plot(specs.fastwam_grad,100,350)+plot(specs.sidney_grad,480,350),865),resources:svg('Resources: memory available, Fast-WAM GPUs idle',st,plot(specs.gpu,100,350)+plot(specs.ram,480,350),865)};
const all=svg('Current training and server dashboard',st,Object.values(specs).map((s,i)=>plot(s,100+i*380,360)).join(''),120+Object.keys(specs).length*380);
const html=`<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>14:19 训练与服务器快照</title><style>body{margin:0;background:#e9f0f5;color:${ink};font:16px 'Segoe UI','Microsoft YaHei',sans-serif}main{max-width:1100px;margin:20px auto;background:#f7fafc}header{padding:22px 32px}a{color:${teal};margin-right:18px}svg{width:100%;height:auto}.tip{position:fixed;display:none;pointer-events:none;max-width:380px;padding:12px;background:#172f43ef;color:white;border-radius:8px;font-size:14px;z-index:3}</style><main><header><b>Fast-WAM完成Step1，Step2出现停滞；Sidney完成Step46。</b><p>两项任务不同，成功率不用于模型排名。TB横轴+1与日志完整step对齐；没有虚构Step0。Fast-WAM无fixed eval点，不能判断学习改善。</p><a href="success.png">成功率PNG</a><a href="optimization.png">优化PNG</a><a href="resources.png">资源PNG</a></header>${all}</main><div class="tip"></div><script>const specs=${JSON.stringify(specs)};const tip=document.querySelector('.tip');document.querySelectorAll('.hoverarea').forEach(el=>{const sp=specs[el.dataset.id],line=el.parentNode.querySelector('.hoverline');el.addEventListener('pointermove',e=>{const svg=el.closest('svg'),p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const q=p.matrixTransform(svg.getScreenCTM().inverse()),x=(q.x-+el.dataset.left)/+el.dataset.width*sp.xmax;line.setAttribute('x1',q.x);line.setAttribute('x2',q.x);line.setAttribute('visibility','visible');tip.textContent=sp.series.filter(s=>s.points.length).map(s=>{const a=s.points.reduce((b,v)=>Math.abs(v.x-x)<Math.abs(b.x-x)?v:b);return s.name+': '+a.y.toFixed(3)+' @ '+a.x.toFixed(2)}).join(' | ');tip.style.display='block';tip.style.left=Math.max(8,Math.min(e.clientX+14,innerWidth-400))+'px';tip.style.top=Math.max(8,e.clientY-100)+'px'});el.addEventListener('pointerleave',()=>{line.setAttribute('visibility','hidden');tip.style.display='none'})});</script></html>`;
(async()=>{fs.mkdirSync(out,{recursive:true});for(const [name,s]of Object.entries(svgs))await sharp(Buffer.from(s)).png().toFile(path.join(out,name+'.png'));fs.writeFileSync(path.join(out,'dashboard.html'),html);fs.writeFileSync(path.join(out,'summary.json'),JSON.stringify(summary,null,2));fs.writeFileSync(path.join(out,'plot-data.json'),JSON.stringify({snapshot:data.time,specs}));console.log(JSON.stringify(summary,null,2));})();
