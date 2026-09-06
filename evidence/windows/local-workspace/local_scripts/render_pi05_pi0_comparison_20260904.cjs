const fs=require('fs');
const sharp=require('C:/Users/86136/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const base='docs/rlinf-shenzhen-multitask-pi05/evidence/';
const a=JSON.parse(fs.readFileSync(base+'PI05_PI0_COMPARISON_ANALYSIS_20260904.json','utf8'));
const out=base+'pi05-pi0-comparison-20260904';fs.mkdirSync(out,{recursive:true});
const colors={pi0:'#d97800',sidney:'#008697',official_pi05:'#667889'};
const names={pi0:'Pi0 / adjust_bottle',sidney:'Sidney Pi0.5 / pillbottle',official_pi05:'Task-SFT Pi0.5 / adjust_bottle'};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const txt=(x,y,s,size=19,extra='')=>`<text x="${x}" y="${y}" font-size="${size}" ${extra}>${esc(s)}</text>`;
const start=(title,sub,h)=>`<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="${h}" viewBox="0 0 1100 ${h}"><rect width="1100" height="${h}" fill="#f6fafc"/><g font-family="Arial, sans-serif" fill="#193449">${txt(45,42,title,28,'font-weight="700"')}${txt(45,73,sub,17,'fill="#60778b"')}`;
const end='</g></svg>';
const mean=x=>x.reduce((s,p)=>s+p,0)/x.length;
const series=(k,t,scale=1,ma=1)=>a.series[k][t].map((p,i,arr)=>({x:p.step,y:mean(arr.slice(Math.max(0,i-ma+1),i+1).map(q=>q.value))*scale}));
function plot(title,sub,y,data,{max=100,ticks=[0,25,50,75,100],xmax=100}={}){
 const left=90,top=y+79,w=950,h=175,xx=x=>left+x/xmax*w,yy=v=>top+h-v/max*h;
 let s=txt(45,y+24,title,23,'font-weight="600"')+txt(45,y+51,sub,16,'fill="#60778b"');
 for(const v of ticks)s+=`<path d="M${left} ${yy(v)}H1040" stroke="#dce7ee"/>`+txt(50,yy(v)+6,v,16);
 for(let v=0;v<=xmax;v+=xmax===100?20:10)s+=txt(xx(v)-7,top+h+26,v,16);
 for(const d of data){
  const c=colors[d.key],pts=d.pts.filter(p=>p.x<=xmax);
  s+=`<g data-run="${d.key}"><polyline points="${pts.map(p=>`${xx(p.x)},${yy(p.y)}`).join(' ')}" fill="none" stroke="${c}" stroke-width="${d.width||2.5}" ${d.dash?'stroke-dasharray="7 5"':''}/>`;
  for(const p of pts)s+=`<circle cx="${xx(p.x)}" cy="${yy(p.y)}" r="${d.dash?4:2.5}" fill="${d.dash?'#f6fafc':c}" stroke="${c}"><title>${names[d.key]} | Step ${p.x}: ${p.y.toFixed(2)}</title></circle>`;
  s+='</g>';
 }
 return s+txt(480,top+h+54,'Completed outer step',16,'fill="#60778b"');
}
const legend=()=>Object.keys(names).map((k,i)=>`<g data-run="${k}"><path d="M${45+i*350} 106h25" stroke="${colors[k]}" stroke-width="3"/>${txt(80+i*350,112,names[k],16)}</g>`).join('');
let success=start('Training gain is not the same as fixed-evaluation gain','Read-only 2026-09-04 17:10 CST | Different tasks / SFT starts; not a model ranking',790)+legend();
success+=plot('Training success (%) — rolling mean over 5 steps','No invented Step 0. Main comparison uses equal windows within the first 50 steps.',133,Object.keys(names).map(key=>({key,pts:series(key,'env/success_once',100,5)})));
success+=plot('Fixed evaluation (%) — 32 episodes per point','Pi0 stabilizes near 30/32; Sidney has not persistently exceeded 19/32 at Step10.',463,Object.keys(names).map(key=>({key,pts:series(key,'eval/success_once',100),dash:true})))+end;
let opt=start('Optimizer diagnostics — same 256-trajectory step budget','First 50 steps only | Logged averages; nonzero values are not proof of good generalization',785)+legend();
opt+=plot('Approximate KL','Pi0 mean 0.0156; Sidney mean 0.0176. No evidence of a nearly-zero-update regime.',130,['pi0','sidney'].map(key=>({key,pts:series(key,'train/actor/approx_kl',1,5)})),{max:.12,ticks:[0,.03,.06,.09,.12],xmax:50});
opt+=plot('Clip fraction (%)','Pi0 mean 6.78%; Sidney mean 5.16%. Same MB32; active-mask counts not logged.',460,['pi0','sidney'].map(key=>({key,pts:series(key,'train/actor/clip_fraction',100,5)})),{max:15,ticks:[0,5,10,15],xmax:50})+end;
let res=start('Wall-clock comparison at equal training budget','50 steps = 12,800 train trajectories + 100 optimizer calls | Same 2-GPU layout',705)+legend();
function bars(title,y,values,max,unit){
 let s=txt(45,y,title,24,'font-weight="600"');
 for(const[k,v]of Object.entries(values)){
  const i=Object.keys(values).indexOf(k),yy=y+28+i*43;
  s+=`<g data-run="${k}">`+txt(45,yy+21,names[k],18)+`<rect x="395" y="${yy}" width="${v/max*550}" height="29" rx="4" fill="${colors[k]}"/>`+txt(410+v/max*550,yy+21,v.toFixed(2)+' '+unit,18)+'</g>';
 }return s;
}
res+=bars('Hours from launch to completed Step50',162,Object.fromEntries(Object.keys(names).map(k=>[k,a.runs[k].wall_hours_to50])),26,'h');
res+=bars('Mean outer-step time, first 50 steps (includes periodic eval/save)',369,Object.fromEntries(Object.keys(names).map(k=>[k,a.runs[k].time['time/step'].first50_mean/60])),31,'min');
res+=txt(45,571,'Sidney reaches Step50 in 19.52 h vs Pi0 22.37 h; it is not slower in wall time.',21);
res+=txt(45,609,'M10 increases model inference work, but environment/reset work dominates these runs.',19);
res+=txt(45,646,'Historical runtime observations, not an isolated hardware or model-speed benchmark.',18,'fill="#60778b"')+end;
const charts={success,optimization:opt,resources:res};
Promise.all(Object.entries(charts).map(async([name,svg])=>sharp(Buffer.from(svg)).png().toFile(out+'/'+name+'.png'))).then(()=>{
 const controls=Object.keys(names).map(k=>`<label><input type="checkbox" data-key="${k}" checked>${names[k]}</label>`).join(' ');
 fs.writeFileSync(out+'/dashboard.html',`<!doctype html><meta charset="utf-8"><title>π0 / Sidney π0.5 learning audit</title><style>body{font:18px Arial;background:#eaf0f4;margin:24px;color:#193449}header{position:sticky;top:0;background:#fff;padding:14px;z-index:2}label{margin-right:20px}svg{display:block;max-width:100%;height:auto;margin:22px auto;background:white}p{max-width:1100px;margin:12px auto}</style><header>${controls}<p>勾选曲线；悬停数据点查看数值。不同任务，不作严格模型排名。</p></header>${Object.values(charts).join('')}<script>document.querySelectorAll('input').forEach(x=>x.onchange=()=>document.querySelectorAll('[data-run="'+x.dataset.key+'"]').forEach(g=>g.style.display=x.checked?'':'none'));</script>`);
 console.log(JSON.stringify({out,files:[...Object.keys(charts).map(k=>k+'.png'),'dashboard.html']}));
}).catch(e=>{console.error(e);process.exitCode=1});
