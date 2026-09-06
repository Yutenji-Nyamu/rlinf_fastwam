const fs=require('fs');
const path=require('path');
const sharp=require('C:/Users/86136/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const base='C:/Users/86136/Documents/rl';
const input=process.argv[2]||path.join(base,'docs/fastwam-robotwin-rlinf-grpo/evidence/OIDN_RECURRENCE_INVESTIGATION_20260904.raw.txt');
const output=path.join(base,'docs/rlinf-shenzhen-multitask-pi05/evidence/sidney-live-20260904');
const jsonLine=fs.readFileSync(input,'utf8').split(/\r?\n/).find(l=>l.startsWith('SIDNEY_JSON '));
if(!jsonLine)throw new Error('No audited Sidney data');
const d=JSON.parse(jsonLine.slice(12));
const train=d.scalars['env/success_once'],evaluations=d.scalars['eval/success_once'];
const latest=train.at(-1).step;
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
const ma5=mean(train.slice(-5).map(x=>x.value))*100,ma10=mean(train.slice(-10).map(x=>x.value))*100;
const teal='#087f8c',orange='#c65d08',muted='#64748b',navy='#162b3d',grid='#dbe4ea';
const W=1100;
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const pct=x=>(x*100).toFixed(2)+'%';
const stamp=d.cst.slice(0,16).replace('T',' ')+' CST';
const resource=d.resource.filter(x=>Number.isFinite(Date.parse(x.timestamp)));
const t0=Date.parse(resource[0].timestamp);
const elapsed=resource.map(x=>(Date.parse(x.timestamp)-t0)/3600000),hours=elapsed.at(-1);
const xy=(a,fn)=>a.map(p=>({x:p.step,y:fn(p.value)}));
const seriesSpec={
 success:{id:'success',title:'Success rate',unit:'%',xmax:latest,ymax:100,yticks:[0,25,50,75,100],xticks:Array.from({length:Math.floor(latest/5)+1},(_,i)=>i*5),xlabel:'Completed training step (not optimizer updates)',series:[
  {name:'Train / 256 episodes',color:teal,opacity:.35,width:2,points:xy(train,v=>v*100)},
  {name:'Train MA10',color:teal,width:4,points:train.filter((p,i)=>i>=9).map((p,i)=>({x:p.step,y:mean(train.slice(i,i+10).map(x=>x.value))*100}))},
  {name:'Fixed eval / 32 episodes',color:orange,width:3,dash:true,markers:true,points:xy(evaluations,v=>v*100),labels:evaluations.map(p=>Math.round(p.value*32)+'/32')}
 ]},
 kl:{id:'kl',title:'Policy update: approximate KL',xmax:latest,ymax:Math.max(.12,...d.scalars['train/actor/approx_kl'].map(p=>p.value))*1.05,yticks:[0,.04,.08,.12],xticks:[0,10,20,30],xlabel:'Completed training step',series:[{name:'Approx. KL',color:teal,points:xy(d.scalars['train/actor/approx_kl'],v=>v)}]},
 grad:{id:'grad',title:'Policy update: gradient norm',xmax:latest,ymax:Math.max(...d.scalars['train/actor/grad_norm'].map(p=>p.value))*1.12,yticks:[0,10,20,30],xticks:[0,10,20,30],xlabel:'Completed training step',series:[{name:'Grad norm',color:orange,points:xy(d.scalars['train/actor/grad_norm'],v=>v)}]},
 gpu:{id:'gpu',title:'GPU memory during the run',unit:'GiB',xmax:hours,ymax:80,yticks:[0,20,40,60,80],xticks:[0,3,6,9,12],xlabel:'Hours since launch',series:['4','5'].map((g,i)=>({name:'GPU '+g,color:i?orange:teal,opacity:.75,width:1.8,points:resource.map((p,j)=>({x:elapsed[j],y:Number(p['gpu'+g+'_used_mib'])/1024}))}))},
 ram:{id:'ram',title:'Host RAM available (whole server)',unit:'TiB',xmax:hours,ymax:2,yticks:[0,.5,1,1.5,2],xticks:[0,3,6,9,12],xlabel:'Hours since launch',series:[{name:'MemAvailable',color:teal,points:resource.map((p,j)=>({x:elapsed[j],y:Number(p.host_mem_available_kib)/2**30}))}]}
};
function plot(spec,y,h=340){
 const left=92,right=45,top=y+63,bottom=y+h-56,plotW=W-left-right,plotH=bottom-top;
 const fx=x=>left+x/spec.xmax*plotW,fy=v=>bottom-v/spec.ymax*plotH;
 let s=`<g data-plot="${spec.id}"><text x="${left}" y="${y+30}" font-size="24" font-weight="600">${esc(spec.title)}${spec.unit?' ('+spec.unit+')':''}</text>`;
 for(const v of spec.yticks)s+=`<line x1="${left}" y1="${fy(v)}" x2="${W-right}" y2="${fy(v)}" stroke="${grid}"/><text x="${left-15}" y="${fy(v)+6}" text-anchor="end" font-size="17" fill="${muted}">${v}</text>`;
 for(const v of spec.xticks.filter(x=>x<=spec.xmax))s+=`<text x="${fx(v)}" y="${bottom+26}" text-anchor="middle" font-size="17" fill="${muted}">${v}</text>`;
 s+=`<text x="${left+plotW/2}" y="${bottom+49}" text-anchor="middle" font-size="17" fill="${muted}">${esc(spec.xlabel)}</text>`;
 let lx=left;
 for(const a of spec.series){
  const pts=a.points.filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
  s+=`<path d="${pts.map((p,i)=>(i?'L':'M')+fx(p.x).toFixed(2)+','+fy(p.y).toFixed(2)).join(' ')}" fill="none" stroke="${a.color}" stroke-width="${a.width||3}" opacity="${a.opacity||1}" ${a.dash?'stroke-dasharray="9 6"':''}/>`;
  if(a.markers)pts.forEach((p,i)=>{s+=`<circle cx="${fx(p.x)}" cy="${fy(p.y)}" r="6" fill="white" stroke="${a.color}" stroke-width="3"/><text x="${fx(p.x)}" y="${fy(p.y)-14}" text-anchor="middle" font-size="20" fill="${a.color}" font-weight="600">${a.labels[i]}</text>`});
  s+=`<line x1="${lx}" y1="${y+50}" x2="${lx+26}" y2="${y+50}" stroke="${a.color}" stroke-width="3"/><text x="${lx+34}" y="${y+55}" font-size="16" fill="${muted}">${esc(a.name)}</text>`;
  lx+=a.name.length*8.5+58;
 }
 s+=`<line class="hoverline" x1="0" x2="0" y1="${top}" y2="${bottom}" stroke="${navy}" stroke-dasharray="3 3" visibility="hidden"/><rect class="hoverarea" x="${left}" y="${top}" width="${plotW}" height="${plotH}" fill="transparent" data-id="${spec.id}" data-left="${left}" data-width="${plotW}"/></g>`;
 return s;
}
function svg(title,body,height,subtitle){return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${height}" viewBox="0 0 ${W} ${height}"><rect width="100%" height="100%" fill="#f6f9fb"/><g font-family="Segoe UI,Arial,sans-serif" fill="${navy}"><text x="52" y="48" font-size="30" font-weight="700">${title}</text><text x="52" y="80" font-size="18" fill="${muted}">${esc(subtitle||stamp)}</text>${body}<text x="52" y="${height-18}" font-size="15" fill="${muted}">Live read-only evidence | Step 1 is the first observation; no fabricated Step 0 value.</text></g></svg>`}
const summary=`Step ${latest}/100   |   Train ${pct(train.at(-1).value)}   |   MA5 ${ma5.toFixed(2)}%   |   MA10 ${ma10.toFixed(2)}%`;
const success=svg('Sidney pi0.5 / move_pillbottle_pad',`<text x="52" y="115" font-size="21" font-weight="600">${summary}</text>`+plot(seriesSpec.success,135,375)+`<text x="92" y="553" font-size="18" fill="${muted}">Step 35 eval: 19/32, matching Step 10 best. Sustained improvement is not yet established.</text>`,600,stamp+' | GRPO | 64 env x 4 rollouts | horizon 200');
const optim=svg('Sidney: optimization diagnostics',plot(seriesSpec.kl,95,300)+plot(seriesSpec.grad,420,300),770);
const resources=svg('Sidney: resource history',plot(seriesSpec.gpu,95,300)+plot(seriesSpec.ram,420,300),770,stamp+' | Memory is phase-dependent; host RAM includes other jobs.');
const dashboard=svg('Sidney pi0.5: live experiment snapshot',`<text x="52" y="114" font-size="21" font-weight="600">${summary}</text>`+plot(seriesSpec.success,135,370)+plot(seriesSpec.kl,535,300)+plot(seriesSpec.grad,860,300)+plot(seriesSpec.gpu,1185,300)+plot(seriesSpec.ram,1510,300),1860,stamp+' | Hover charts to inspect recorded values.');
const html=`<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sidney experiment snapshot</title><style>body{margin:0;background:#eaf0f4;color:${navy};font:16px 'Segoe UI','Microsoft YaHei',sans-serif}main{max-width:1100px;margin:24px auto;background:#f6f9fb;border-radius:16px;overflow:hidden}svg{width:100%;height:auto}header{padding:20px 40px 0}header a{color:${teal};margin-right:20px}.tip{position:fixed;display:none;pointer-events:none;padding:12px 16px;background:#162b3ded;color:white;border-radius:8px;max-width:340px;font-size:14px;white-space:pre-line;z-index:3}</style><main><header>只读快照：训练、固定评估、优化与资源分开呈现。<p><a href="success.png">成功率 PNG</a><a href="optimization.png">优化 PNG</a><a href="resources.png">资源 PNG</a></p></header>${dashboard}</main><div class="tip"></div><script>const specs=${JSON.stringify(seriesSpec)};const tip=document.querySelector('.tip');document.querySelectorAll('.hoverarea').forEach(el=>{const g=el.parentNode,line=g.querySelector('.hoverline'),sp=specs[el.dataset.id];el.addEventListener('pointermove',e=>{const svg=el.closest('svg'),p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const q=p.matrixTransform(svg.getScreenCTM().inverse()),x=(q.x-Number(el.dataset.left))/Number(el.dataset.width)*sp.xmax;line.setAttribute('x1',q.x);line.setAttribute('x2',q.x);line.setAttribute('visibility','visible');const text=sp.series.map(s=>{const a=s.points.reduce((best,v)=>Math.abs(v.x-x)<Math.abs(best.x-x)?v:best,s.points[0]);return s.name+': '+a.y.toFixed(3)+' (x='+a.x.toFixed(2)+')'}).join(' | ');tip.textContent=text;tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-350)+'px';tip.style.top=Math.max(10,e.clientY-90)+'px'});el.addEventListener('pointerleave',()=>{line.setAttribute('visibility','hidden');tip.style.display='none'})});</script></html>`;
(async()=>{
 fs.mkdirSync(output,{recursive:true});
 for(const [name,content] of Object.entries({success,optimization:optim,resources}))await sharp(Buffer.from(content)).png().toFile(path.join(output,name+'.png'));
 fs.writeFileSync(path.join(output,'dashboard.html'),html);
 fs.writeFileSync(path.join(output,'data.json'),JSON.stringify(d,null,2));
 console.log(JSON.stringify({output,cst:d.cst,latest,success:train.at(-1).value,ma5,ma10,evaluations:evaluations.map(x=>({step:x.step,success:x.value*32})),resources:{first:resource[0],last:resource.at(-1)}}));
})();
