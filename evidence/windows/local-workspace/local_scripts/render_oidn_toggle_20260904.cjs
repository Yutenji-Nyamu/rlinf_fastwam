const fs=require('fs'),path=require('path');
const sharp=require('C:/Users/86136/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const root='docs/fastwam-robotwin-rlinf-grpo/evidence/oidn-toggle-20260904';
const on=JSON.parse(fs.readFileSync(path.join(root,'oidn/summary.json'))),off=JSON.parse(fs.readFileSync(path.join(root,'none/summary.json')));
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
const prefix=(mode,e)=>`${mode}/episode_${e.episode}_seed_${e.requested_seed}`;
const uri=p=>'data:image/png;base64,'+fs.readFileSync(path.join(root,p)).toString('base64');
(async()=>{
 const rows=[];
 for(let i=0;i<on.episodes.length;i++){
  const a=on.episodes[i],b=off.episodes[i];
  if(a.requested_seed!==b.requested_seed)throw Error('seed mismatch');
  const cameras={};
  for(const cam of ['head','left','right','model_input']){
   const aa=await sharp(path.join(root,prefix('oidn',a),`q00_${cam}.png`)).removeAlpha().raw().toBuffer({resolveWithObject:true});
   const bb=await sharp(path.join(root,prefix('none',b),`q00_${cam}.png`)).removeAlpha().raw().toBuffer({resolveWithObject:true});
   if(aa.data.length!==bb.data.length)throw Error('image shape mismatch');
   let abs=0,sq=0,max=0;const diff=Buffer.alloc(aa.data.length);
   for(let j=0;j<aa.data.length;j++){const d=Math.abs(aa.data[j]-bb.data[j]);abs+=d;sq+=d*d;max=Math.max(max,d);diff[j]=Math.min(255,d*4)}
   cameras[cam]={mae_255:abs/aa.data.length,rmse_255:Math.sqrt(sq/aa.data.length),max_abs:max};
   if(cam==='model_input')await sharp(diff,{raw:aa.info}).png().toFile(path.join(root,`seed_${a.requested_seed}_diff4.png`));
  }
  const joint=[],grip=[];
  for(let q=0;q<24;q++)for(let d=0;d<14;d++)(d===6||d===13?grip:joint).push(Math.abs(a.first_actions[q][d]-b.first_actions[q][d]));
  rows.push({seed:a.requested_seed,on_success:a.success,off_success:b.success,on_queries:a.queries.length,off_queries:b.queries.length,state_max_abs:Math.max(...a.initial_state.flat().map((x,j)=>Math.abs(x-b.initial_state.flat()[j]))),same_prompt:JSON.stringify(a.instruction)===JSON.stringify(b.instruction),cameras,first_action_joint_mae:mean(joint),first_action_joint_max:Math.max(...joint),first_action_gripper_mae:mean(grip),first_action_gripper_max:Math.max(...grip)});
 }
 const summary={rows,on_elapsed:on.elapsed_seconds,off_elapsed:off.elapsed_seconds,on_success:on.episodes.filter(x=>x.success).length,off_success:off.episodes.filter(x=>x.success).length,on_calls:on.denoiser_calls,off_calls:off.denoiser_calls,note:'Small qualitative trial with release SFT checkpoint; not a success-rate or long-run stability estimate. Pixel differences include stochastic rendering noise.'};
 fs.writeFileSync(path.join(root,'comparison.json'),JSON.stringify(summary,null,2));
 const W=1100,H=160+rows.length*460;
 let svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}"><rect width="100%" height="100%" fill="#f6f9fb"/><g font-family="Segoe UI,Arial,sans-serif" fill="#173047"><text x="40" y="40" font-size="28" font-weight="700">Fast-WAM: OIDN on / off</text><text x="40" y="72" font-size="17">3 short episodes per mode | release SFT weights | old lifecycle patch retained</text><text x="45" y="110" font-size="21" fill="#087f8c">OIDN on</text><text x="395" y="110" font-size="21" fill="#d87814">OIDN off</text><text x="745" y="110" font-size="21">Absolute pixel difference x4</text>`;
 rows.forEach((r,i)=>{const y=135+i*460,a=on.episodes[i],b=off.episodes[i];svg+=`<text x="40" y="${y}" font-size="18">Seed ${r.seed} | success: on ${r.on_success?'yes':'no'} / off ${r.off_success?'yes':'no'} | input MAE ${r.cameras.model_input.mae_255.toFixed(2)}/255</text>`;[`${prefix('oidn',a)}/q00_model_input.png`,`${prefix('none',b)}/q00_model_input.png`,`seed_${r.seed}_diff4.png`].forEach((p,j)=>svg+=`<image x="${40+j*350}" y="${y+15}" width="320" height="384" href="${uri(p)}"/>`);svg+=`<text x="40" y="${y+425}" font-size="16">First action chunk: joint mean abs delta ${r.first_action_joint_mae.toFixed(4)} rad | gripper mean abs delta ${r.first_action_gripper_mae.toFixed(4)}</text>`});
 svg+=`<text x="40" y="${H-16}" font-size="15">Initial observations shown. Human visual similarity does not imply identical policy behavior.</text></g></svg>`;
 fs.writeFileSync(path.join(root,'comparison.svg'),svg);await sharp(Buffer.from(svg)).png().toFile(path.join(root,'comparison.png'));
 const example=rows[2],ae=on.episodes[2],be=off.episodes[2];
 const brief=`<svg xmlns="http://www.w3.org/2000/svg" width="720" height="480"><rect width="100%" height="100%" fill="#f6f9fb"/><g font-family="Segoe UI,Arial,sans-serif" fill="#173047"><text x="24" y="31" font-size="23" font-weight="700">Same initial scene: OIDN on / off</text><text x="24" y="58" font-size="16">Seed ${example.seed} | actual Fast-WAM input | no training</text><text x="24" y="85" font-size="18" fill="#087f8c">OIDN on</text><text x="376" y="85" font-size="18" fill="#d87814">OIDN off</text><image x="24" y="98" width="288" height="346" href="${uri(prefix('oidn',ae)+'/q00_model_input.png')}"/><image x="376" y="98" width="288" height="346" href="${uri(prefix('none',be)+'/q00_model_input.png')}"/><text x="24" y="468" font-size="15">Off retains scene structure, but adds visible grain. Input MAE: ${example.cameras.model_input.mae_255.toFixed(2)}/255.</text></g></svg>`;
 await sharp(Buffer.from(brief)).png().toFile(path.join(root,'comparison-brief.png'));
 const cards=rows.map((r,i)=>{const a=on.episodes[i],b=off.episodes[i];return `<section><h2>Seed ${r.seed}：开 ${r.on_success?'成功':'未成功'} / 关 ${r.off_success?'成功':'未成功'}</h2><p>初始模型输入 MAE ${r.cameras.model_input.mae_255.toFixed(2)}/255；首块关节动作平均差 ${r.first_action_joint_mae.toFixed(4)} rad。仅供小样本观察。</p><div class="pair">${[['oidn',a],['none',b]].map(([m,e])=>`<article><h3>${m==='oidn'?'OIDN 开':'OIDN 关'}</h3><img class="frame" src="${prefix(m,e)}/q00_model_input.png" data-prefix="${prefix(m,e)}"><label>策略查询后画面 <input type="range" min="0" max="${e.queries.length}" value="0"><span>0</span></label><p>原始初始相机：${['head','left','right'].map(c=>`<a href="${prefix(m,e)}/q00_${c}.png">${c}</a>`).join(' / ')}</p></article>`).join('')}</div></section>`}).join('');
 fs.writeFileSync(path.join(root,'index.html'),`<!doctype html><meta charset="utf-8"><title>Fast-WAM OIDN 开关小尝试</title><style>body{font:17px system-ui;max-width:1100px;margin:30px auto;color:#173047;background:#f6f9fb;padding:0 18px}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.pair{display:flex;gap:24px}article{flex:1}img{width:100%;max-width:400px;image-rendering:auto}label{display:block}input{max-width:90%}@media(max-width:650px){.pair{gap:10px}h2{font-size:20px}}</style><h1>Fast-WAM：OIDN 开／关</h1><p>2026-09-04 · move_stapler_pad · 原始SFT权重 · 各3短回合 · 旧补丁保留 · 无训练更新</p><p>只关降噪，保留 rt / spp32 / depth8。结果不能作为长期稳定性或成功率结论。</p>${cards}<p><a href="comparison.png">汇总图</a> · <a href="comparison.json">指标原始数据</a></p><script>document.querySelectorAll('input').forEach(x=>x.oninput=()=>{const a=x.closest('article'),img=a.querySelector('img');img.src=img.dataset.prefix+'/q'+String(x.value).padStart(2,'0')+'_model_input.png';a.querySelector('span').textContent=x.value})</script>`);
 console.log(JSON.stringify(summary,null,2));
})();
