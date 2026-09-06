const urls=[
 'https://api.github.com/repos/RenderKit/oidn/commits/9f816f77eb3d6bddaf8d07a96c480444f3d0ee4b',
 'https://api.github.com/repos/RenderKit/oidn/commits/d4af2c667497a42a5c06c95e6d6b5d7f5cd35349',
 'https://raw.githubusercontent.com/RenderKit/oidn/v2.0.1/api/api.cpp',
 'https://raw.githubusercontent.com/RenderKit/oidn/v2.0.1/include/OpenImageDenoise/oidn.hpp',
 'https://raw.githubusercontent.com/haosulab/SAPIEN/d8228489d05775b8615ef3edd1d47fadf25d6d7a/src/sapien_renderer/sapien_renderer_system.cpp'
];
(async()=>{await Promise.all(urls.map(async url=>{
 try{const r=await fetch(url,{headers:{'User-Agent':'Read-only-OIDN-investigation'},signal:AbortSignal.timeout(20000)});const s=await r.text();let result=s;
 if(url.includes('/commits/')){const d=JSON.parse(s);result={sha:d.sha,date:d.commit?.committer?.date,message:d.commit?.message,files:d.files?.map(f=>({filename:f.filename,patch:f.patch,raw_url:f.raw_url}))}}
 console.log(JSON.stringify({url,status:r.status,result}));
 }catch(e){console.log(JSON.stringify({url,error:e.message}))}
}));})();
