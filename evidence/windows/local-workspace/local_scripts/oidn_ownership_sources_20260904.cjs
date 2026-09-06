const base='https://raw.githubusercontent.com/RenderKit/oidn/v2.0.1/';
const urls=['core/device.h','core/engine.h','core/engine.cpp','core/buffer.h','core/buffer.cpp','core/memory.h','core/tensor.h','devices/cuda/cuda_engine.h','devices/cuda/cuda_engine.cu','devices/cuda/cuda_device.h','core/unet_filter.cpp'].map(p=>base+p);
urls.push('https://api.github.com/repos/haosulab/SAPIEN/git/trees/d8228489d05775b8615ef3edd1d47fadf25d6d7a?recursive=1','https://api.github.com/repos/haosulab/sapien-vulkan-2/git/trees/d8516a4f1467167122ae85f53a8532dbceb1eec2?recursive=1');
(async()=>{await Promise.all(urls.map(async url=>{
 try{const r=await fetch(url,{headers:{'User-Agent':'Read-only-OIDN-investigation'},signal:AbortSignal.timeout(20000)});const s=await r.text();
 let result=s;
 if(url.includes('/git/trees/')){const d=JSON.parse(s);result=d.tree?.filter(x=>/render_system|resource_manager|render_camera|renderer|render\.cpp|scene\.cpp|entity\.cpp|sapien_renderer|CMakeLists/.test(x.path)).map(x=>({path:x.path,sha:x.sha,type:x.type}))||d}
 console.log(JSON.stringify({url,status:r.status,result}));
 }catch(e){console.log(JSON.stringify({url,error:e.message}))}
}));})();
