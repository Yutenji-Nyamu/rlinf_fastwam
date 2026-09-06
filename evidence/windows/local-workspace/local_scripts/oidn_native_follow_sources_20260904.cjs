const sap='https://raw.githubusercontent.com/haosulab/SAPIEN/d8228489d05775b8615ef3edd1d47fadf25d6d7a/';
const sv='https://raw.githubusercontent.com/haosulab/sapien-vulkan-2/d8516a4f1467167122ae85f53a8532dbceb1eec2/';
const urls=[...['src/sapien_renderer/sapien_renderer_default.cpp','src/sapien_renderer/camera_component.cpp','python/pybind/sapien_renderer.cpp','src/scene.cpp'].map(p=>sap+p),sv+'src/renderer/rt_renderer.cpp',sv+'src/resource/manager.cpp',sv+'include/svulkan2/resource/manager.h','https://raw.githubusercontent.com/RenderKit/oidn/v2.0.1/core/scratch.cpp','https://raw.githubusercontent.com/RenderKit/oidn/v2.0.1/core/scratch.h','https://api.github.com/repos/RenderKit/oidn/compare/v2.0.1...v2.1.0'];
(async()=>{await Promise.all(urls.map(async url=>{
 try{const r=await fetch(url,{headers:{'User-Agent':'Read-only-OIDN-investigation'},signal:AbortSignal.timeout(20000)});const s=await r.text();let result=s;
 if(url.includes('/compare/')){const d=JSON.parse(s);result={commits:d.commits?.map(x=>({sha:x.sha,message:x.commit.message})),files:d.files?.filter(x=>/^core\/(buffer|engine|scratch|device)|^devices\/cuda\/(cuda_device|cuda_engine)/.test(x.filename)).map(x=>({filename:x.filename,patch:x.patch,raw_url:x.raw_url}))}}
 console.log(JSON.stringify({url,status:r.status,result}));
 }catch(e){console.log(JSON.stringify({url,error:e.message}))}
}));})();
