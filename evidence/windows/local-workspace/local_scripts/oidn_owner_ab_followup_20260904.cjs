const sv='https://raw.githubusercontent.com/haosulab/sapien-vulkan-2/d8516a4f1467167122ae85f53a8532dbceb1eec2/';
const sap='https://raw.githubusercontent.com/haosulab/SAPIEN/d8228489d05775b8615ef3edd1d47fadf25d6d7a/';
const oidn='https://raw.githubusercontent.com/RenderKit/oidn/v2.0.1/';
const urls=[sv+'src/renderer/denoiser_oidn.cpp',sv+'src/renderer/denoiser.h',sv+'src/renderer/rt_renderer.cpp',sap+'src/sapien_renderer/camera_component.cpp',oidn+'core/device.cpp',oidn+'core/device.h',oidn+'core/thread.h',oidn+'doc/api.md'];
(async()=>{await Promise.all(urls.map(async url=>{try{const r=await fetch(url,{headers:{'User-Agent':'Read-only-OIDN-owner-review'},signal:AbortSignal.timeout(25000)});console.log(JSON.stringify({url,status:r.status,text:await r.text()}));}catch(e){console.log(JSON.stringify({url,error:e.message}))}}));})();
