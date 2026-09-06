const urls = [
 'https://raw.githubusercontent.com/RenderKit/oidn/v2.0.1/core/thread.cpp',
 'https://raw.githubusercontent.com/RenderKit/oidn/v2.0.1/core/thread.h',
 'https://raw.githubusercontent.com/RenderKit/oidn/v2.0.1/core/device.cpp',
 'https://raw.githubusercontent.com/RenderKit/oidn/v2.0.1/devices/cuda/cuda_device.cpp',
 'https://raw.githubusercontent.com/python/cpython/v3.11.14/Python/pystate.c',
 'https://raw.githubusercontent.com/python/cpython/v3.11.14/Python/thread_pthread.h',
 'https://raw.githubusercontent.com/bminor/glibc/glibc-2.35/nptl/pthread_key_create.c',
 'https://api.github.com/repos/haosulab/SAPIEN/git/trees/v3.0.1?recursive=1',
 'https://api.github.com/repos/haosulab/svulkan2/git/trees/master?recursive=1',
];
(async()=>{
 const results=await Promise.all(urls.map(async url=>{
  try {
   const response=await fetch(url,{signal:AbortSignal.timeout(20000)});const content=await response.text();
   if(url.includes('api.github.com')){
    const d=JSON.parse(content);return {url,status:response.status,sha:d.sha,paths:d.tree?.filter(x=>/svulkan|oidn|denois/i.test(x.path)),error:d.message};
   }
   const lines=content.split('\n'),selected=new Set();
   const pat=/pthread_key|ThreadLocal|thread_local|autoTSSkey mapping|PyThread_tss_(set|create)|EAGAIN|~Device|~CUDADevice|cleanup|threadLocal/;
   for(let i=0;i<lines.length;i++)if(pat.test(lines[i]))for(let j=Math.max(0,i-8);j<Math.min(lines.length,i+18);j++)selected.add(j);
   return {url,status:response.status,lines:[...selected].sort((a,b)=>a-b).map(i=>`${i+1}: ${lines[i]}`)};
  }catch(e){return {url,error:String(e)}}
 }));for(const r of results)console.log(JSON.stringify(r));
})();
