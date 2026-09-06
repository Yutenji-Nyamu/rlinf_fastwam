// Read-only, pinned upstream sources and narrowly targeted issue/commit searches.
const sv='https://raw.githubusercontent.com/haosulab/sapien-vulkan-2/d8516a4f1467167122ae85f53a8532dbceb1eec2/';
const urls=[
 ...['src/core/queue.cpp','include/svulkan2/core/queue.h','src/scene/scene.cpp','src/core/image.cpp','include/svulkan2/scene/scene.h'].map(p=>sv+p),
 'https://raw.githubusercontent.com/haosulab/SAPIEN/d8228489d05775b8615ef3edd1d47fadf25d6d7a/src/sapien_renderer/sapien_renderer_system.cpp',
 'https://raw.githubusercontent.com/haosulab/sapien-vulkan-2/4914f8747a8cf9f0138c8dbc93e972df19a307d0/src/renderer/rt_renderer.cpp',
 'https://api.github.com/repos/haosulab/sapien-vulkan-2/commits?path=src/renderer/rt_renderer.cpp&per_page=35',
 'https://api.github.com/repos/haosulab/SAPIEN/commits?path=src/sapien_renderer/camera_component.cpp&per_page=25',
 ...['repo:haosulab/sapien-vulkan-2 synchronization','repo:haosulab/sapien-vulkan-2 hang','repo:haosulab/SAPIEN denoiser','repo:haosulab/SAPIEN thread safe','repo:RLinf/RLinf robotwin stuck','repo:RoboTwin-Platform/RoboTwin denoiser'].map(q=>'https://api.github.com/search/issues?q='+encodeURIComponent(q)+'&per_page=20')
];
Promise.all(urls.map(async url=>{
 try {const r=await fetch(url,{headers:{'User-Agent':'Read-only-causal-investigation'},signal:AbortSignal.timeout(25000)});const body=await r.text();let result=body;
 if(url.includes('api.github.com')) {const j=JSON.parse(body); if(url.includes('/search/'))result={count:j.total_count,items:j.items?.map(x=>({url:x.html_url,title:x.title,body:x.body,state:x.state}))};else if(Array.isArray(j))result=j.map(x=>({sha:x.sha,url:x.html_url,message:x.commit.message,date:x.commit.author.date}));else result=j;}
 console.log(JSON.stringify({url,status:r.status,result}));}catch(e){console.log(JSON.stringify({url,error:e.message}));}
}));
