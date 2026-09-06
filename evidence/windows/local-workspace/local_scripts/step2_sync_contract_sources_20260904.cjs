const sv='https://raw.githubusercontent.com/haosulab/sapien-vulkan-2/d8516a4f1467167122ae85f53a8532dbceb1eec2/';
const urls=[
 sv+'src/core/rt.cpp',sv+'include/svulkan2/core/rt.h',sv+'src/renderer/renderer.cpp',sv+'src/core/buffer.cpp',
 'https://api.github.com/repos/haosulab/sapien-vulkan-2/git/trees/d8516a4f1467167122ae85f53a8532dbceb1eec2?recursive=1',
 'https://api.github.com/repos/haosulab/sapien-vulkan-2/commits/28852c021815b9b62f4442240e07376fb7f93936',
 'https://api.github.com/repos/haosulab/sapien-vulkan-2/commits/670c2948a6481368bf57aa9ad576808c42612a0b',
 'https://api.github.com/search/issues?q='+encodeURIComponent('"mASUpdateCommandBuffer"')+'&per_page=10',
 'https://api.github.com/search/issues?q='+encodeURIComponent('"SAPIEN" "synchronization"')+'&per_page=20',
 'https://api.github.com/search/issues?q='+encodeURIComponent('repo:haosulab/SAPIEN "thread"')+'&per_page=20',
];
Promise.all(urls.map(async url=>{try{const r=await fetch(url,{headers:{'User-Agent':'Read-only-render-sync-audit'},signal:AbortSignal.timeout(25000)});const body=await r.text();let result=body;if(url.includes('api.github.com')){const j=JSON.parse(body);if(j.tree)result=j.tree.filter(x=>/src\/core\/|include\/svulkan2\/core\//.test(x.path));else if(j.files)result={sha:j.sha,message:j.commit.message,files:j.files.filter(x=>/rt_renderer|scene.cpp/.test(x.filename)).map(x=>({name:x.filename,patch:x.patch}))};else if(j.items)result={count:j.total_count,items:j.items.map(x=>({url:x.html_url,title:x.title,body:x.body?.slice(0,3000)}))};else result=j;}console.log(JSON.stringify({url,status:r.status,result}));}catch(e){console.log(JSON.stringify({url,error:e.message}));}}));
