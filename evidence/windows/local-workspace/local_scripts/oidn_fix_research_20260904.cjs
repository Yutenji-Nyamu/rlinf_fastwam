const urls=[
 'https://api.github.com/repos/RenderKit/oidn/compare/v2.2.0...v2.2.1',
 'https://api.github.com/repos/RenderKit/oidn/compare/v2.2.1...v2.2.2',
 'https://api.github.com/repos/haosulab/SAPIEN/issues/243/comments',
 'https://api.github.com/search/issues?q=repo:haosulab/SAPIEN+OIDN',
 'https://api.github.com/search/issues?q=repo:RenderKit/oidn+leak',
 'https://api.github.com/search/issues?q=repo:haosulab/sapien-vulkan-2+leak',
 'https://api.github.com/search/issues?q=repo:RoboTwin-Platform/RoboTwin+OIDN'
];
(async()=>{await Promise.all(urls.map(async url=>{
 try{
  const r=await fetch(url,{headers:{'User-Agent':'Read-only-OIDN-investigation'},signal:AbortSignal.timeout(20000)});
  const d=await r.json();
  if(!r.ok){console.log(JSON.stringify({url,status:r.status,error:d.message}));return}
  let result=d;
  if(d.files)result={commits:d.commits.map(c=>({sha:c.sha,message:c.commit.message})),files:d.files.map(f=>({filename:f.filename,patch:f.patch,raw_url:f.raw_url}))};
  else if(d.items)result=d.items.map(x=>({url:x.html_url,title:x.title,state:x.state,body:x.body,comments:x.comments}));
  else if(Array.isArray(d))result=d.map(x=>({url:x.html_url,user:x.user?.login,association:x.author_association,date:x.created_at,body:x.body}));
  console.log(JSON.stringify({url,status:r.status,result}));
 }catch(e){console.log(JSON.stringify({url,error:e.message}))}
}));})();
