const urls=[
 'https://api.github.com/search/issues?q=repo:RLinf/RLinf+robotwin&per_page=100',
 'https://api.github.com/search/issues?q=repo:RLinf/RLinf+OIDN&per_page=100',
 'https://api.github.com/search/issues?q=repo:RLinf/RLinf+pthread_key_create&per_page=100',
 'https://api.github.com/search/issues?q=repo:RoboTwin-Platform/RoboTwin+memory+leak&per_page=100',
 'https://api.github.com/repos/RoboTwin-Platform/RoboTwin/issues/89',
 'https://api.github.com/repos/RoboTwin-Platform/RoboTwin/issues/89/comments',
 'https://api.github.com/repos/haosulab/SAPIEN/issues/219',
 'https://api.github.com/repos/haosulab/SAPIEN/issues/219/comments',
 'https://api.github.com/repos/RLinf/RLinf/pulls/897',
 'https://api.github.com/repos/RLinf/RLinf/pulls/897/files'
];
(async()=>{for(let i=0;i<urls.length;i+=3)await Promise.all(urls.slice(i,i+3).map(async url=>{
 try{const r=await fetch(url,{headers:{'User-Agent':'Read-only-FastWAM-comparison'},signal:AbortSignal.timeout(25000)});const d=await r.json();
 const brief=x=>({url:x.html_url,title:x.title,body:x.body,state:x.state,comments:x.comments,merged_at:x.merged_at,merge_commit_sha:x.merge_commit_sha,sha:x.sha,filename:x.filename,patch:x.patch});
 console.log(JSON.stringify({url,status:r.status,total_count:d.total_count,incomplete_results:d.incomplete_results,result:d.items?d.items.map(brief):Array.isArray(d)?d.map(brief):brief(d)}));
 }catch(e){console.log(JSON.stringify({url,error:e.message}))}
}));})();
