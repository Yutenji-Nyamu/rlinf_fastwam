// Read-only primary-source retrieval for the Step2 diagnosis.
const root='https://raw.githubusercontent.com/haosulab/SAPIEN/';
const pin='d8228489d05775b8615ef3edd1d47fadf25d6d7a/';
const specs=[
 [root+pin+'src/sapien_renderer/camera_component.cpp',[[110,145]]],
 [root+pin+'python/pybind/sapien_renderer.cpp',[[1033,1047]]],
 [root+'master/python/pybind/sapien_renderer.cpp',null],
 ['https://api.github.com/search/issues?q=repo:haosulab/SAPIEN+GIL&per_page=10','search'],
 ['https://api.github.com/search/issues?q=repo:RLinf/RLinf+robotwin+hang&per_page=10','search'],
 ['https://api.github.com/search/issues?q=repo:RoboTwin-Platform/RoboTwin+hang&per_page=10','search'],
 ['https://api.github.com/search/issues?q=repo:hungpham2511/toppra+thread&per_page=10','search'],
 ['https://api.github.com/repos/RLinf/RLinf/issues/1040/comments','comments'],
 ['https://api.github.com/repos/haosulab/SAPIEN/issues/171/comments','comments'],
];
Promise.all(specs.map(async([url,ranges])=>{
 try{
  const r=await fetch(url,{headers:{'User-Agent':'Read-only-Step2-diagnosis'},signal:AbortSignal.timeout(20000)});
  const body=await r.text();let result;
  if(ranges==='search'){const j=JSON.parse(body);result={count:j.total_count,items:j.items?.map(v=>({url:v.html_url,title:v.title,state:v.state,body:v.body?.slice(0,3500)}))};}
  else if(ranges==='comments'){const j=JSON.parse(body);result=Array.isArray(j)?j.map(v=>({url:v.html_url,date:v.updated_at,body:v.body})):j;}
  else{const a=body.split('\n');let ii=[];if(ranges){for(const [lo,hi] of ranges)for(let i=lo-1;i<Math.min(a.length,hi);i++)ii.push(i);}else{a.forEach((s,i)=>{if(/get_picture|gil_scoped/.test(s))for(let j=Math.max(0,i-3);j<Math.min(a.length,i+7);j++)ii.push(j)});}result=[...new Set(ii)].sort((a,b)=>a-b).map(i=>`${i+1}: ${a[i]}`).join('\n');}
  console.log(JSON.stringify({url,status:r.status,result}));
 }catch(e){console.log(JSON.stringify({url,error:e.message}));}
}));
