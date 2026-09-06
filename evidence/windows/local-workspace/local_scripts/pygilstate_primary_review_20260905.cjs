const fs = require('fs');
const out = 'docs/fastwam-robotwin-rlinf-grpo/evidence/PYGILSTATE_PRIMARY_SOURCES_20260905.jsonl';
const sap = 'https://raw.githubusercontent.com/haosulab/SAPIEN/d8228489d05775b8615ef3edd1d47fadf25d6d7a/';
const urls = [
  'https://raw.githubusercontent.com/python/cpython/v3.11.14/Python/pystate.c',
  ...['.gitmodules','CMakeLists.txt','cmake/pybind11.cmake','python/pybind/sapien_renderer.cpp','python/pybind/pysapien.cpp','python/pybind/entity.cpp','python/pybind/component.cpp','python/pybind/scene.cpp','src/scene.cpp','src/entity.cpp','src/component.cpp'].map(p=>sap+p),
  'https://api.github.com/repos/haosulab/SAPIEN/git/trees/d8228489d05775b8615ef3edd1d47fadf25d6d7a?recursive=1',
  ...['repo:haosulab/SAPIEN PyGILState_Release','repo:RoboTwin-Platform/RoboTwin PyGILState_Release','repo:RLinf/RLinf PyGILState_Release','repo:pybind/pybind11 "auto-releasing thread-state"'].map(q=>'https://api.github.com/search/issues?q='+encodeURIComponent(q)+'&per_page=10'),
];
const terms = /PyGILState|gil_scoped|PYBIND11|pybind11|std::async|std::thread|ThreadState_Delete|autoTSSkey|~Scene|~Component|~Entity/;
(async()=>{
  const results = await Promise.all(urls.map(async url=>{
    try {
      const r=await fetch(url,{headers:{'User-Agent':'Read-only-PyGILState-source-review'},signal:AbortSignal.timeout(25000)});
      const body=await r.text();
      let result;
      if(url.includes('/search/')){const j=JSON.parse(body);result={count:j.total_count,items:j.items?.map(x=>({url:x.html_url,title:x.title,body:x.body,state:x.state}))};}
      else if(url.includes('/git/trees/')){const j=JSON.parse(body);result=j.tree?.filter(x=>/pybind|cmake|thread/.test(x.path)).map(x=>({path:x.path,sha:x.sha}));}
      else {const lines=body.split('\n');const keep=new Set();for(let i=0;i<lines.length;i++)if(terms.test(lines[i]))for(let j=Math.max(0,i-5);j<Math.min(lines.length,i+15);j++)keep.add(j);result=[...keep].sort((a,b)=>a-b).map(i=>`${i+1}: ${lines[i]}`).join('\n');}
      return {url,status:r.status,result};
    }catch(e){return {url,error:e.message};}
  }));
  fs.writeFileSync(out,results.map(x=>JSON.stringify(x)).join('\n')+'\n');
  for(const r of results)console.log(JSON.stringify({url:r.url,status:r.status,error:r.error,result:typeof r.result==='string'?r.result.slice(0,16000):r.result}));
})();
