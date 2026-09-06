// Decode a complete read-only transport payload into the existing artifact format.
const fs=require('fs'),zlib=require('zlib');
const [input,output]=process.argv.slice(2);
const text=fs.readFileSync(input,'utf8');
const line=text.split(/\r?\n/).find(x=>x.startsWith('SNAPSHOT_ZLIB '));
if(!line)throw Error('Missing compressed snapshot');
const payload=zlib.inflateSync(Buffer.from(line.slice(14),'base64')).toString('utf8');
const data=JSON.parse(payload);
fs.writeFileSync(output,'SNAPSHOT_JSON '+payload+'\n'+text.split(/\r?\n/).filter(x=>!x.startsWith('SNAPSHOT_ZLIB ')).join('\n'));
console.log(JSON.stringify({time:data.time,end:data.end_time,bytes:payload.length,runs:Object.fromEntries(Object.entries(data.runs).map(([k,r])=>[k,{step:r.state.completed_step,alive:r.state.wrapper_alive,exit:r.state['exit_code.txt'],errors:r.state.errors,train:r.scalars['env/success_once']?.slice(-10),eval:r.scalars['eval/success_once'],checkpoints:r.checkpoints.slice(-1),phase:r.state.phase.slice(-3)}])),health:Object.fromEntries(['gpu','loadavg','pressure','vmstat','disk','core_processes','top_rss','services','failed_services','owned_workers'].map(k=>[k,data.health[k]])),git:data.git},null,2));
