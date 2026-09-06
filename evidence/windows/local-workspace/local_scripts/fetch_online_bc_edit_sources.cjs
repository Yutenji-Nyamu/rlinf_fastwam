const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const cp = require('node:child_process');
const pin = 'dc9b87cc49334c7516487ead68ebeb060fd7c090';
const repo = path.resolve('.research-rlinf');
const root = path.resolve('worktrees/pi0-online-bc');
const files = [
  'rlinf/config.py', 'examples/embodiment/train_embodied_agent.py',
  'rlinf/workers/env/env_worker.py', 'rlinf/workers/rollout/hf/huggingface_worker.py',
  'rlinf/workers/actor/fsdp_dagger_policy_worker.py', 'rlinf/workers/actor/embodied_fsdp_actor_worker.py',
  'rlinf/models/embodiment/openpi/openpi_action_model.py',
  'rlinf/runners/embodied_runner.py', 'rlinf/envs/robotwin/robotwin_env.py',
  'rlinf/envs/utils.py', 'rlinf/data/schema/embodied_trajectory_builder.py',
  'rlinf/data/storage/replay/trajectory_replay_buffer.py',
  'examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml',
  'examples/embodiment/config/env/robotwin_adjust_bottle.yaml',
  'examples/embodiment/config/model/pi0.yaml',
  'examples/embodiment/config/hybrid_engines/fsdp.yaml',
];
async function main() {
  const tree = cp.execFileSync('git', ['-c', `safe.directory=${repo.replaceAll('\\','/')}`, '-C', repo, 'ls-tree', '-r', pin], {encoding:'utf8'});
  const expected = new Map(tree.trim().split('\n').map(l => { const [m,p] = l.split('\t'); return [p,m.split(' ')[2]]; }));
  for (const rel of files) {
    const dest = path.join(root,rel);
    if (fs.existsSync(dest)) throw new Error(`Existing target: ${dest}`);
    if (!expected.has(rel)) { console.log('PATH_NOT_IN_PIN '+rel); continue; }
    const cache = path.join('docs/rlinf-robotwin-pi0-online-bc/evidence/source-audit-20260904/RLinf__RLinf/source',rel);
    let b;
    if (fs.existsSync(cache)) b = fs.readFileSync(cache);
    else {
      const r = await fetch(`https://raw.githubusercontent.com/RLinf/RLinf/${pin}/${rel}`, {signal:AbortSignal.timeout(20000)});
      if (!r.ok) throw new Error(`${r.status} ${rel}`);
      b = Buffer.from(await r.arrayBuffer());
    }
    const sha = crypto.createHash('sha1').update(`blob ${b.length}\0`).update(b).digest('hex');
    if (sha !== expected.get(rel)) throw new Error(`Official blob mismatch: ${rel}`);
    fs.mkdirSync(path.dirname(dest), {recursive:true});
    fs.writeFileSync(dest,b);
    console.log('VERIFIED '+rel);
  }
}
main().catch(e=>{console.error(e.message);process.exitCode=1;});
