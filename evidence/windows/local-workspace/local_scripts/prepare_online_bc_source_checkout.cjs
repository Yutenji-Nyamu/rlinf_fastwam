// Materialize official, pinned Git blobs from the author's source archive when
// smart-HTTP is unavailable. No repository config or existing checkout is edited.
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const cp = require('node:child_process');
const pin = 'dc9b87cc49334c7516487ead68ebeb060fd7c090';
const repo = path.resolve('.research-rlinf');
const target = path.resolve('worktrees/pi0-online-bc');
function git(args, input) {
  const r = cp.spawnSync('git', ['-c', `safe.directory=${repo.replaceAll('\\', '/')}`, '-C', repo, ...args],
    {input, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024, timeout: 45000});
  if (r.status !== 0) throw new Error(r.stderr || String(r.error));
  return r.stdout;
}
async function main() {
  if (fs.existsSync(target)) throw new Error('Target already exists; inspect before retrying');
  const response = await fetch(`https://codeload.github.com/RLinf/RLinf/tar.gz/${pin}`,
    {signal: AbortSignal.timeout(45000)});
  if (!response.ok) throw new Error(`Archive HTTP ${response.status}`);
  const zipped = Buffer.from(await response.arrayBuffer());
  const tar = zlib.gunzipSync(zipped, {maxOutputLength: 200 * 1024 * 1024});
  const temp = fs.mkdtempSync(path.resolve('local_scripts/online-bc-source-'));
  const prefix = `RLinf-${pin}/`;
  const paths = [];
  for (let offset = 0; offset + 512 <= tar.length;) {
    const h = tar.subarray(offset, offset + 512);
    if (h.every(b => b === 0)) break;
    let name = h.subarray(0, 100).toString().split('\0')[0];
    const dir = h.subarray(345, 500).toString().split('\0')[0];
    if (dir) name = `${dir}/${name}`;
    const size = parseInt(h.subarray(124, 136).toString().replace(/\0/g, '').trim() || '0', 8);
    const type = h[156];
    if (name.startsWith(prefix) && (type === 0 || type === 48)) {
      const rel = name.slice(prefix.length);
      if (rel.split('/').includes('..') || path.isAbsolute(rel)) throw new Error('Unsafe archive member');
      const dest = path.join(temp, rel);
      fs.mkdirSync(path.dirname(dest), {recursive: true});
      fs.writeFileSync(dest, tar.subarray(offset + 512, offset + 512 + size));
      paths.push(dest.replaceAll('\\', '/'));
    }
    offset += 512 + Math.ceil(size / 512) * 512;
  }
  const hashes = git(['hash-object', '-w', '--no-filters', '--stdin-paths'], paths.join('\n') + '\n').trim().split('\n');
  const expected = new Map(git(['ls-tree', '-r', pin]).trim().split('\n').map(line => {
    const [meta, rel] = line.split('\t');
    return [rel, meta.split(' ')[2]];
  }));
  for (let i = 0; i < paths.length; i++) {
    const rel = path.relative(temp, paths[i]).replaceAll('\\', '/');
    if (expected.get(rel) !== hashes[i]) throw new Error(`Blob mismatch: ${rel}`);
  }
  console.log(JSON.stringify({compressed_bytes: zipped.length, files: paths.length, all_downloaded_blobs_match: true, cache: temp}));
  console.log(git(['worktree', 'add', target, 'codex/sz-pi0-online-bc']));
}
main().catch(e => { console.error(e.message); process.exitCode = 1; });
