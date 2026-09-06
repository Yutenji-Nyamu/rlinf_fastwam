"""Create only the final documentation/script delta for the evidence branch."""
import argparse,getpass,tarfile
from pathlib import Path
from lightweight_archive_20260906 import Archive
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--name',default='closeout');args=p.parse_args()
assert args.name in {'closeout','publication-receipt'}
dest=root/'.tmp/archive-packet-20260906'/args.name
arc=Archive(dest,getpass.getpass('Known credential to redact (process only): '))
files=[root/n for n in ['AGENTS.md','PROJECT_CONTEXT.md','HANDOFF.md',
 'docs/server-admin/SZ_ARCHIVE_CLEANUP_CONTEXT_RESULT_20260906.md',
 'docs/server-admin/SZ_ARCHIVE_CLEANUP_CONTEXT_LEDGER_20260906.md',
 'docs/server-admin/SZ_ARCHIVE_RUNTIME_SOURCE_RECHECK_20260906.txt',
 'docs/server-admin/SZ_ARCHIVE_IMPORT_COMMIT_20260906.txt',
 'docs/server-admin/SZ_ARCHIVE_PUBLISH_20260906.txt',
 'docs/rlinf-robotwin-pi0-online-bc/evidence/PI05_ARCHIVE_CLOSEOUT_LIVE_20260906.json',
 'docs/rlinf-robotwin-pi0-online-bc/evidence/PI05_BC_DVAC_FULL_DIFFERENCE_AUDIT_20260906.md']]
files+=list((root/'local_scripts').glob('*20260906.py'))
files+=list((root/'local_scripts/remote_commands').glob('*20260906*'))
for file in sorted(set(files)):
 if file.is_file():arc.file(file,'local-workspace/'+file.relative_to(root).as_posix())
arc.finish({'scope':'Final task delta overriding earlier point-in-time copies; models/data not included'})
for name in ['archive-manifest.jsonl.gz','archive-summary.json']:
 # Rename only generated metadata to mark this as a delta, not a full manifest.
 q=dest/name;new=dest/'late-final'/name;new.parent.mkdir(exist_ok=True);q.rename(new)
packet=dest.parent/(args.name+'.tar.gz');assert not packet.exists()
with tarfile.open(packet,'w:gz',compresslevel=3) as tar:
 for file in dest.rglob('*'):
  if file.is_file():tar.add(file,arcname=file.relative_to(dest).as_posix(),recursive=False)
print('CLOSEOUT_PACKET_BYTES='+str(packet.stat().st_size),flush=True)
