"""Review only generated staging; exclude opaque Git packs and create final packet."""
import datetime, getpass, gzip, hashlib, json, os, tarfile
from pathlib import Path
from lightweight_archive_20260906 import Archive

root=Path(__file__).resolve().parents[1]
staging=root/'.tmp/archive-packet-20260906/staging'
packet=staging.parent/'local-evidence-reviewed.tar.gz'
assert staging.resolve().is_relative_to(root.resolve()) and staging.name=='staging' and not packet.exists()
manifest=staging/'archive-manifest.jsonl.gz'
assert manifest.is_file(), 'Initial archive walk must finish before review'
with gzip.open(manifest,'rt',encoding='utf-8') as f:rows=[json.loads(l) for l in f]
secret=getpass.getpass('Known credential to recheck (process only): ')
policy=Archive.__new__(Archive)
excluded=[];removed_names=set()
for row in rows:
 name=row.get('archived')
 if not name:continue
 path=staging/name
 if name in removed_names:
  row['disposition']='excluded_alias_of_opaque_object';del row['archived'];continue
 reason=policy.classify(name,path.stat().st_size) if path.is_file() else None
 if reason:
  assert path.resolve().is_relative_to(staging.resolve()) and not path.is_symlink()
  path.unlink()  # Generated archive copy only, never the source named by row['source'].
  removed_names.add(name)
  excluded.append({'archived':name,'reason':reason})
  row['disposition']='excluded_at_final_review:'+reason
  del row['archived']

# Include this turn's final evidence created after the initial directory walk.
late=Archive(staging.parent/'late-reviewed',secret)
files=[root/'AGENTS.md',root/'PROJECT_CONTEXT.md',root/'HANDOFF.md']
files += list((root/'docs/server-admin').glob('*20260906*'))
files += list((root/'docs/rlinf-robotwin-pi0-online-bc/evidence').glob('*20260906*'))
files += list((root/'local_scripts').glob('*20260906.py'))
files += list((root/'local_scripts/remote_commands').glob('*20260906*'))
for file in files:
 if file.is_file():late.file(file,'local-workspace/'+file.relative_to(root).as_posix())
late.finish({'scope':'Late files from this task; override older archived copies explicitly'})
overrides={r['archived'] for r in late.rows if r.get('archived')}
rows=[r for r in rows if r.get('archived') not in overrides]+late.rows
for file in late.dest.rglob('*'):
 if not file.is_file() or file.name in {'archive-manifest.jsonl.gz','archive-summary.json'}:continue
 target=staging/file.relative_to(late.dest)
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(file.read_bytes())

with gzip.open(manifest,'wt',encoding='utf-8') as f:
 for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
current=[]; residual=[]
for file in staging.rglob('*'):
 if not file.is_file():continue
 data=file.read_bytes()
 if file.suffix=='.gz':
  try:check=gzip.decompress(data)
  except OSError:check=data
 else:check=data
 if secret.encode() in check:residual.append(str(file.relative_to(staging)))
 assert len(data)<50*1024**2
 current.append((file,len(data)))
assert not residual, 'Residual credential in generated copies: '+repr(residual)
summary={'time':datetime.datetime.now().astimezone().isoformat(),'files':len(current),'bytes':sum(s for _,s in current),'removed_generated_opaque_objects':len(excluded),'credential_residual_files':len(residual)}
(staging/'final-review.json').write_text(json.dumps(summary,indent=2))
with tarfile.open(packet,'w:gz',compresslevel=3) as tar:
 for file,size in current:tar.add(file,arcname=file.relative_to(staging).as_posix(),recursive=False)
 tar.add(staging/'final-review.json',arcname='final-review.json')
summary['packet_bytes']=packet.stat().st_size
print('LOCAL_REVIEW='+json.dumps(summary),flush=True)
