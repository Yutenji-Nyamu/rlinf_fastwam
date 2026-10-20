import concurrent.futures,shutil,subprocess,json,os,tarfile
from pathlib import Path
R=Path('/data/chenyiteng/exports/noise4-gallery-20261020')
ffmpeg=shutil.which('ffmpeg')
assert ffmpeg,'Use the existing encoder; no new package install'
items=json.loads((R/'manifest.json').read_text())['selected']
def convert(row):
 d=R/row['id'];dest=d/'episode-h264.mp4'
 if not dest.is_file():
  subprocess.run([ffmpeg,'-v','error','-nostdin','-i',str(d/'episode.mp4'),
   '-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
   '-threads','1','-movflags','+faststart',str(dest)],check=True,
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=120)
 return dest
with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:
 files=list(pool.map(convert,items))
out=R.with_name('noise4-h264-20261020.tar.gz')
with tarfile.open(out,'w:gz')as t:
 for f in files:t.add(f,arcname=str(f.relative_to(R)))
print(json.dumps({'videos':len(files),'encoder':ffmpeg,'archive':str(out),'bytes':out.stat().st_size}))
