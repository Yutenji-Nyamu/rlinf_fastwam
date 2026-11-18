"""Make frame-aligned plots from offline predictions. Input PNGs are the recorded head view."""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def describe_peak(values, times):
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < 3:
        return None
    maximum = int(valid[np.argmax(values[valid])])
    interior = valid[1:-1]
    # An endpoint maximum has no two-sided illustration. Preserve it in metadata.
    peaks = [i for i in interior if np.isfinite(values[i-1:i+2]).all()
             and values[i] >= values[i-1] and values[i] >= values[i+1]]
    peak = maximum if maximum in interior else int(max(peaks or interior, key=values.__getitem__))
    before = valid[valid < peak]
    after = valid[valid > peak]
    pre = int(before[np.argmin(abs(times[before] - (times[peak] - 1.0)))])
    post = int(after[np.argmin(abs(times[after] - (times[peak] + 1.0)))])
    others = [i for i in peaks if abs(times[i]-times[peak]) >= 1.0]
    other = int(max(others,key=values.__getitem__)) if others else None
    separation = (float(values[peak]-values[other]) if other is not None else None)
    return dict(global_step=maximum+1, plotted_step=int(peak)+1,
                frame_steps=[pre+1,int(peak)+1,post+1],
                other_peak_step=other+1 if other is not None else None, peak_gap=separation)


def make_video(framepaths, output, size, rate):
    # Every sync row appears once in this preview. CSV times preserve capture timing.
    tmp = output.with_name('recording-writing.mp4')
    cmd = ['ffmpeg','-v','error','-y','-threads','2','-f','rawvideo','-pix_fmt','rgb24',
           '-s',f'{size[0]}x{size[1]}','-r',str(rate),'-i','pipe:0','-an','-c:v','libx264',
           '-threads','2','-crf','20','-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as enc:
        try:
            for frame in framepaths:
                with Image.open(frame) as im:
                    enc.stdin.write(im.convert('RGB').tobytes())
        finally:
            enc.stdin.close()
        if enc.wait():
            raise RuntimeError('ffmpeg encoding failed')
    frames = subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                                     '-show_entries','stream=nb_frames','-of','csv=p=0',str(tmp)],text=True)
    if int(frames.strip()) != len(framepaths):
        raise RuntimeError('Video frame count differs from sync.csv')
    tmp.replace(output)


def render(folder, row, sync, framepaths, timestamps, entropy, valid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    times = timestamps-timestamps[0]
    n = len(sync)
    with (folder/'curve.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['action_step','video_frame','time_s','entropy','depth_valid','video_index_0based','image_name','controller_time_s'])
        for i,(rec,h,ok) in enumerate(zip(sync,entropy,valid)):
            writer.writerow([i+1,rec['frame_idx'],round(float(times[i]),6),float(h) if ok else '',
                             int(ok),i,rec['head_image'],rec['controller_time_s']])
    with Image.open(framepaths[0]) as im:
        size = im.size
    rate = round(1/float(np.median(np.diff(timestamps))),3)
    make_video(framepaths,folder/'recording.mp4',size,rate)
    peak = describe_peak(entropy,times)
    fig = plt.figure(figsize=(9.2,4.4),layout='constrained')
    gs = fig.add_gridspec(2,3,height_ratios=[1.3,1])
    name = row['task'].replace('_',' ').title()
    if peak is not None:
        for j,(label,step) in enumerate(zip(['Before','Peak','After'],peak['frame_steps'])):
            i=step-1
            with Image.open(framepaths[i]) as src:
                im=src.convert('RGB')
            im.save(folder/(label.lower()+'-original.png'))
            ax=fig.add_subplot(gs[0,j])
            ax.imshow(im);ax.axis('off');ax.set_title(f'{label} · Step {step}',fontsize=11)
            draw=ImageDraw.Draw(im);caption=f'{label}  |  Step {step}  |  Frame {sync[i]["frame_idx"]}  |  {times[i]:.1f}s'
            draw.rectangle((0,0,im.width,25),fill='#19252c')
            draw.text((8,5),caption,fill='white')
            im.save(folder/(label.lower()+'.png'))
    ax=fig.add_subplot(gs[1,:] if peak else gs[:,:])
    ax.plot(np.arange(1,n+1),entropy,lw=1.25,color='#1c638c')
    if peak:
        for step in peak['frame_steps']:
            ax.plot(step,entropy[step-1],'o',color='#c85227',ms=4)
    ax.set(xlim=(1,n),xlabel='Recorded action step',ylabel='Action entropy (nats)')
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'{row["id"]} · {name}',fontsize=12)
    for ext in ['png','pdf']:
        fig.savefig(folder/f'figure.{ext}',dpi=165)
    plt.close(fig)
    return dict(id=row['id'],episode=row['key'],task=row['task'],split=row['split'],
                observations=n,valid=int(valid.sum()),video_fps=rate,peak=peak,
                missing_steps=(np.flatnonzero(~valid)+1).tolist())
