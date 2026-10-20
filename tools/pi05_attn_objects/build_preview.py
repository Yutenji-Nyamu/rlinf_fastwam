"""Use the accepted full-curve/peak-frame gallery layout for the new channels."""
import html
import json
import os
import sys
from pathlib import Path

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path, help="Extracted export directory containing manifest.json")
parser.add_argument("--plot-deps", type=Path, help="Optional installed plotting package directory")
parser.add_argument("--font", type=Path, help="Optional TrueType font for the contact sheet")
args = parser.parse_args()
if args.plot_deps:
    sys.path.insert(0, str(args.plot_deps))
O = args.directory.resolve()
os.environ.setdefault("MPLCONFIGDIR", str(O / ".mpl-cache"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

m = json.loads((O / "manifest.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
cards = []
contact = Image.new("RGB", (1320, 340 * len(m["selected"])), "white")
draw = ImageDraw.Draw(contact)
font = ImageFont.truetype(str(args.font), 18) if args.font else ImageFont.load_default()
for row, c in enumerate(m["selected"]):
    d = O / c["id"]
    x = np.asarray(c["action"])
    selected = [c["options"][i] for i in c["selected_indices"]]
    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.5, 1, 1, 1]})
    for key, label, color, style in (
        ("i_direct", "Direct object words", "#27649b", "-"),
        ("i_via_visual", "Via visual overlap", "#23846f", "-"),
        ("attention_total", "Direct + via", "#b55528", "--"),
    ):
        axes[0].plot(x, c[key], label=label, color=color, lw=1.25, ls=style)
    axes[0].set_ylabel("Attention mass")
    axes[0].legend(frameon=False, loc="upper right", ncol=3, fontsize=9)
    for ax, key, label, color in zip(axes[1:], ("u", "noise4", "dv"),
                                    ("U (10 vs 5)", "Noise4 consistency", "DV"),
                                    ("#365f99", "#8d5ca6", "#6e6b45")):
        ax.plot(x, c[key], color=color, lw=1.2)
        ax.set_ylabel(label)
    for ax in axes:
        ax.set_xlim(x[0], x[-1])
        ax.grid(alpha=.15)
        for letter, p in zip("ABC", selected):
            ax.axvline(p["action"], color="#999", lw=.6, ls="--")
            if ax is axes[0]:
                ax.annotate(letter, (p["action"], c["attention_total"][p["index"]]),
                            xytext=(3, 6), textcoords="offset points")
    axes[-1].set_xlabel("Executed action index (controller-projected prefix)")
    fig.suptitle(f"{c['id']} · {c['task']} · seed {c['actual']} · " +
                 ("SUCCESS" if c["success"] else "FAILURE"), fontsize=12)
    fig.savefig(d / "curve.svg")
    fig.savefig(d / "curve.png", dpi=110)
    plt.close(fig)
    pictures = ""
    for col, (letter, p) in enumerate(zip("ABC", selected)):
        pictures += (f'<button class="frame" onclick="seek(this,{p["frame"]/c["fps"]:.6f})">'
                     f'<img src="{c["id"]}/{p["image"]}"><span>{letter} · 动作 {p["action"]} · 帧 {p["frame"]}</span></button>')
        im = Image.open(d / p["image"]).convert("RGB")
        im.thumbnail((430, 270))
        contact.paste(im, (440 * col, 340 * row + 30))
        draw.text((440 * col + 4, 340 * row + 307),
                  f"{letter}: action {p['action']}, frame {p['frame']}, map error {p['h_error']:.1f}", font=font, fill="#345")
    draw.text((8, 340 * row + 4), f"{c['id']} {c['task']} | seed {c['actual']}", font=font, fill="#234")
    cards.append(f'''<article><h2>{c["id"]} · {html.escape(c["task"])}</h2><p>{"成功" if c["success"] else "失败"} · seed {c["actual"]} · {len(x)}个已执行动作点</p>
<img class="curve" src="{c["id"]}/curve.svg"><div class="frames">{pictures}</div>
<details><summary>完整视频与原始曲线</summary><video controls preload="metadata" src="{c["id"]}/episode-h264.mp4"></video>
<p>帧来自同一轨迹的总Attn峰附近；h到TOPP帧沿用原流程的几何投影，误差保留。</p>
<a href="{c["id"]}/series.json">逐动作六信号</a> · <a href="{c["id"]}/frame_map.json">帧映射</a> · <a href="{c["id"]}/curve.svg">SVG</a></details></article>''')
contact.save(O / "contact.jpg", quality=94)
page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Attn · 首批信号预览</title><style>body{font:16px/1.65 Arial,"Microsoft Yahei",sans-serif;background:#f3f5f7;color:#26384a;margin:0}
main{max-width:1250px;margin:auto;padding:24px}header,article{background:white;padding:24px;border-radius:12px;margin:18px 0}
h1{font-size:28px}h2{font-size:21px}.curve{width:100%}.frames{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.frame{border:0;background:#eef2f6;padding:0;cursor:pointer}.frame img{width:100%}.frame span{display:block;padding:8px}video{max-width:100%;width:780px}
a{color:#27649b}summary{cursor:pointer}p{color:#526171}</style><main><header><h1>两路 Attn 与总和 · 首批预览</h1>
<p>蓝色：动作直接读对象词；绿色：经视觉位置与对象词关联；橙色虚线：两路相加。下面分别是原U、四噪声一致性、DV。</p>
<p>取末次去噪、全部已执行动作；未平滑、未按块求均值。对象词由明确规则提取，真实词跨度留在数据中，可后续修正。总分不是成功概率。</p>
<p>这3条精选轨迹的总Attn最大值均在动作49（首块尾端），需要进一步区分动作位置效应，不能直接当作任务关键性证据。</p>
<p>这里只展示首批两个任务的绘图候选，成功优先；全50任务×32回合继续采集。不能用这些精选样例判断信号与成功的统计关系。</p></header>'''
page += "".join(cards)
page += '''</main><script>function seek(b,t){let d=b.closest('article').querySelector('details'),v=d.querySelector('video');d.open=true;if(v.readyState>=1)v.currentTime=t;else v.addEventListener('loadedmetadata',()=>v.currentTime=t,{once:true});v.scrollIntoView({block:'nearest',behavior:'smooth'})}</script></html>'''
(O / "index.html").write_text(page, encoding="utf-8")
print(json.dumps(dict(page=str(O / "index.html"), selected=len(m["selected"]), errors=m["errors"])))
