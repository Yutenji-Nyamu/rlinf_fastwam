"""Redraw standalone scientific figures and a local gallery from saved arrays."""

import argparse
import html
import itertools
import json
from pathlib import Path

import numpy as np

from .common import entropy, uniform_indices, write_json


def main(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import spearmanr

    root = Path(args.data)
    out = root / "derived"
    public = out / "public"
    private = out / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    cards = []
    private_cards = []

    def figure(fig, name, category, title, description, private_flag=False):
        directory = private if private_flag else public
        fig.tight_layout()
        fig.savefig(directory / (name + ".svg"), bbox_inches="tight")
        fig.savefig(directory / (name + ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        (private_cards if private_flag else cards).append(
            dict(name=name, category=category, title=title, description=description)
        )

    def preview_frames(video, target):
        import cv2

        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in uniform_indices(count, 4):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, im = cap.read()
            if ok:
                frames.append(cv2.resize(im, (256, 192)))
        cap.release()
        if frames:
            cv2.imwrite(str(target), np.concatenate(frames, axis=1))

    old_index = json.loads((root / "existing/index.json").read_text())
    by_task = {}
    for ep in old_index:
        j = json.loads((root / "existing" / ep["id"] / "series.json").read_text())
        points = j["positions"]
        fig, ax = plt.subplots(figsize=(6.8, 2.3))
        ax.plot(
            [x["action"] for x in points],
            [x["entropy4"] for x in points],
            color="#156b8a",
            lw=1.25,
        )
        ax.set(
            xlabel="Executed model-action index (native projection)",
            ylabel="K4 entropy (nats)",
            title=f"{ep['task']} | {ep['id']} | seed {ep['actual']}",
        )
        ax.grid(alpha=0.15)
        figure(
            fig,
            "old_" + ep["id"],
            "RT2已有",
            ep["task"] + " · " + ep["id"],
            f"{'成功' if ep['success'] else '失败'}，{len(points)}个实际到达的位置；D14，ddof=1，epsilon=1e-6。横轴由原生控制路径投影得到。",
        )
        preview_frames(
            Path(ep["source"]) / ep["video"],
            public / ("old_" + ep["id"] + "_frames.jpg"),
        )
        by_task.setdefault(ep["task"], (ep, points))
    tasks = [
        "grab_roller",
        "hanging_mug",
        "lift_pot",
        "pick_dual_bottles",
        "place_a2b_right",
        "place_cans_plasticbox",
        "place_dual_shoes",
        "place_object_basket",
        "place_object_stand",
        "place_phone_stand",
        "press_stapler",
        "shake_bottle_horizontally",
        "stack_bowls_two",
        "turn_switch",
    ]
    with PdfPages(public / "14_tasks_entropy.pdf") as pdf:
        for start in range(0, len(tasks), 6):
            fig, axes = plt.subplots(3, 2, figsize=(9, 8.2))
            for ax, task in zip(axes.flat, tasks[start : start + 6]):
                ep, points = by_task[task]
                ax.plot(
                    [x["action"] for x in points],
                    [x["entropy4"] for x in points],
                    color="#156b8a",
                    lw=1,
                )
                ax.set(
                    title=task.replace("_", " "),
                    xlabel="Projected executed action",
                    ylabel="Entropy (nats)",
                )
                ax.grid(alpha=0.15)
            for ax in list(axes.flat)[len(tasks[start : start + 6]) :]:
                ax.axis("off")
            fig.suptitle(
                "Fixed-observation K4 disagreement | normalized D14 | existing trajectories",
                fontsize=10,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(public / f"14_tasks_page_{start // 6 + 1}.png", dpi=150)
            plt.close(fig)
    kroot = root / "rt-k8"
    if (kroot / "complete.json").exists():
        with np.load(kroot / "bank_0.npz", allow_pickle=False) as z:
            a = z["model_actions"]
        with np.load(kroot / "bank_1.npz", allow_pickle=False) as z:
            b = z["model_actions"]
        reference = entropy(b).mean(1)
        metrics = []
        for k in [2, 3, 4, 8]:
            rows = []
            for ids in itertools.combinations(range(8), k):
                estimate = entropy(a[:, ids])
                means = estimate.mean(1)
                rows.append(
                    dict(
                        subset=list(ids),
                        rank_rho=float(spearmanr(means, reference).statistic),
                        raw_rmse=float(np.sqrt(np.mean((estimate - entropy(b)) ** 2))),
                        centered_rmse=float(
                            np.sqrt(
                                np.mean(
                                    (
                                        (estimate - estimate.mean())
                                        - (entropy(b) - entropy(b).mean())
                                    )
                                    ** 2
                                )
                            )
                        ),
                    )
                )
            metrics.append(
                dict(
                    K=k,
                    subsets=len(rows),
                    rho_median=float(np.median([r["rank_rho"] for r in rows])),
                    rho_p10=float(np.quantile([r["rank_rho"] for r in rows], 0.1)),
                    rho_p90=float(np.quantile([r["rank_rho"] for r in rows], 0.9)),
                    rows=rows,
                )
            )
        write_json(public / "k-stability.json", metrics)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].boxplot(
            [[r["rank_rho"] for r in m["rows"]] for m in metrics],
            tick_labels=[str(m["K"]) for m in metrics],
            showfliers=False,
        )
        axes[0].set(
            xlabel="K from bank A",
            ylabel="Rank correlation with independent K8 bank B",
            ylim=(-0.05, 1.05),
        )
        axes[0].grid(alpha=0.15)
        for key, label in [
            ("raw_rmse", "Raw entropy RMSE"),
            ("centered_rmse", "Bias-centered shape RMSE"),
        ]:
            axes[1].plot(
                [m["K"] for m in metrics],
                [np.median([r[key] for r in m["rows"]]) for m in metrics],
                marker="o",
                label=label,
            )
        axes[1].set(xlabel="K", ylabel="Nats")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.15)
        figure(
            fig,
            "k_stability",
            "K与成本",
            "K稳定性：独立采样库比较",
            "64个固定观测、两个任务；K2/3/4枚举第一库子集，统一与独立第二库K8比较。K8只有一次独立重复，不能视为稳定真值。",
        )
        costs = json.loads((kroot / "costs.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for kind in sorted({r["kind"] for r in costs}):
            ks = sorted({r["K"] for r in costs if r["kind"] == kind})
            axes[0].plot(
                ks,
                [
                    np.median(
                        [
                            r["seconds"]
                            for r in costs
                            if r["kind"] == kind and r["K"] == k
                        ]
                    )
                    for k in ks
                ],
                marker="o",
                label=kind,
            )
        axes[0].set(xlabel="K", ylabel="Synchronized wall time (s)")
        axes[0].legend(fontsize=6)
        axes[0].grid(alpha=0.15)
        ks = [1, 2, 4, 8]
        axes[1].plot(
            ks,
            [max(r["allocated_peak_gib"] for r in costs if r["K"] == k) for k in ks],
            marker="o",
        )
        axes[1].set(xlabel="K", ylabel="Peak allocated GPU memory (GiB)")
        axes[1].grid(alpha=0.15)
        figure(
            fig,
            "k_cost",
            "K与成本",
            "K的实际推理成本",
            "同一卡、同一观测、NFE10；每项预热后3次。分别测合并推理和保留主执行再补额外样本。只测推理，不冒充总训练耗时。",
        )
    for taskdir in (
        sorted((root / "dojo").glob("*")) if (root / "dojo").exists() else []
    ):
        if not (taskdir / "complete.json").exists():
            continue
        queries = json.loads((taskdir / "queries.json").read_text())
        eps = json.loads((taskdir / "episodes.json").read_text())
        for ep in eps:
            slot = ep["slot"]
            x = []
            y = []
            for q in queries:
                with np.load(taskdir / q["path"], allow_pickle=False) as z:
                    n = int(z["after"][slot] - z["before"][slot])
                    if z["active"][slot] and n > 0:
                        x.append(int(z["before"][slot]))
                        y.append(float(z["entropy"][slot, :n].mean()))
            if not x:
                continue
            fig, ax = plt.subplots(figsize=(6.8, 2.4))
            ax.plot(x, y, "o-", ms=3, color="#dc7b28")
            ax.set(
                xlabel="Native action step at observation",
                ylabel="Executed-prefix K4 entropy (nats)",
                title=f"{ep['task']} | layout {ep['layout_id']}",
            )
            ax.grid(alpha=0.15)
            figure(
                fig,
                f"dojo_{ep['task']}_{slot:02d}",
                "Dojo采集",
                f"{ep['task']} · {slot}",
                f"冻结原VLA，{'成功' if ep['success'] else '失败'}；{len(x)}次实测观测点。只统计随后实际执行的前缀；连线仅为阅读引导。",
            )
    gro = root / "grbc"
    if (gro / "features-complete.json").exists():
        triples = json.loads((gro / "triplets.json").read_text())
        nodes = json.loads((gro / "nodes.json").read_text())
        if triples:
            fig, ax = plt.subplots(figsize=(4.7, 4))
            ax.scatter(
                [t["negative_mean"] for t in triples],
                [t["positive_mean"] for t in triples],
                s=18,
                alpha=0.65,
            )
            maximum = (
                max(max(t["negative_mean"], t["positive_mean"]) for t in triples) * 1.05
            )
            ax.plot([0, maximum], [0, maximum], "--", lw=1, c="gray")
            ax.set(
                xlabel="Different time / same light: MCR distance",
                ylabel="Same physical state / changed light: MCR distance",
            )
            figure(
                fig,
                "grbc_distances",
                "GR-BC机制",
                "外观扰动与时间差：原始高维距离",
                "每点是夹爪状态匹配的三元组；对角线上方表示视觉扰动距离更大。不同时间不自动等于不同语义阶段；尚需逐帧复核。",
            )
            selected = sorted(
                triples,
                key=lambda x: x["positive_mean"] - x["negative_mean"],
                reverse=True,
            )[:6]
            write_json(public / "grbc-review-candidates.json", selected)
            for rank, t in enumerate(selected):
                fig, axes = plt.subplots(1, 3, figsize=(9, 2.6))
                for ax, key, label in zip(
                    axes,
                    ["anchor", "same_state_other_light", "other_time_same_light"],
                    [
                        "A: anchor",
                        "B: same state, changed light",
                        "C: other time, same light",
                    ],
                ):
                    node = nodes[t[key]]
                    with np.load(node["raw"], allow_pickle=False) as z:
                        rgb = z["images"][node["appearance"], 0]
                    ax.imshow(rgb)
                    ax.axis("off")
                    ax.set_title(f"{label}\nquery {node['query']}", fontsize=8)
                figure(
                    fig,
                    f"grbc_triplet_{rank:02d}",
                    "GR-BC机制",
                    f"GR-BC候选 {rank + 1}",
                    f"三视角平均距离：AB={t['positive_mean']:.3f}，AC={t['negative_mean']:.3f}；AB实际连边保留={t['paired_edge_retained']}。这是固定32对里的诊断排序。",
                )
    rr = root / "rynn"
    if (rr / "index.json").exists():
        for ep in json.loads((rr / "index.json").read_text()):
            with np.load(rr / ep["id"] / "values.npz", allow_pickle=False) as z:
                x = z["anchor_coordinate"]
                v = z["remaining_seconds"]
            fig, ax = plt.subplots(figsize=(6.8, 2.5))
            ax.plot(x, v, "o-", c="#7956b8", ms=3)
            ax.grid(alpha=0.15)
            ax.set(
                xlabel=ep["coordinate_unit"],
                ylabel="Predicted remaining time (s)",
                title=ep["task"]
                + " | "
                + (
                    "mismatched instruction"
                    if ep.get("counterfactual_of")
                    else ("success" if ep["success"] else "failure")
                ),
            )
            figure(
                fig,
                "rynn_" + ep["id"],
                "RynnValue",
                ep["id"],
                "仅输入当前及以前8帧，读取最后value槽；因果SDPA。输出是剩余秒数，不能当成功概率。回退/恢复须结合画面判断，非单调本身不算错误。",
            )
    gg = root / "geowam"
    if gg.exists():
        groups = {}
        for receipt in sorted(gg.glob("*/complete.json")):
            row = json.loads(receipt.read_text())
            groups.setdefault(row["episode"], []).append((receipt.parent, row))
        for episode, rows in groups.items():
            rows.sort(key=lambda r: r[1]["observation_time_s"])
            x = []
            means = []
            heat = []
            for directory, row in rows:
                with np.load(directory / "actions.npz", allow_pickle=False) as z:
                    h = z["entropy"]
                x.append(row["observation_time_s"])
                means.append(float(h.mean()))
                heat.append(h)
            fig, axes = plt.subplots(
                2, 1, figsize=(7, 4.5), gridspec_kw={"height_ratios": [1, 1.3]}
            )
            axes[0].plot(x, means, "o-", c="#17866a", ms=4)
            axes[0].set(
                ylabel="Chunk-mean entropy (nats)",
                title=rows[0][1]["task"] + " | " + episode,
            )
            axes[0].grid(alpha=0.15)
            xx = np.asarray(x)
            edges = (
                np.r_[
                    max(0, xx[0] - (xx[1] - xx[0]) / 2),
                    (xx[:-1] + xx[1:]) / 2,
                    xx[-1] + (xx[-1] - xx[-2]) / 2,
                ]
                if len(xx) > 1
                else np.asarray([max(0, xx[0] - 0.05), xx[0] + 0.05])
            )
            im = axes[1].pcolormesh(
                edges,
                np.linspace(0.05, 3.25, 33),
                np.stack(heat).T,
                cmap="viridis",
                shading="flat",
            )
            axes[1].set(
                xlabel="Recorded observation time (s)",
                ylabel="Predicted future offset (s)",
            )
            fig.colorbar(im, ax=axes[1], label="Entropy (nats)")
            figure(
                fig,
                "geo_" + episode,
                "真机离线",
                rows[0][1]["task"] + " · " + episode,
                f"{rows[0][1]['split']}；{len(rows)}个录制观测，K4/H32/D7。预测没有实际执行；热图按真实观测时刻定位，单元格边界取相邻观测中点。",
                True,
            )
    css = "body{font:15px system-ui;margin:32px auto;max-width:1180px;background:#f4f6f8;color:#243342}h1{font-size:27px}nav{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;background:#f4f6f8;padding:12px 0}button,input{padding:9px 13px;border:1px solid #d7dfe7;border-radius:7px;background:white}article{background:white;border:1px solid #e1e6eb;border-radius:12px;padding:18px;margin:15px 0}article img{max-width:100%;display:block}h2{font-size:17px}p{line-height:1.65;color:#526273}.label{color:#176e88;font-size:12px}a{color:#176e88}.frames{margin-top:8px;width:100%}"
    for folder, items, title in [
        (public, cards, "附录信号与机制图"),
        (private, private_cards, "GeoWAM 真机离线图 · 私有"),
    ]:
        body = []
        for c in items:
            thumb = folder / (c["name"] + "_frames.jpg")
            frames = (
                f'<img class="frames" loading="lazy" src="{thumb.name}">'
                if thumb.exists()
                else ""
            )
            body.append(
                f'<article data-cat="{html.escape(c["category"])}"><span class="label">{html.escape(c["category"])}</span><h2>{html.escape(c["title"])}</h2><img loading="lazy" src="{c["name"]}.svg">{frames}<p>{html.escape(c["description"])}</p><a href="{c["name"]}.svg">SVG</a> · <a href="{c["name"]}.png">PNG</a></article>'
            )
        buttons = "".join(
            "<button onclick=\"filter('" + cat + "')\">" + cat + "</button>"
            for cat in ["全部"] + sorted({c["category"] for c in items})
        )
        page = (
            '<!doctype html><html lang="zh"><meta charset="utf-8"><title>'
            + title
            + "</title><style>"
            + css
            + "</style><h1>"
            + title
            + "</h1><p>每张图来自保存的原始数组。实测点、已有记录与推断范围在图下说明；空缺项目等待采集，不补造结果。</p><nav>"
            + buttons
            + '<input id="search" placeholder="搜索任务 / 编号" oninput="filter(active)"></nav>'
            + "".join(body)
            + """<script>let active='全部';function filter(cat){active=cat;let q=document.getElementById('search').value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.hidden=!((cat==='全部'||a.dataset.cat===cat)&&a.textContent.toLowerCase().includes(q)));}</script></html>"""
        )
        (folder / "index.html").write_text(page, encoding="utf8")
    write_json(
        out / "render-receipt.json",
        dict(
            public_figures=len(cards),
            private_figures=len(private_cards),
            source=str(root),
        ),
    )
    print(
        json.dumps(dict(public_figures=len(cards), private_figures=len(private_cards)))
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    main(p.parse_args())
