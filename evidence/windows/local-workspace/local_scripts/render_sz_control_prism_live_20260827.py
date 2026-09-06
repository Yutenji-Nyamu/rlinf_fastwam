from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\86136\Documents\rl")
SNAPSHOT = (
    ROOT
    / "docs"
    / "rlinf-shenzhen-prism-dvac-grpo"
    / "evidence"
    / "control-prism-live-20260827"
)
SERIES = SNAPSHOT / "series.json"
OUTPUT = SNAPSHOT / "01_control_vs_prism_live_success.png"

INK = "#142033"
GRAY = "#64748B"
GRID = "#D8E0EA"
WHITE = "#FFFFFF"
BG = "#F4F7FB"
CONTROL = "#D97706"
CONTROL_LIGHT = "#FBBF24"
PRISM = "#0F766E"
PRISM_LIGHT = "#5EEAD4"


def font(size: int, bold: bool = False):
    for path in (
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(40, True)
F_SUB = font(22)
F_PANEL = font(27, True)
F_AXIS = font(18)
F_LEGEND = font(19, True)
F_NOTE = font(20)


def trailing(rows: list[dict], window: int) -> list[dict]:
    out = []
    for idx, row in enumerate(rows):
        if idx + 1 < window:
            continue
        out.append(
            {
                "step": row["step"],
                "value": statistics.mean(x["value"] for x in rows[idx - window + 1 : idx + 1]),
            }
        )
    return out


def panel(draw: ImageDraw.ImageDraw, box, title, x_max, y_min, y_max):
    left, top, right, bottom = box
    draw.rounded_rectangle((left, top, right, bottom), radius=22, fill=WHITE, outline=GRID, width=2)
    draw.text((left + 25, top + 18), title, fill=INK, font=F_PANEL)
    plot = (left + 85, top + 72, right - 28, bottom - 55)
    pl, pt, pr, pb = plot
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pb - frac * (pb - pt)
        value = y_min + frac * (y_max - y_min)
        draw.line((pl, y, pr, y), fill=GRID, width=1)
        draw.text((left + 18, y - 11), f"{value:.2f}", fill=GRAY, font=F_AXIS)
    for step in range(0, x_max + 1, 10):
        x = pl + step / max(1, x_max) * (pr - pl)
        draw.line((x, pt, x, pb), fill=GRID, width=1)
        draw.text((x - 10, pb + 12), str(step), fill=GRAY, font=F_AXIS)
    draw.line((pl, pt, pl, pb), fill=INK, width=2)
    draw.line((pl, pb, pr, pb), fill=INK, width=2)
    return plot


def xy(plot, step, value, x_max, y_min, y_max):
    left, top, right, bottom = plot
    x = left + step / max(1, x_max) * (right - left)
    y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
    return x, y


def draw_series(draw, plot, rows, color, x_max, y_min, y_max, width=4, dots=False):
    points = [xy(plot, row["step"], row["value"], x_max, y_min, y_max) for row in rows]
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")
    if dots:
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def draw_dashed_series(draw, plot, rows, color, x_max, y_min, y_max, width=3):
    points = [xy(plot, row["step"], row["value"], x_max, y_min, y_max) for row in rows]
    for idx, (a, b) in enumerate(zip(points, points[1:])):
        if idx % 2 == 0:
            draw.line((a, b), fill=color, width=width)


def legend(draw, x, y, color, label, shape="line"):
    if shape == "square":
        draw.rectangle((x, y + 4, x + 15, y + 19), fill=color)
    else:
        draw.line((x, y + 11, x + 28, y + 11), fill=color, width=5)
    draw.text((x + 38, y), label, fill=color, font=F_LEGEND)


def main():
    data = json.loads(SERIES.read_text(encoding="utf-8"))
    control = data["control"]["train_success"]
    prism = data["prism"]["train_success"]
    x_max = max(60, control[-1]["step"], prism[-1]["step"])
    snapshot = datetime.fromtimestamp(max(control[-1]["wall_time"], prism[-1]["wall_time"]))

    image = Image.new("RGB", (1580, 1420), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 30), "Live Shenzhen GRPO: Control vs Prism-DVAC", fill=INK, font=F_TITLE)
    draw.text(
        (58, 86),
        f"Snapshot through Control g{control[-1]['step']} / Prism g{prism[-1]['step']} | {snapshot:%Y-%m-%d %H:%M}",
        fill=GRAY,
        font=F_SUB,
    )
    legend(draw, 760, 38, CONTROL, "Control")
    legend(draw, 1000, 38, PRISM, "Prism-DVAC")

    raw_plot = panel(draw, (40, 135, 1540, 515), "Per-step train rollout success", x_max, 0.60, 1.00)
    draw_series(draw, raw_plot, control, CONTROL, x_max, 0.60, 1.00, dots=True)
    draw_series(draw, raw_plot, prism, PRISM, x_max, 0.60, 1.00, dots=True)

    mean_plot = panel(draw, (40, 540, 1540, 925), "Moving means: solid = 5-step, dashed = 10-step", x_max, 0.65, 1.00)
    draw_series(draw, mean_plot, trailing(control, 5), CONTROL, x_max, 0.65, 1.00, width=5)
    draw_dashed_series(draw, mean_plot, trailing(control, 10), CONTROL_LIGHT, x_max, 0.65, 1.00, width=4)
    draw_series(draw, mean_plot, trailing(prism, 5), PRISM, x_max, 0.65, 1.00, width=5)
    draw_dashed_series(draw, mean_plot, trailing(prism, 10), PRISM_LIGHT, x_max, 0.65, 1.00, width=4)

    eval_plot = panel(draw, (40, 950, 1540, 1300), "Fixed-32 evaluation", x_max, 0.75, 1.00)
    for label, color in (("control", CONTROL), ("prism", PRISM)):
        for row in data[label]["eval_success"]:
            x, y = xy(eval_plot, row["step"], row["value"], x_max, 0.75, 1.00)
            draw.rectangle((x - 7, y - 7, x + 7, y + 7), fill=color)
            draw.text((x - 18, y - 34), f"{row['value']*32:.0f}/32", fill=color, font=F_AXIS)

    control5 = statistics.mean(x["value"] for x in control[-5:])
    control10 = statistics.mean(x["value"] for x in control[-10:])
    prism5 = statistics.mean(x["value"] for x in prism[-5:])
    draw.text(
        (60, 1330),
        f"Latest: Control {control[-1]['value']:.1%} (5-step {control5:.1%}, 10-step {control10:.1%}); "
        f"Prism {prism[-1]['value']:.1%} (5-step {prism5:.1%}). Prism is still only {prism[-1]['step']} completed steps.",
        fill=INK,
        font=F_NOTE,
    )
    image.save(OUTPUT, quality=95)
    print(OUTPUT)


if __name__ == "__main__":
    main()
