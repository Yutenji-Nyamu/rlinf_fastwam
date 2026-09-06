from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw/run/dvac_train"
OUT = HERE / "analysis"
FONT_REGULAR = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_SEMIBOLD = Path(r"C:\Windows\Fonts\seguisb.ttf")


def font(size: int, semibold: bool = False):
    return ImageFont.truetype(str(FONT_SEMIBOLD if semibold else FONT_REGULAR), size)


def load_final() -> tuple[np.ndarray, np.ndarray]:
    residuals: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    for path in sorted(RAW.glob("actor_rank*/rollout_step0050.npz")):
        with np.load(path) as data:
            mask = data["loss_mask"].reshape(-1).astype(bool)
            residuals.append(data["residual_z"].reshape(-1, 50)[mask].astype(np.float64))
            advantages.append(data["advantages"].reshape(-1)[mask].astype(np.float64))
    if len(residuals) != 2:
        raise RuntimeError(f"expected two final rank NPZ files, found {len(residuals)}")
    return np.concatenate(residuals), np.concatenate(advantages)


def continuous(residual: np.ndarray, low: float, high: float) -> np.ndarray:
    r = np.clip(residual, -2.0, 2.0)
    return 1.0 + (1.0 - low) / 2.0 * np.minimum(r, 0.0) + (high - 1.0) / 2.0 * np.maximum(r, 0.0)


def top20(residual: np.ndarray, selected: float, other: float) -> np.ndarray:
    result = np.full_like(residual, other)
    indices = np.argpartition(residual, -10, axis=1)[:, -10:]
    np.put_along_axis(result, indices, selected, axis=1)
    return result


def summarize(name: str, weights: np.ndarray, advantages: np.ndarray) -> dict[str, float | str]:
    ess = np.square(weights.sum(axis=1)) / (50.0 * np.square(weights).sum(axis=1))
    angle = np.degrees(np.arccos(np.clip(np.sqrt(ess), 0.0, 1.0)))
    sorted_weights = np.sort(weights, axis=1)[:, ::-1]
    top20 = sorted_weights[:, :10].sum(axis=1) / sorted_weights.sum(axis=1)
    base = np.repeat(advantages[:, None], 50, axis=1).reshape(-1)
    changed = (advantages[:, None] * weights).reshape(-1)
    cosine = float(np.dot(base, changed) / (np.linalg.norm(base) * np.linalg.norm(changed)))
    return {
        "mapping": name,
        "p05": float(np.quantile(weights, 0.05)),
        "median": float(np.quantile(weights, 0.50)),
        "p95": float(np.quantile(weights, 0.95)),
        "mean": float(weights.mean()),
        "zero_fraction": float(np.mean(weights == 0.0)),
        "two_fraction": float(np.mean(weights == 2.0)),
        "weight_ess_ratio": float(ess.mean()),
        "effective_h": float(50.0 * ess.mean()),
        "top20_weight_mass": float(top20.mean()),
        "coefficient_angle_deg": float(angle.mean()),
        "advantage_weighted_angle_deg": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
    }


def draw_bar(draw, x0: int, x1: int, y: int, value: float, maximum: float, color: str) -> None:
    draw.rounded_rectangle((x0, y, x1, y + 24), radius=12, fill="#E8EDF4")
    width = int((x1 - x0) * min(max(value / maximum, 0.0), 1.0))
    if width > 0:
        draw.rounded_rectangle((x0, y, x0 + width, y + 24), radius=12, fill=color)


def render(rows: list[dict[str, float | str]]) -> None:
    image = Image.new("RGB", (1700, 1030), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    draw.text((65, 45), "Global Step 51: stronger-weight counterfactual", fill="#17233D", font=font(42, True))
    draw.text(
        (67, 105),
        "Same final R-only residuals; only the residual-to-gradient-weight map changes.",
        fill="#5E6C84",
        font=font(22),
    )
    columns = [
        ("Effective actions / 50", "effective_h", 50.0, "#2563EB"),
        ("Top-20% weight mass", "top20_weight_mass", 1.0, "#EA580C"),
        ("Coefficient angle", "coefficient_angle_deg", 70.0, "#7C3AED"),
    ]
    x_starts = [420, 835, 1250]
    for x, (title, _, _, _) in zip(x_starts, columns):
        draw.text((x, 175), title, fill="#17233D", font=font(21, True))
    y = 245
    for row in rows:
        name = str(row["mapping"])
        draw.text((70, y - 5), name, fill="#17233D", font=font(23, True))
        draw.text(
            (70, y + 31),
            f"p05 / med / p95 = {row['p05']:.3f} / {row['median']:.3f} / {row['p95']:.3f}",
            fill="#5E6C84",
            font=font(17),
        )
        for x, (_, key, maximum, color) in zip(x_starts, columns):
            value = float(row[key])
            draw_bar(draw, x, x + 305, y + 5, value, maximum, color)
            label = f"{100*value:.1f}%" if key == "top20_weight_mass" else (f"{value:.1f}°" if "angle" in key else f"{value:.1f}")
            draw.text((x + 313, y + 2), label, fill="#17233D", font=font(18, True))
        y += 145
    draw.rounded_rectangle((65, 885, 1635, 980), radius=18, fill="#F2F6FB")
    draw.text((92, 905), "Reading", fill="#17233D", font=font(21, True))
    draw.text(
        (92, 940),
        "[0,2] is a clear strengthening experiment: much stronger than v2, close to top-20 x2 concentration, and still far from hard 80/20.",
        fill="#34425B",
        font=font(20),
    )
    image.save(OUT / "V3_RANGE_COUNTERFACTUAL_G51.png", optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    residual, advantage = load_final()
    candidates = [
        ("v2 [0.5, 1.2]", continuous(residual, 0.5, 1.2)),
        ("middle [0.25, 1.5]", continuous(residual, 0.25, 1.5)),
        ("v3 candidate [0, 2]", continuous(residual, 0.0, 2.0)),
        ("A3PO-like top-20 x2", top20(residual, 2.0, 1.0)),
        ("hard top-20 {0,5}", top20(residual, 5.0, 0.0)),
    ]
    rows = [summarize(name, weights, advantage) for name, weights in candidates]
    with (OUT / "V3_RANGE_COUNTERFACTUAL_G51.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "V3_RANGE_COUNTERFACTUAL_G51.json").write_text(
        json.dumps({"valid_queries": int(residual.shape[0]), "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    render(rows)
    print(json.dumps({"valid_queries": int(residual.shape[0]), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
