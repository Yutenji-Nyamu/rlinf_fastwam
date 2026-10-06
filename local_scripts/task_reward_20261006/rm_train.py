"""Small single-GPU task reward training, adapted from the official RLinf recipe.

Defaults: ImageNet ResNet18 / 256 head / BCE, AdamW1e-4, global64/micro32,
negative:positive training sampling2:1. All images may stay preprocessed on
GPU. Profiling changes microbatch only; data amount/global batch stay fixed.
Validation chooses checkpoint and threshold; held-out test is evaluated once.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from rm_inference import PREPROCESS, SingleTaskReward, sha256


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def metrics(labels, scores, episode_uids, threshold):
    # Float64 preserves nextafter thresholds. A float32 array can otherwise
    # round a just-above-score scalar back onto the score during comparison.
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=np.float64)
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn, tn = int(np.sum(~predicted & (labels == 1))), int(np.sum(~predicted & (labels == 0)))
    negative_episodes = {str(uid) for uid, label in zip(episode_uids, labels) if label == 0}
    false_episodes = {str(uid) for uid, label, pred in zip(episode_uids, labels, predicted) if label == 0 and pred}
    return dict(threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp / (tp + fn), false_positive_rate=fp / (fp + tn),
                false_alarm_episodes=len(false_episodes), negative_episodes=len(negative_episodes),
                episode_false_alarm_rate=len(false_episodes) / len(negative_episodes))


def auc(labels, scores):
    positive = np.asarray(scores)[np.asarray(labels) == 1]
    negative = np.asarray(scores)[np.asarray(labels) == 0]
    return float(np.mean((positive[:, None] > negative).astype(float) + .5 * (positive[:, None] == negative)))


def choose_threshold(labels, scores, uids, max_false_alarm):
    # Export float32-representable thresholds: the online WM adapter compares
    # float32 scores, so a float64-only epsilon would round back at deployment.
    score32 = np.asarray(scores, dtype=np.float32)
    above32 = np.nextafter(score32, np.float32(np.inf))
    candidates = np.unique(np.r_[0., score32.astype(np.float64), above32.astype(np.float64), 1.])
    # Include a threshold just above1 in the report only if rounded scores1
    # make "always continue" the only qualifying decision; never hide failure.
    candidates = np.r_[candidates, float(np.nextafter(np.float32(1.), np.float32(np.inf)))]
    rows = [metrics(labels, scores, uids, value) for value in candidates]
    feasible = [row for row in rows if row["episode_false_alarm_rate"] <= max_false_alarm]
    return max(feasible, key=lambda row: (row["recall"], -row["episode_false_alarm_rate"], row["threshold"]))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_split(root, info, model, cache_device):
    path = root / info["path"]
    if sha256(path) != info["sha256"]:
        raise ValueError("Dataset checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        raw = archive["images"]
        labels = archive["labels"].copy()
        uids = archive["episode_uids"].copy()
        steps = archive["action_steps"].copy()
        blocks = [model.preprocess_images(torch.from_numpy(raw[start:start+128])).to(cache_device)
                  for start in range(0, len(raw), 128)]
    if len(labels) != info["count"] or set(labels.tolist()) != {0, 1}:
        raise ValueError("Dataset labels/count mismatch")
    return dict(images=torch.cat(blocks), labels=torch.tensor(labels, dtype=torch.float32, device=cache_device),
                labels_numpy=labels, uids=uids, steps=steps)


@torch.no_grad()
def evaluate(model, data, device, batch_size):
    model.eval()
    logits = torch.cat([model(data["images"][start:start+batch_size].to(device), preprocessed=True).float().cpu()
                        for start in range(0, len(data["labels"]), batch_size)])
    loss = float(F.binary_cross_entropy_with_logits(logits, torch.tensor(data["labels_numpy"], dtype=torch.float32)))
    scores = logits.sigmoid().numpy()
    if not math.isfinite(loss) or not np.isfinite(scores).all():
        raise RuntimeError("Nonfinite evaluation output")
    return loss, scores


def profile(model, images, labels, device, candidates, global_batch):
    """Few real forward/backward calls, restoring BN buffers/RNG afterwards."""
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    rows = []
    source_ids = torch.arange(global_batch, device=images.device) % len(images)
    x, y = images[source_ids], labels[source_ids]
    model.train()
    try:
        for micro in candidates:
            if micro > global_batch or global_batch % micro:
                continue
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                durations = []
                for repeat in range(3):
                    model.zero_grad(set_to_none=True)
                    sync(device)
                    start = time.perf_counter()
                    for offset in range(0, global_batch, micro):
                        loss = F.binary_cross_entropy_with_logits(model(x[offset:offset+micro].to(device), preprocessed=True), y[offset:offset+micro].to(device))
                        (loss * micro / global_batch).backward()
                    sync(device)
                    if repeat:
                        durations.append(time.perf_counter() - start)
                seconds = sum(durations) / len(durations)
                rows.append(dict(micro_batch=micro, global_batch=global_batch, seconds=seconds,
                                 images_per_second=global_batch/seconds,
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None))
            except torch.OutOfMemoryError:
                rows.append(dict(micro_batch=micro, global_batch=global_batch, error="CUDA OOM"))
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
    finally:
        model.zero_grad(set_to_none=True)
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    return rows


@torch.no_grad()
def profile_inference(model, images, device):
    model.eval()
    ids = torch.arange(256, device=images.device) % len(images)
    x = images[ids]
    rows = []
    for batch in (128, 256):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for repeat in range(3):
            sync(device)
            start = time.perf_counter()
            for offset in range(0, len(x), batch):
                model(x[offset:offset+batch].to(device), preprocessed=True)
            sync(device)
            if repeat:
                durations.append(time.perf_counter() - start)
        seconds = sum(durations) / len(durations)
        rows.append(dict(batch=batch, count=len(x), seconds=seconds, images_per_second=len(x)/seconds,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--auto-micro", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--negative-positive-ratio", type=float, default=2.)
    parser.add_argument("--max-val-episode-false-alarm", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.global_batch, args.micro_batch, args.eval_batch) < 1 or args.negative_positive_ratio <= 0:
        raise ValueError("Positive epochs/batch sizes/ratio required")
    if not 0 <= args.max_val_episode_false_alarm <= 1:
        raise ValueError("False alarm constraint must be in [0,1]")
    if args.global_batch % args.micro_batch or args.micro_batch > args.global_batch:
        raise ValueError("Microbatch must divide the unchanged global batch")
    if args.auto_micro and not args.profile:
        raise ValueError("auto-micro requires measured profile")
    device = torch.device(args.device)
    if device.type == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Launch with an explicit authorized CUDA_VISIBLE_DEVICES mask")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported dataset")
    model = SingleTaskReward(pretrained_path=args.pretrained_path).to(device)
    cache_device = device if args.cache == "gpu" else torch.device("cpu")
    splits = {key: load_split(args.dataset_dir, info, model, cache_device) for key, info in manifest["splits"].items()}
    groups = [{x["group_id"] for x in manifest["splits"][key]["samples"]} for key in ("train", "val", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Dataset seed groups leak across splits")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(pretrained_sha256=sha256(args.pretrained_path), dataset_manifest_sha256=sha256(manifest_path),
                  task_name=manifest["task_name"], task_config=manifest["task_config"],
                  precision="fp32", optimizer="AdamW beta0.9,0.999 eps1e-8 weight_decay1e-5 clip1.0; constantLR; no warmup",
                  torch_version=str(torch.__version__), parameters=sum(p.numel() for p in model.parameters()),
                  official_recipe="WorldArena/RLinf reward_training.yaml architecture/LR/global64; independent tiny trainer and fixed held-out test")
    write_json(args.output_dir / "config.json", config)
    train = splits["train"]
    if args.profile:
        measured = profile(model, train["images"], train["labels"], device, sorted({32, 64, args.micro_batch}), args.global_batch)
        valid = [row for row in measured if "error" not in row]
        if not valid:
            raise RuntimeError("All microbatch profiles failed")
        if args.auto_micro:
            args.micro_batch = min(valid, key=lambda row: row["seconds"])["micro_batch"]
        inference = profile_inference(model, train["images"], device)
        args.eval_batch = min(inference, key=lambda row: row["seconds"])["batch"]
        write_json(args.output_dir / "profile.json", dict(rows=measured, inference=inference, selected_micro_batch=args.micro_batch,
                                                          selected_eval_batch=args.eval_batch, global_batch=args.global_batch,
                                                          note="BN/RNG restored; microbatch can affect BN statistics; effective global batch unchanged"))
    config.update(requested_micro_batch=config["micro_batch"], requested_eval_batch=config["eval_batch"],
                  micro_batch=args.micro_batch, eval_batch=args.eval_batch)
    write_json(args.output_dir / "config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    negative = np.flatnonzero(train["labels_numpy"] == 0)
    positive = np.flatnonzero(train["labels_numpy"] == 1)
    best_loss, best_epoch, stale, updates = math.inf, 0, 0, 0
    started = time.time()
    logs = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        selected = np.r_[positive, rng.choice(negative, size=min(len(negative), round(len(positive) * args.negative_positive_ratio)), replace=False)]
        rng.shuffle(selected)
        model.train()
        running_loss, running_samples, gradient_norms = 0., 0, []
        for offset in range(0, len(selected), args.global_batch):
            window = selected[offset:offset+args.global_batch]
            optimizer.zero_grad(set_to_none=True)
            for micro_offset in range(0, len(window), args.micro_batch):
                indices = torch.as_tensor(window[micro_offset:micro_offset+args.micro_batch], device=cache_device)
                x, y = train["images"][indices].to(device), train["labels"][indices].to(device)
                logits = model(x, preprocessed=True)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite training loss")
                (loss * len(indices) / len(window)).backward()
                running_loss += float(loss.detach()) * len(indices)
                running_samples += len(indices)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            gradient_norms.append(float(norm))
            optimizer.step()
            updates += 1
        val_loss, _ = evaluate(model, splits["val"], device, args.eval_batch)
        row = dict(epoch=epoch, updates=updates, training_samples=running_samples, train_loss=running_loss/running_samples,
                   val_loss=val_loss, grad_norm=float(np.mean(gradient_norms)), elapsed_seconds=time.time()-started)
        logs.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss - .0001:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            checkpoint = dict(schema_version=1, architecture="resnet18_mlp256", preprocess=PREPROCESS,
                              task_name=manifest["task_name"], task_config=manifest["task_config"],
                              epoch=epoch, updates=updates, val_loss=val_loss,
                              dataset_manifest_sha256=sha256(manifest_path), pretrained_sha256=config["pretrained_sha256"],
                              model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})
            torch.save(checkpoint, args.output_dir / "best.pt.tmp")
            (args.output_dir / "best.pt.tmp").replace(args.output_dir / "best.pt")
        else:
            stale += 1
        write_json(args.output_dir / "status.json", dict(phase="training", **row, best_epoch=best_epoch, best_val_loss=best_loss))
        if stale >= args.patience:
            break
    model = SingleTaskReward(checkpoint_path=args.output_dir / "best.pt").to(device)
    _, val_scores = evaluate(model, splits["val"], device, args.eval_batch)
    chosen = choose_threshold(splits["val"]["labels_numpy"], val_scores, splits["val"]["uids"], args.max_val_episode_false_alarm)
    # First access of test model outputs. Neither checkpoint nor threshold is
    # changed after this evaluation.
    test_loss, test_scores = evaluate(model, splits["test"], device, args.eval_batch)
    report = dict(schema_version=1, task_name=manifest["task_name"], task_config=manifest["task_config"],
                  checkpoint=str(args.output_dir / "best.pt"), checkpoint_sha256=sha256(args.output_dir / "best.pt"),
                  dataset_manifest_sha256=sha256(manifest_path), best_epoch=best_epoch, epochs_run=len(logs), updates=updates,
                  selected_micro_batch=args.micro_batch, global_batch=args.global_batch, elapsed_seconds=time.time()-started,
                  threshold_selection="validation only: maximum recall under episode false alarm constraint; tie lower alarms then higher threshold",
                  validation=chosen, validation_auc=auc(splits["val"]["labels_numpy"], val_scores),
                  test=metrics(splits["test"]["labels_numpy"], test_scores, splits["test"]["uids"], chosen["threshold"]),
                  test_auc=auc(splits["test"]["labels_numpy"], test_scores), test_loss=test_loss,
                  wm_domain_validated=False, deployment_authorized_by_this_report=False,
                  caveat="Small native held-out diagnostic, not a deployment guarantee; first-positive milestone may occur within an action chunk",
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)
    for split, scores in (("val", val_scores), ("test", test_scores)):
        write_json(args.output_dir / f"{split}_predictions.json", [dict(episode_uid=str(uid), action_step=int(step), label=int(label), score=float(score))
                   for uid, step, label, score in zip(splits[split]["uids"], splits[split]["steps"], splits[split]["labels_numpy"], scores)])
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    write_json(args.output_dir / "report.json", report)
    write_json(args.output_dir / "status.json", dict(phase="completed", **report))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
