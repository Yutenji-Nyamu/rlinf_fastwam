"""One fixed native dataset, language Success and numeric terminal separation.

No policy training, progress-difference reward, or reward threshold deployment.
Reuses the already verified B16 service and pinned official numeric forward.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np


def summarize(rows):
    positive = [r for r in rows if r["reference_success"]]
    negative = [r for r in rows if not r["reference_success"]]
    known = [r for r in rows if r["language"]["parse_status"] == "ok"]
    confusion = {
        "true_positive": sum(r["reference_success"] and r["language"]["success"] is True for r in known),
        "false_negative": sum(r["reference_success"] and r["language"]["success"] is False for r in known),
        "false_positive": sum(not r["reference_success"] and r["language"]["success"] is True for r in known),
        "true_negative": sum(not r["reference_success"] and r["language"]["success"] is False for r in known),
        "unknown": len(rows) - len(known),
    }
    values_pos = [r["last_remaining_value"] for r in positive]
    values_neg = [r["last_remaining_value"] for r in negative]
    auc = (sum((a < b) + 0.5 * (a == b) for a in values_pos for b in values_neg)
           / (len(values_pos) * len(values_neg))) if values_pos and values_neg else None
    stats = lambda xs: dict(count=len(xs), minimum=min(xs), median=float(np.median(xs)), maximum=max(xs)) if xs else dict(count=0)
    # Describe threshold feasibility; do not fit or promote an operational threshold.
    separation = bool(max(values_pos) < min(values_neg)) if values_pos and values_neg else None
    return dict(total=len(rows), native_successes=len(positive), native_failures=len(negative),
        language=confusion, numeric_success=stats(values_pos), numeric_failure=stats(values_neg),
        auc_lower_value_predicts_success=auc, strict_low_value_separation_on_this_sample=separation,
        deployed_threshold=None,
        interpretation="Terminal 0/1 diagnostic only. Small native sample is not an OpenDW-generated-domain guarantee. Unknown language output is excluded, not counted as No.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("numeric-module", "service-module", "official-inference", "model-path", "manifest-path", "cases-json", "samples-npz", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--physical-gpu", type=int, choices=(4,), required=True)
    p.add_argument("--batch-size", type=int, choices=(4, 8, 16), default=16)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("Probe requires exactly physical GPU4 / PCI_BUS_ID")
    if args.output.exists() or args.output.suffix != ".json":
        raise ValueError("A fresh JSON output path is required")
    import importlib.util
    spec = importlib.util.spec_from_file_location("binary_pinned_numeric", str(args.numeric_module))
    numeric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(numeric)
    service = numeric.load_module("binary_pinned_service", args.service_module)
    cases, arrays = numeric.load_cases(args.cases_json, args.samples_npz)
    if not 1 <= len(cases) <= 48 or any(c["kind"] != "native_eval" or type(c.get("reference_success")) is not bool for c in cases):
        raise ValueError("Expected at most48 native cases with explicit boolean simulator labels")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(schema_version=1, mode="native_binary_diagnostic", physical_gpu=4,
        started_at_unix=time.time(), engineering_passed=False, reward_replacement_authorized=False,
        cases=[], batches=[], sources={k: dict(path=str(getattr(args, k)), sha256=numeric.sha_file(getattr(args, k)))
            for k in ("numeric_module", "service_module", "official_inference", "manifest_path", "cases_json", "samples_npz")})
    numeric.save(args.output, result)
    runtime, failed = None, None
    try:
        block = numeric.official_numeric(args.official_inference)
        runtime = service.RynnSuccessService(str(args.model_path), str(args.manifest_path), 4,
            batch_size=args.batch_size, robot_description=numeric.ROBOT, camera_description=numeric.CAMERA)
        result["fingerprint"] = runtime.fingerprint
        prepared, statistics = [], []
        for case in cases:
            sample, stats = numeric.prepare(runtime, case, arrays[case["frames_key"]])
            prepared.append(sample)
            statistics.append(stats)
        del arrays
        runtime.onload()
        wrapper = numeric.CaptureModel(runtime.model)
        namespace = dict(model=wrapper, torch=runtime.torch, device=runtime.device)
        exec(block, namespace)
        ordered = [None] * len(cases)
        for indices in service.bucket_indices(prepared, args.batch_size):
            samples = [prepared[i] for i in indices]
            numbers, timing = numeric.forward(runtime, namespace["run_batch"], wrapper, samples)
            result["batches"].append(dict(mode="numeric", case_ids=[cases[i]["id"] for i in indices], **timing))
            language, diag = runtime._generate_batch(samples, 128)
            result["batches"].append(dict(mode="language", case_ids=[cases[i]["id"] for i in indices], **diag))
            attempts = [[dict(row)] for row in language]
            retry = [j for j, row in enumerate(language) if row["parse_status"] in ("missing_success", "truncated")]
            if retry:
                rows_retry, retry_diag = runtime._generate_batch([samples[j] for j in retry], 256)
                result["batches"].append(dict(mode="language_retry", case_ids=[cases[indices[j]]["id"] for j in retry], **retry_diag))
                for j, row in zip(retry, rows_retry):
                    attempts[j].append(dict(row))
                    language[j] = row
            for j, index in enumerate(indices):
                ordered[index] = {**cases[index], **statistics[index], **numbers[j], "language": language[j], "language_attempts": attempts[j]}
            result["cases"] = [row for row in ordered if row is not None]
            numeric.save(args.output, result)
            print(json.dumps(dict(completed=len(result["cases"]), total=len(cases))), flush=True)
        result["summary"] = summarize(result["cases"])
        result["engineering_passed"] = True
    except BaseException as exc:
        failed = exc
        result["error"] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        if runtime is not None:
            try:
                result["offload"] = runtime.offload()
                if not result["offload"]["ok"] or not result["offload"]["is_offloaded"]:
                    raise RuntimeError("Synchronized CPU offload was not acknowledged")
            except BaseException as exc:
                failed = failed or exc
                result["offload_error"] = str(exc)
        result["finished_at_unix"] = time.time()
        result["elapsed_s"] = result["finished_at_unix"] - result["started_at_unix"]
        result["engineering_passed"] = result["engineering_passed"] and failed is None
        numeric.save(args.output, result)
    print(json.dumps(dict(result=str(args.output), engineering_passed=result["engineering_passed"], summary=result.get("summary"))), flush=True)
    if failed is not None:
        raise RuntimeError("Rynn binary diagnostic failed; inspect result.json") from failed


if __name__ == "__main__":
    main()
