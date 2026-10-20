"""Action-position direct object reading and same-layer visual/object overlap."""
import numpy as np
import torch

from objects import masks


def repeat_keys(key, heads):
    assert heads % key.shape[1] == 0
    return key.repeat_interleave(heads // key.shape[1], dim=1)


def reconstruct(raw, metadata, device="cuda"):
    source = raw["source"].to(device)
    valid = raw["key_valid"].to(device)
    mask, annotations, labels = masks(metadata, raw["key_valid"].numpy())
    object_mask = torch.as_tensor(mask, device=device, dtype=torch.float32)
    B, K, J = object_mask.shape
    P, V = raw["prefix_valid"].shape[-1], metadata["language_start"]
    visual_valid = (source[:, :V] == 0) & valid[:, :V]
    maps, prefix_checks, action_checks = [], [], []
    max_row_error = 0.
    with torch.inference_mode():
        for layer in range(3):
            q = raw["q_prefix"][:, layer].to(device)
            key = repeat_keys(raw["k_prefix"][:, layer].to(device), q.shape[1])
            pm = raw["prefix_attention_mask"].to(device)
            pieces = []
            for lo in range(0, V, 128):
                hi = min(lo + 128, V)
                logits = (q[:, :, lo:hi].float() @ key.float().transpose(-2, -1)) * raw["prefix_scaling"][layer]
                probs = torch.softmax(logits + pm[:, :, lo:hi], dim=-1)
                max_row_error = max(max_row_error, float((probs.sum(-1) - 1).abs().max()))
                pieces.append(torch.matmul(probs.mean(1), object_mask[:, :P]))
                if "prefix_attention_reference" in raw:
                    native_logits = (q[:1, :, lo:hi] @ key[:1].transpose(-2, -1)) * raw["prefix_scaling"][layer]
                    native = torch.softmax(native_logits + pm[:1, :, lo:hi], dim=-1, dtype=torch.float32).to(q.dtype)
                    reference = raw["prefix_attention_reference"][:, layer, :, lo:hi].to(device)
                    prefix_checks.append(float((native - reference).abs().max()))
            maps.append(torch.cat(pieces, dim=1) * visual_valid[..., None])
        direct, via, masses, entropies = [], [], [], []
        per_layer_direct, per_layer_via = [], []
        diagnostic_maps = []
        for step in range(10):
            ds, vs, ms, es, visual_maps = [], [], [], [], []
            for layer in range(3):
                q = raw["q_action"][:, step, layer].to(device=device, dtype=torch.float32)
                key = torch.cat([raw["k_prefix"][:, layer], raw["k_action"][:, step, layer]], dim=-2).to(device=device, dtype=torch.float32)
                key = repeat_keys(key, q.shape[1])
                logits = (q @ key.transpose(-2, -1)) * raw["scaling"][layer]
                a = torch.softmax(logits + raw["attention_mask"][:, step].to(device=device, dtype=torch.float32), dim=-1)
                max_row_error = max(max_row_error, float((a.sum(-1) - 1).abs().max()))
                if "native_attention_reference" in raw:
                    nq = raw["q_action"][:1, step, layer].to(device)
                    nk = repeat_keys(torch.cat([raw["k_prefix"][:1, layer], raw["k_action"][:1, step, layer]], dim=-2).to(device), nq.shape[1])
                    native_logits = (nq @ nk.transpose(-2, -1)) * raw["scaling"][layer]
                    native = torch.softmax(native_logits + raw["attention_mask"][:1, step].to(device),
                                           dim=-1, dtype=torch.float32).to(nq.dtype)
                    reference = raw["native_attention_reference"][:, step, layer].to(device)
                    action_checks.append(float((native - reference).abs().max()))
                mean = a.mean(1)
                ds.append(torch.matmul(a, object_mask[:, None]))
                vs.append(torch.matmul(a[:, :, :, :V], maps[layer][:, None]))
                ms.append(torch.stack([(mean * ((source == gi) & valid)[:, None]).sum(-1) for gi in range(5)], dim=-1))
                vp = a[:, :, :, :V] * visual_valid[:, None, None]
                vp = vp / vp.sum(-1, keepdim=True).clamp_min(1e-30)
                es.append(-(vp * vp.clamp_min(1e-30).log()).sum(-1).mean(1))
                visual_maps.append(mean[:, :, :V])
            d, v = torch.stack(ds, dim=1), torch.stack(vs, dim=1)
            per_layer_direct.append(d.cpu().numpy())
            per_layer_via.append(v.cpu().numpy())
            direct.append(d.mean((1, 2)))
            via.append(v.mean((1, 2)))
            masses.append(torch.stack(ms).mean(0))
            entropies.append(torch.stack(es).mean(0))
            if step in (0, 4, 9):
                diagnostic_maps.append(torch.stack(visual_maps).mean(0).cpu().numpy())
        d, v, m = [torch.stack(rows, dim=1) for rows in (direct, via, masses)]
        present = object_mask.sum(1) > 0
        d = torch.where(present[:, None, None], d, torch.nan)
        v = torch.where(present[:, None, None], v, torch.nan)
        dc = torch.where(m[..., 1:2] > 0, d / m[..., 1:2], torch.nan)
        vc = torch.where(m[..., 0:1] > 0, v / m[..., 0:1], torch.nan)
        assert max_row_error < 2e-5
        assert torch.all(torch.isnan(d) | ((d >= 0) & (d <= m[..., 1:2] + 2e-6)))
        assert torch.all(torch.isnan(v) | ((v >= 0) & (v <= m[..., 0:1] + 2e-6)))
        if prefix_checks:
            assert max(prefix_checks) == 0, "Prefix native attention reconstruction mismatch"
        if action_checks:
            assert max(action_checks) == 0, "Action native attention reconstruction mismatch"
        arrays = {name: tensor.cpu().numpy() for name, tensor in (
            ("I_direct", d), ("I_via_visual", v), ("I_total", d + v),
            ("I_direct_conditional", dc), ("I_via_visual_conditional", vc),
            ("modality_mass", m), ("object_valid", present),
            ("visual_entropy", torch.stack(entropies, dim=1)))}
        arrays.update(
            I_direct_layer_head=np.stack(per_layer_direct, axis=1),
            I_via_visual_layer_head=np.stack(per_layer_via, axis=1),
            object_visual_maps=np.stack([x.cpu().numpy() for x in maps], axis=1),
            action_visual_maps_1_5_10=np.stack(diagnostic_maps, axis=1),
            object_token_mask=mask,
            dv=raw["native_dv_telemetry"]["z_endpoint"][:, -5:].float().numpy().astype(np.float64).var(axis=1).sum(-1),
        )
    report = dict(labels=labels, annotations=annotations, main_denoise_index=9,
                  modality_order=["visual", "instruction", "state", "action", "other"],
                  row_sum_max_error=max_row_error, prefix_native_max_error=max(prefix_checks, default=None),
                  action_native_max_error=max(action_checks, default=None),
                  object_valid_slots=int(present[:, 0].sum()), slots=B, passed=True,
                  total_definition="raw direct + raw visual-overlap; not a probability",
                  via_definition="same-layer descriptive overlap, not causal flow")
    return arrays, report
