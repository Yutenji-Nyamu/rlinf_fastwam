"""Extend the accepted native observer with prefix Q and final velocity-head input.

The main M10 branch is observed unchanged. The existing M5 and noise4 probes
are excluded by their already-explicit record_signals=False sampler scope.
"""
import importlib
import time

from legacy_observer import AttentionObserver, LAYERS, cpu
import torch


class ObjectAttentionObserver(AttentionObserver):
    def __init__(self, model):
        super().__init__(model)
        prefix_model = model.paligemma_with_expert.paligemma.language_model
        modules = prefix_model.layers
        selected = {id(modules[i].self_attn): i for i in LAYERS}
        gm = importlib.import_module(type(modules[0].self_attn).__module__)
        original = gm.eager_attention_forward

        def attention(module, query, key, value, attention_mask, scaling, *args, **kwargs):
            result = original(module, query, key, value, attention_mask, scaling, *args, **kwargs)
            if self.active and id(module) in selected:
                started = time.perf_counter()
                lid = selected[id(module)]
                assert lid not in self.prefix_q
                self.prefix_q[lid] = cpu(query)
                self.prefix_k_native[lid] = cpu(key)
                self.prefix_scales[lid] = float(scaling)
                if lid == LAYERS[0]:
                    self.prefix_attention_mask = cpu(attention_mask)
                if self.reference:
                    self.prefix_refs[lid] = cpu(result[1][:1])
                self.qk_copy_sync_seconds += time.perf_counter() - started
            return result

        gm.eager_attention_forward = attention
        self.originals.append((gm, "eager_attention_forward", original))
        original_forward = prefix_model.forward

        def forward(*args, **kwargs):
            if self.active:
                self.native_prefix_positions = cpu(kwargs["position_ids"])
            return original_forward(*args, **kwargs)

        prefix_model.forward = forward
        self.originals.append((prefix_model, "forward", original_forward))
        self.wrap("_sample_actions_with_prefix_cache", self.sampler_scope)

        def head_input(module, args):
            if self.active and self.step == 9:
                self.pre_velocity_hidden = cpu(args[0])

        self.handles.append(model.action_out_proj.register_forward_pre_hook(head_input))

    def begin(self, reference=False):
        super().begin(reference)
        self.prefix_q, self.prefix_k_native, self.prefix_scales, self.prefix_refs = {}, {}, {}, {}
        self.pre_velocity_hidden = None

    def sampler_scope(self, original, *args, **kwargs):
        active = self.active
        if active and not kwargs.get("record_signals", False):
            self.active = False
        try:
            return original(*args, **kwargs)
        finally:
            self.active = active

    def finish(self, result, actions):
        raw, meta = super().finish(result, actions)
        assert set(self.prefix_q) == set(LAYERS)
        assert self.pre_velocity_hidden is not None
        for lid in LAYERS:
            assert torch.equal(self.prefix_k_native[lid], self.prefix[lid]), "Native prefix cache changed"
        raw.update(
            q_prefix=torch.stack([self.prefix_q[l] for l in LAYERS], dim=1),
            prefix_attention_mask=self.prefix_attention_mask,
            prefix_position_ids=self.native_prefix_positions,
            prefix_scaling=[self.prefix_scales[l] for l in LAYERS],
            pre_velocity_hidden=self.pre_velocity_hidden,
        )
        if self.reference:
            raw["prefix_attention_reference"] = torch.stack([self.prefix_refs[l] for l in LAYERS], dim=1)
        meta.update(
            schema="pi05-attn-objects-v1",
            parent_observer="pi05-attn-recovery-20261012/tools/pi05_attn/observer.py",
            main_chain_only=True, prefix_cache_exact=True,
            prefix_query_heads=raw["q_prefix"].shape[2], prefix_key_heads=raw["k_prefix"].shape[2],
            action_query_heads=raw["q_action"].shape[3], action_key_heads=raw["k_action"].shape[3],
            hidden_capture="last main get_velocity action_out_proj input, after expert final norm",
            prefix_capture="actual native eager prefix call, post-RoPE/pre-scaling",
        )
        return raw, meta
