"""Torch port of RynnValue's jaxrl2 PixelIQL, independent of the VLA actor.

Reference: RynnValue@10e0d333f5f3811d0d130587e50f1faf48da49e5,
pi-rl/third_party/jaxrl2/{agents/pi_iql,networks/encoders,networks/values}.
The local adaptation is a 50x14 normalized action proposal, not model padding.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rlinf.algorithms.online_iql import advantage_weights, expectile_loss, weight_diagnostics


@dataclass(frozen=True)
class PixelIQLConfig:
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.8
    beta: float = 10.0
    max_weight: float = 100.0
    seed: int = 20260909
    inference_batch_size: int = 64

    def __post_init__(self):
        for key in ("critic_lr", "value_lr", "beta", "max_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("discount", "tau"):
            if not math.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        if not 0 < self.expectile < 1:
            raise ValueError("expectile must be in (0,1)")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (isinstance(self.inference_batch_size, bool) or
                int(self.inference_batch_size) != self.inference_batch_size or self.inference_batch_size < 1):
            raise ValueError("inference_batch_size must be a positive integer")


def stack_pixels(image: torch.Tensor, wrist_image: torch.Tensor) -> torch.Tensor:
    """Raw main BHWC + wrists B2HWC -> private uint8 B9x224x224.

    Use OpenPI's exact deterministic resize-with-pad, before its float image
    transform.  This helper never invokes the actor or consumes RNG.
    """
    main = torch.as_tensor(image)
    wrist = torch.as_tensor(wrist_image)
    if (main.dtype != torch.uint8 or wrist.dtype != torch.uint8 or main.ndim != 4 or
            wrist.ndim != 5 or wrist.shape[1] != 2 or main.shape[0] != wrist.shape[0] or
            main.shape[-1] != 3 or wrist.shape[-1] != 3 or main.shape[0] == 0):
        raise ValueError("Expected uint8 main[B,H,W,3] and wrists[B,2,H,W,3]")
    cameras = (main, wrist[:, 0], wrist[:, 1])
    resized = []
    for camera in cameras:
        arr = camera.detach().cpu().numpy()
        if min(arr.shape[1:3]) < 1:
            raise ValueError("Empty camera image")
        if arr.shape[1:3] != (224, 224):
            from openpi_client import image_tools
            arr = np.asarray(image_tools.resize_with_pad(arr, 224, 224))
        if arr.dtype != np.uint8 or arr.shape[1:] != (224, 224, 3):
            raise ValueError("OpenPI resize must preserve uint8 and produce 224x224 RGB")
        resized.append(torch.from_numpy(np.array(arr, copy=True)).permute(0, 3, 1, 2))
    return torch.cat(resized, dim=1).contiguous()


def same_padding(x: torch.Tensor, kernel: int, stride: int, value: float = 0.0) -> torch.Tensor:
    """Flax/JAX SAME: odd padding goes on the trailing spatial edge."""
    h, w = x.shape[-2:]
    ph = max((math.ceil(h / stride) - 1) * stride + kernel - h, 0)
    pw = max((math.ceil(w / stride) - 1) * stride + kernel - w, 0)
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=value)


class SameConv(nn.Conv2d):
    def forward(self, x):
        return super().forward(same_padding(x, self.kernel_size[0], self.stride[0]))


class ResNetBlock(nn.Module):
    def __init__(self, incoming: int, channels: int, stride: int):
        super().__init__()
        self.conv1 = SameConv(incoming, channels, 3, stride=stride, bias=False, dtype=torch.float32)
        self.norm1 = nn.GroupNorm(4, channels, eps=1e-5, dtype=torch.float32)
        self.conv2 = SameConv(channels, channels, 3, bias=False, dtype=torch.float32)
        self.norm2 = nn.GroupNorm(4, channels, eps=1e-5, dtype=torch.float32)
        self.projection = None
        if incoming != channels or stride != 1:
            self.projection = SameConv(incoming, channels, 1, stride=stride, bias=False, dtype=torch.float32)
            self.projection_norm = nn.GroupNorm(4, channels, eps=1e-5, dtype=torch.float32)

    def forward(self, x):
        residual = x if self.projection is None else self.projection_norm(self.projection(x))
        y = F.relu(self.norm1(self.conv1(x)))
        return F.relu(residual + self.norm2(self.conv2(y)))


class SpatialSoftmax(nn.Module):
    def __init__(self, height: int = 28, width: int = 28):
        super().__init__()
        # Match the reference meshgrid's default xy indexing and concatenation.
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, width, dtype=torch.float32),
                                torch.linspace(-1, 1, height, dtype=torch.float32), indexing="ij")
        self.register_buffer("pos_x", xx.flatten())
        self.register_buffer("pos_y", yy.flatten())

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2] * x.shape[-1] != self.pos_x.numel():
            raise ValueError("Spatial-softmax feature shape mismatch")
        p = x.flatten(2).softmax(dim=-1)  # fixed temperature=1
        return torch.cat(((p * self.pos_x).sum(-1), (p * self.pos_y).sum(-1)), dim=-1)


def _truncated_variance_normal_(tensor: torch.Tensor, variance: float) -> None:
    # JAX variance_scaling's truncated-normal correction for truncation at +/-2.
    std = math.sqrt(variance) / 0.87962566103423978
    nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)


class PixelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(9, 64, 7, stride=2, padding=3, bias=False, dtype=torch.float32)
        self.stem_norm = nn.GroupNorm(4, 64, eps=1e-5, dtype=torch.float32)
        layers, incoming = [], 64
        for stage, channels in enumerate((64, 128, 256, 512)):
            for block in range(2):
                stride = 2 if stage == 1 and block == 0 else 1
                layers.append(ResNetBlock(incoming, channels, stride))
                incoming = channels
        self.blocks = nn.ModuleList(layers)
        self.spatial = SpatialSoftmax()
        self.bottleneck = nn.Linear(1024, 50, dtype=torch.float32)
        self.bottleneck_norm = nn.LayerNorm(50, eps=1e-6, dtype=torch.float32)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                fan_in = layer.weight.shape[1] * layer.weight.shape[2] * layer.weight.shape[3]
                _truncated_variance_normal_(layer.weight, 2.0 / fan_in)
        _truncated_variance_normal_(self.bottleneck.weight, 2.0 / (1024 + 50))
        nn.init.zeros_(self.bottleneck.bias)

    def forward(self, pixels):
        if pixels.dtype != torch.uint8 or pixels.ndim != 4 or pixels.shape[1:] != (9, 224, 224):
            raise ValueError("Pixel encoder expects uint8[B,9,224,224]")
        x = F.relu(self.stem_norm(self.stem(pixels.float() / 255.0)))
        x = F.max_pool2d(same_padding(x, 3, 2, -float("inf")), 3, stride=2)
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.bottleneck_norm(self.bottleneck(self.spatial(x))))


class ValueMLP(nn.Module):
    def __init__(self, incoming: int, layer_norm: bool):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(incoming, 256, dtype=torch.float32),
                                     nn.Linear(256, 256, dtype=torch.float32), nn.Linear(256, 1, dtype=torch.float32)])
        self.norms = nn.ModuleList([nn.LayerNorm(256, eps=1e-6, dtype=torch.float32) for _ in range(2)]) if layer_norm else None
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < 2:
                x = F.relu(x if self.norms is None else self.norms[i](x))
        return x.squeeze(-1)


class PixelQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PixelEncoder()
        self.heads = nn.ModuleList([ValueMLP(750, True), ValueMLP(750, True)])

    def forward(self, pixels, actions):
        # Official _flatten_dict sorts 'actions' before 'states'/'pixels'.
        x = torch.cat((actions.flatten(1), self.encoder(pixels)), dim=-1)
        return torch.stack([head(x) for head in self.heads], dim=0)


class PixelV(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PixelEncoder()
        self.head = ValueMLP(50, False)

    def forward(self, pixels):
        return self.head(self.encoder(pixels))


def random_crop(pixels: torch.Tensor, generator: torch.Generator,
                offsets: torch.Tensor | None = None) -> torch.Tensor:
    """Edge-pad4/crop with one shift per sample shared by its nine channels."""
    b, _, h, w = pixels.shape
    if offsets is None:
        offsets = torch.randint(0, 9, (b, 2), generator=generator, device="cpu")
    if offsets.shape != (b, 2) or ((offsets < 0) | (offsets > 8)).any():
        raise ValueError("Crop offsets must be [B,2] integers in 0..8")
    if offsets.is_floating_point():
        raise ValueError("Crop offsets must be integral")
    offsets = offsets.to(device=pixels.device, dtype=torch.long)
    ys = (torch.arange(h, device=pixels.device)[None] + offsets[:, :1] - 4).clamp(0, h - 1)
    xs = (torch.arange(w, device=pixels.device)[None] + offsets[:, 1:] - 4).clamp(0, w - 1)
    bhwc = pixels.permute(0, 2, 3, 1)
    result = bhwc[torch.arange(b, device=pixels.device)[:, None, None], ys[:, :, None], xs[:, None, :]]
    return result.permute(0, 3, 1, 2).contiguous()


def _cpu_clone(tree):
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().clone()
    if isinstance(tree, dict):
        return {key: _cpu_clone(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_cpu_clone(x) for x in tree]
    if isinstance(tree, tuple):
        return tuple(_cpu_clone(x) for x in tree)
    return copy.deepcopy(tree)


class PixelIQLLearner:
    """Single-device FP32 learner; owns no actor state and no global RNG."""
    def __init__(self, config: Mapping | PixelIQLConfig | None = None, device="cpu"):
        self.config = config if isinstance(config, PixelIQLConfig) else PixelIQLConfig(**dict(config or {}))
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        # Module constructors themselves initialize parameters. Save/restore CPU
        # RNG around both construction and our explicit reference initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(self.config.seed))
            self.critic = PixelQ()
            self.value = PixelV()
        self.critic.to(device=self.device, dtype=torch.float32)
        self.value.to(device=self.device, dtype=torch.float32)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False).eval()
        opts = dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr, **opts)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.config.value_lr, **opts)
        self.crop_generator = torch.Generator(device="cpu").manual_seed(int(self.config.seed) + 1)
        self.update_count = 0

    def _inputs(self, pixels, actions=None):
        if not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.uint8:
            raise ValueError("Critic pixels must be uint8")
        if pixels.ndim != 4 or pixels.shape[1:] != (9, 224, 224) or not pixels.shape[0]:
            raise ValueError("Critic pixels must be nonempty [B,9,224,224]")
        p = pixels.detach().to(self.device)
        if actions is None:
            return p
        if actions.shape != (len(pixels), 50, 14) or not actions.is_floating_point():
            raise ValueError("Critic actions must be normalized floating [B,50,14]")
        a = actions.detach().to(self.device, dtype=torch.float32)
        if not torch.isfinite(a).all():
            raise ValueError("Nonfinite critic actions")
        return p, a

    @staticmethod
    def _finite(value, label):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite IQL {label}")

    @staticmethod
    def _grad_norm(model):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        PixelIQLLearner._finite(norm, "gradient norm")
        return float(norm)

    @torch.no_grad()
    def _soft_update(self):
        current = dict(self.critic.named_parameters())
        target = dict(self.target_critic.named_parameters())
        if current.keys() != target.keys():
            raise RuntimeError("IQL target parameter keys differ")
        for name, p in target.items():
            p.mul_(1.0 - self.config.tau).add_(current[name], alpha=self.config.tau)
        current_buffers = dict(self.critic.named_buffers())
        target_buffers = dict(self.target_critic.named_buffers())
        if current_buffers.keys() != target_buffers.keys():
            raise RuntimeError("IQL target buffer keys differ")
        for name, buffer in target_buffers.items():
            buffer.copy_(current_buffers[name])

    def update(self, pixels, actions, next_pixels, rewards, bootstrap_masks):
        p, a = self._inputs(pixels, actions)
        pn = self._inputs(next_pixels)
        if p.shape != pn.shape:
            raise ValueError("Current/next critic image batches differ")
        r = torch.as_tensor(rewards, device=self.device, dtype=torch.float32).detach()
        masks = torch.as_tensor(bootstrap_masks, device=self.device, dtype=torch.float32).detach()
        if r.shape != (len(p),) or masks.shape != r.shape:
            raise ValueError("Rewards and bootstrap masks must be [B]")
        self._finite(r, "rewards")
        if not ((masks == 0) | (masks == 1)).all():
            raise ValueError("IQL bootstrap masks must be binary")
        self.critic.train()
        self.value.train()
        p = random_crop(p, self.crop_generator)
        pn = random_crop(pn, self.crop_generator)
        with torch.autocast(device_type=self.device.type, enabled=False):
            with torch.no_grad():
                qt = self.target_critic(p, a).amin(0)
            self._finite(qt, "target Q")
            self.value_optimizer.zero_grad(set_to_none=True)
            v = self.value(p)
            self._finite(v, "V")
            delta = qt - v
            lv = expectile_loss(delta, self.config.expectile).mean()
            self._finite(lv, "V loss")
            lv.backward()
            v_grad = self._grad_norm(self.value)
            self.value_optimizer.step()
            self.value_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                next_v = self.value(pn)
                self._finite(next_v, "next V")
                y = r + self.config.discount * masks * next_v
            self.critic_optimizer.zero_grad(set_to_none=True)
            qs = self.critic(p, a)
            self._finite(qs, "Q")
            lq = (qs - y[None]).square().mean()
            self._finite(lq, "Q loss")
            lq.backward()
            q_grad = self._grad_norm(self.critic)
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            self._soft_update()
        self.update_count += 1
        return {
            "value_loss": float(lv.detach()), "critic_loss": float(lq.detach()),
            "value_grad_norm": v_grad, "critic_grad_norm": q_grad,
            "q_mean": float(qs.detach().mean()), "v_mean": float(v.detach().mean()),
            "q_target_mean": float(y.mean()), "q_in_v_mean": float(qt.mean()),
            "next_v_mean": float(next_v.mean()), "td_q_abs_mean": float((qs.detach() - y).abs().mean()),
            "td_v_abs_mean": float(delta.detach().abs().mean()),
            "expectile_pos_fraction": float((delta.detach() > 0).float().mean()),
            "q_disagreement_mean": float((qs[0].detach() - qs[1].detach()).abs().mean()),
            "batch_size": float(len(p)), "updates": float(self.update_count),
        }

    @torch.no_grad()
    def advantages(self, pixels, actions, batch_size=None):
        size = self.config.inference_batch_size if batch_size is None else int(batch_size)
        if size < 1:
            raise ValueError("inference batch size must be positive")
        # Validate on each inference slice: never transfer all 1024 image stacks
        # to GPU just to split them afterwards.
        if len(pixels) != len(actions) or not len(pixels):
            raise ValueError("Invalid IQL advantage batch")
        qs_out, vs_out = [], []
        q_mode, v_mode = self.critic.training, self.value.training
        self.critic.eval()
        self.value.eval()
        try:
            with torch.autocast(device_type=self.device.type, enabled=False):
                for start in range(0, len(pixels), size):
                    p, a = self._inputs(pixels[start:start + size], actions[start:start + size])
                    qs, v = self.critic(p, a), self.value(p)
                    self._finite(qs, "advantage Q")
                    self._finite(v, "advantage V")
                    qs_out.append(qs)
                    vs_out.append(v)
                qs, v = torch.cat(qs_out, 1), torch.cat(vs_out)
                advantage = qs.amin(0) - v
                weights = advantage_weights(advantage, self.config.beta, self.config.max_weight)
        finally:
            self.critic.train(q_mode)
            self.value.train(v_mode)
        return {"q1": qs[0], "q2": qs[1], "v": v, "advantage": advantage, "weights": weights}

    def weight_metrics(self, result):
        stats = weight_diagnostics(result["weights"])
        a = result["advantage"].detach().float()
        stats.update({"adv_mean": float(a.mean()), "adv_min": float(a.min()), "adv_max": float(a.max()),
                      "adv_pos_fraction": float((a > 0).float().mean()),
                      "weight_clip_fraction": float((result["weights"] >= self.config.max_weight).float().mean())})
        return stats

    def identity(self):
        return {"schema": "pixel_iql_torch_v1", "config": asdict(self.config),
                "architecture": "rynn_resnet18_gn4_stride8_spatial1_latent50_q_ln_v_no_ln",
                "pixels": [9, 224, 224], "action_shape": [50, 14], "dtype": "float32"}

    def state_dict(self):
        return _cpu_clone({"identity": self.identity(), "critic": self.critic.state_dict(),
                           "value": self.value.state_dict(), "target_critic": self.target_critic.state_dict(),
                           "critic_optimizer": self.critic_optimizer.state_dict(),
                           "value_optimizer": self.value_optimizer.state_dict(),
                           "crop_generator_state": self.crop_generator.get_state(), "update_count": self.update_count})

    def load_state_dict(self, state):
        if state.get("identity") != self.identity():
            raise ValueError("IQL checkpoint method/config identity mismatch")
        count = state.get("update_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid IQL checkpoint update count")
        # Preflight all model keys/shapes/finiteness before changing any model.
        for name in ("critic", "value", "target_critic"):
            expected = getattr(self, name).state_dict()
            supplied = state[name]
            if expected.keys() != supplied.keys():
                raise ValueError(f"IQL {name} checkpoint keys differ")
            for key, value in supplied.items():
                if value.shape != expected[key].shape or value.dtype != expected[key].dtype or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid IQL checkpoint tensor {name}/{key}")
        check_rng = torch.Generator(device="cpu")
        check_rng.set_state(state["crop_generator_state"].cpu())
        for name in ("critic", "value", "target_critic"):
            getattr(self, name).load_state_dict(state[name], strict=True)
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])
        self.crop_generator.set_state(state["crop_generator_state"].cpu())
        self.update_count = count
        self.target_critic.requires_grad_(False).eval()
