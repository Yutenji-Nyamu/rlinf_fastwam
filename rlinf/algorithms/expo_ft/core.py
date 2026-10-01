"""PyTorch EXPO-FT, pinned to pd-perry/expo-ft@023cf9cf.

Only the independent vision / action-Q / action editor live here. A driver must
provide current-VLA normalized proposals and success-episode flow-matching BC.
Images are RGB B,V,3,H,W; proprio and actions are already model-normalized.
The editor bounds only delta; it never squashes the combined base+delta action.

Source contracts: expo_ft.py:521-693 selection; :694-730 editor objective;
:775-804 independent target pair, no entropy in TD; :888-938 update order.
Default vision follows the authors' joint-camera ResNetV2 in PyTorch, retaining
three RoboTwin cameras. The prior per-view torchvision encoder is legacy-only.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ExpoConfig:
    chunk_length: int = 10
    action_dim: int = 14
    proprio_dim: int = 14
    num_views: int = 3
    image_size: int = 224
    image_latent_dim: int = 512
    proprio_latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    n_base: int = 8
    n_edit: int = 8
    num_qs: int = 10
    num_min_qs: int = 2
    edit_scale: float = 0.2
    discount: float = 0.99
    target_tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    initial_temperature: float = 1.0
    entropy_scale: float = 1.0
    target_entropy: float | None = None
    critic_updates: int = 20
    critic_microbatch_size: int = 64
    editor_microbatch_size: int = 64
    selection_observation_microbatch_size: int = 64
    selection_candidate_microbatch_size: int = 8
    parallel_devices: int = 1
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    vision_kind: str = "official_resnetv2_joint_groupnorm4_spatial_flatten"

    @property
    def flat_action_dim(self) -> int:
        return self.chunk_length * self.action_dim

    @property
    def resolved_target_entropy(self) -> float:
        # Original create(), adjust_target_entropy=False: -D/2, not -D.
        return -self.flat_action_dim / 2 if self.target_entropy is None else self.target_entropy

    def validate(self) -> None:
        for key in ("chunk_length", "action_dim", "proprio_dim", "num_views", "image_size",
                    "image_latent_dim", "proprio_latent_dim", "n_base", "num_qs", "num_min_qs", "critic_updates",
                    "critic_microbatch_size", "editor_microbatch_size",
                    "selection_observation_microbatch_size", "selection_candidate_microbatch_size", "parallel_devices"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if not 0 <= self.n_edit <= self.n_base:
            raise ValueError("0 <= n_edit <= n_base is required")
        if self.n_base < 2:
            raise ValueError("EXPO candidate selection requires n_base >= 2")
        if self.num_min_qs > self.num_qs:
            raise ValueError("num_min_qs cannot exceed ensemble size")
        if self.edit_scale <= 0 or self.initial_temperature <= 0:
            raise ValueError("edit_scale and initial_temperature must be positive")
        if not 0 < self.discount <= 1 or not 0 < self.target_tau <= 1:
            raise ValueError("invalid discount / target_tau")
        if not self.hidden_dims or min(self.hidden_dims) < 1:
            raise ValueError("hidden_dims must be positive")
        if self.vision_kind not in (
            "official_resnetv2_joint_groupnorm4_spatial_flatten",
            "torchvision_resnet50_groupnorm_weights_none",
            "legacy_torchvision_resnet50_groupnorm_weights_none",
        ):
            raise ValueError("Unknown EXPO vision architecture")


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, *, layer_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for size in hidden_dims:
        layers.append(nn.Linear(input_dim, size))
        if layer_norm:
            layers.append(nn.LayerNorm(size))
        layers.append(nn.ReLU())
        input_dim = size
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def _state_project(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.Tanh())


class ResNetVisionEncoder(nn.Module):
    """Legacy trainable visual tower shared by Q/editor, no downloads.

    GroupNorm avoids minibatch-dependent running statistics at B=1. The same
    tower encodes each RGB view; their features are concatenated then projected.
    This explicit camera-fusion architecture is a port choice, not a claim of
    bitwise identity to the authors' Flax ResNetV2.
    """

    def __init__(self, config: ExpoConfig):
        super().__init__()
        from torchvision.models import resnet50
        self.config = config
        self.backbone = resnet50(weights=None, norm_layer=lambda channels: nn.GroupNorm(32, channels))
        self.backbone.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(2048 * config.num_views, config.image_latent_dim),
            nn.LayerNorm(config.image_latent_dim), nn.Tanh(),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[1:3] != (self.config.num_views, 3):
            raise ValueError("images must be B,V,3,H,W matching num_views")
        b, v, c, h, w = images.shape
        if images.dtype == torch.uint8:
            pixels = images.float() / 255.0
        elif images.is_floating_point():
            if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError("float RGB must be finite and in [0,1]")
            pixels = images.float()
        else:
            raise ValueError("RGB must be uint8 or float [0,1]")
        pixels = pixels.reshape(b * v, c, h, w)
        if (h, w) != (self.config.image_size, self.config.image_size):
            pixels = F.interpolate(pixels, size=(self.config.image_size, self.config.image_size),
                                   mode="bilinear", align_corners=False)
        features = self.backbone(pixels * 2 - 1).reshape(b, v * 2048)
        return self.projection(features)


def _same_pad(pixels: Tensor, kernel_size: int, stride: int, *, value: float = 0.0) -> Tensor:
    """Flax/JAX SAME: an odd surplus pad goes after the image, not before.

    Symmetric PyTorch padding=1 changes the sampling grid of stride-2 3x3
    convolutions and max-pooling on even feature sizes. Keep the source grid.
    """
    height, width = pixels.shape[-2:]
    pad_h = max(((height + stride - 1) // stride - 1) * stride + kernel_size - height, 0)
    pad_w = max(((width + stride - 1) // stride - 1) * stride + kernel_size - width, 0)
    return F.pad(pixels, (pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2), value=value)


class _FlaxSameConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 explicit_padding: int | None = None):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.explicit_padding = explicit_padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=0, bias=False)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, pixels: Tensor) -> Tensor:
        if self.explicit_padding is not None:
            pixels = F.pad(pixels, (self.explicit_padding,) * 4)
        else:
            pixels = _same_pad(pixels, self.kernel_size, self.stride)
        return self.conv(pixels)


class _OfficialResNetV2Block(nn.Module):
    """Source encoders.py:15-38: GN/ReLU/3x3 twice, then raw skip addition.

    The source's projection reads the original residual, not preactivated y.
    This is a basic block with no activation after the addition, not the
    1x1/3x3/1x1 bottleneck used by torchvision's postactivation ResNet50.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, in_channels, eps=1e-5)
        self.conv1 = _FlaxSameConv(in_channels, out_channels, 3, stride)
        self.norm2 = nn.GroupNorm(4, out_channels, eps=1e-5)
        self.conv2 = _FlaxSameConv(out_channels, out_channels, 3)
        self.shortcut = (_FlaxSameConv(in_channels, out_channels, 1, stride)
                         if in_channels != out_channels or stride != 1 else nn.Identity())

    def forward(self, pixels: Tensor) -> Tensor:
        residual = self.shortcut(pixels)
        features = self.conv1(F.relu(self.norm1(pixels)))
        features = self.conv2(F.relu(self.norm2(features)))
        return residual + features


class JointResNetV2VisionEncoder(nn.Module):
    """PyTorch equation port of pinned ResNetV2Encoder + BatchEncoder.

    Source: encoders.py:48-83, pixel_multiplexer.py:45-86. Camera order is the
    caller's main/left-wrist/right-wrist order, stacked into nine channels.
    One tower sees all views jointly. Blocks are (3,4,6,3), channels64..512,
    GN4, Xavier kernels, no conv bias and no global average pooling. At224,
    HWC-ordered 7x7x512 spatial features project to512 via Dense/LN/tanh.
    Three rather than the source's default two cameras is the explicit
    RoboTwin observation adaptation; no image augmentation is added here.
    """
    STAGE_SIZES = (3, 4, 6, 3)

    def __init__(self, config: ExpoConfig):
        super().__init__()
        self.config = config
        self.large_stem = config.image_size == 224
        self.stem = _FlaxSameConv(3 * config.num_views, 64,
                                  7 if self.large_stem else 3,
                                  2 if self.large_stem else 1,
                                  explicit_padding=3 if self.large_stem else None)
        stages = []
        incoming = 64
        for stage_index, count in enumerate(self.STAGE_SIZES):
            outgoing = 64 * 2 ** stage_index
            blocks = []
            for block_index in range(count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(_OfficialResNetV2Block(incoming, outgoing, stride))
                incoming = outgoing
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.final_norm = nn.GroupNorm(4, 512, eps=1e-5)
        divisor = 32 if self.large_stem else 8
        self.spatial_size = (config.image_size + divisor - 1) // divisor
        self.flat_dim = 512 * self.spatial_size ** 2
        self.projection = nn.Sequential(
            nn.Linear(self.flat_dim, config.image_latent_dim),
            nn.LayerNorm(config.image_latent_dim, eps=1e-6), nn.Tanh(),
        )
        nn.init.xavier_uniform_(self.projection[0].weight)
        nn.init.zeros_(self.projection[0].bias)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[1:3] != (self.config.num_views, 3):
            raise ValueError("images must be B,V,3,H,W matching num_views")
        batch, views, channels, height, width = images.shape
        if images.dtype == torch.uint8:
            pixels = images.float() / 255.0
        elif images.is_floating_point():
            if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError("float RGB must be finite and in [0,1]")
            pixels = images.float()
        else:
            raise ValueError("RGB must be uint8 or float [0,1]")
        pixels = pixels.reshape(batch, views * channels, height, width)
        if (height, width) != (self.config.image_size, self.config.image_size):
            pixels = F.interpolate(pixels, size=(self.config.image_size, self.config.image_size),
                                   mode="bilinear", align_corners=False)
        features = self.stem(pixels * 2 - 1)
        if self.large_stem:
            features = F.max_pool2d(_same_pad(features, 3, 2, value=-math.inf),
                                    kernel_size=3, stride=2, padding=0)
        features = F.relu(self.final_norm(self.stages(features)))
        if features.shape[1:] != (512, self.spatial_size, self.spatial_size):
            raise RuntimeError("Official ResNetV2 spatial grid differs")
        # Flax flattens HWC, whereas an ordinary torch.flatten would use CHW.
        features = features.permute(0, 2, 3, 1).contiguous().flatten(1)
        return self.projection(features)


def make_vision_encoder(config: ExpoConfig) -> nn.Module:
    if config.vision_kind == "official_resnetv2_joint_groupnorm4_spatial_flatten":
        return JointResNetV2VisionEncoder(config)
    if config.vision_kind in ("torchvision_resnet50_groupnorm_weights_none",
                              "legacy_torchvision_resnet50_groupnorm_weights_none"):
        # Keep the original module's parameter names for explicit old contracts.
        return ResNetVisionEncoder(config)
    raise ValueError("Unknown EXPO vision architecture")


class EnsembleQ(nn.Module):
    def __init__(self, config: ExpoConfig):
        super().__init__()
        self.state_projection = _state_project(config.proprio_dim, config.proprio_latent_dim)
        dim = config.image_latent_dim + config.proprio_latent_dim + config.flat_action_dim
        self.heads = nn.ModuleList(_mlp(dim, config.hidden_dims, 1, layer_norm=True)
                                   for _ in range(config.num_qs))

    def forward(self, image_features: Tensor, proprio: Tensor, actions: Tensor,
                indices: Tensor | None = None) -> Tensor:
        """Returns Q,B for one action or Q,B,N for candidate actions."""
        state = self.state_projection(proprio)
        if actions.ndim == 4:
            count = actions.shape[1]
            image_features = image_features[:, None, :].expand(-1, count, -1)
            state = state[:, None, :].expand(-1, count, -1)
        elif actions.ndim != 3:
            raise ValueError("actions must be B,C,D or B,N,C,D")
        inputs = torch.cat((image_features, state, actions.flatten(-2)), dim=-1)
        chosen = range(len(self.heads)) if indices is None else indices.tolist()
        return torch.stack([self.heads[index](inputs).squeeze(-1) for index in chosen])


class EditActor(nn.Module):
    def __init__(self, config: ExpoConfig):
        super().__init__()
        self.config = config
        self.state_projection = _state_project(config.proprio_dim, config.proprio_latent_dim)
        dim = config.image_latent_dim + config.proprio_latent_dim + config.flat_action_dim
        self.network = _mlp(dim, config.hidden_dims, 2 * config.flat_action_dim, layer_norm=False)

    def sample(self, image_features: Tensor, proprio: Tensor, reference_actions: Tensor,
               generator: torch.Generator, *, deterministic: bool = False,
               epsilon: Tensor | None = None) -> tuple[Tensor, Tensor]:
        state = self.state_projection(proprio)
        if reference_actions.ndim == 4:
            count = reference_actions.shape[1]
            image_features = image_features[:, None, :].expand(-1, count, -1)
            state = state[:, None, :].expand(-1, count, -1)
        inputs = torch.cat((image_features, state, reference_actions.flatten(-2)), dim=-1)
        mean, log_std = self.network(inputs).chunk(2, dim=-1)
        log_std = log_std.clamp(self.config.log_std_min, self.config.log_std_max)
        if epsilon is not None:
            # The four-GPU adapter samples once with the owned master RNG and
            # scatters noise with observations; replicas never reuse its GPU0
            # generator on another device. Ordinary single-device calls retain
            # the exact original sampling path below.
            if (not torch.is_tensor(epsilon) or epsilon.shape != mean.shape or
                    epsilon.device != mean.device or epsilon.dtype != mean.dtype):
                raise ValueError("External editor epsilon must match mean shape/device/dtype")
            if not torch.isfinite(epsilon).all():
                raise FloatingPointError("Nonfinite external editor epsilon")
            if deterministic and (epsilon != 0).any():
                raise ValueError("Deterministic editor requires zero external epsilon")
        else:
            epsilon = torch.zeros_like(mean) if deterministic else torch.randn(
                mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        pre_tanh = mean + log_std.exp() * epsilon
        bounded = torch.tanh(pre_tanh)
        # Stable log|d tanh(u)/du|; avoids epsilon-dependent saturation bias.
        log_jacobian = 2 * (math.log(2.0) - pre_tanh - F.softplus(-2 * pre_tanh))
        log_normal = -0.5 * (epsilon.square() + 2 * log_std + math.log(2 * math.pi))
        log_prob = (log_normal - log_jacobian).sum(dim=-1)
        log_prob = log_prob - self.config.flat_action_dim * math.log(self.config.edit_scale)
        delta = (self.config.edit_scale * bounded).reshape(reference_actions.shape)
        return delta, log_prob


def discounted_chunk_return(rewards: Tensor, executed_mask: Tensor, discount: float) -> tuple[Tensor, Tensor]:
    """Sum physical-step rewards and count an actually executed contiguous prefix."""
    if rewards.ndim != 2 or rewards.shape != executed_mask.shape:
        raise ValueError("rewards and executed_mask must be matching B,C")
    mask = executed_mask.to(rewards.dtype)
    if not ((mask == 0) | (mask == 1)).all() or (mask[:, 1:] > mask[:, :-1]).any():
        raise ValueError("executed_mask must be a binary contiguous prefix")
    steps = mask.sum(dim=-1).long()
    if (steps < 1).any():
        raise ValueError("empty executed chunks cannot train a transition")
    powers = torch.pow(torch.as_tensor(discount, device=rewards.device, dtype=rewards.dtype),
                       torch.arange(rewards.shape[1], device=rewards.device, dtype=rewards.dtype))
    return (rewards * mask * powers).sum(dim=-1), steps


def chunk_td_target(rewards: Tensor, continuations: Tensor, executed_steps: Tensor,
                    next_q: Tensor, discount: float) -> Tensor:
    """Variable-K chunk target. No entropy bonus, matching original EXPO TD."""
    if not (rewards.shape == continuations.shape == executed_steps.shape == next_q.shape):
        raise ValueError("chunk TD fields must have matching B shapes")
    if ((continuations != 0) & (continuations != 1)).any() or (executed_steps < 1).any():
        raise ValueError("continuations must be binary and executed_steps positive")
    gamma_k = torch.pow(torch.as_tensor(discount, device=next_q.device, dtype=next_q.dtype), executed_steps)
    return rewards + gamma_k * continuations * next_q


def slice_batch(value: Any, start: int, stop: int, batch_size: int) -> Any:
    """Slice a replay batch while retaining driver-owned raw observation fields.

    Tensor/NumPy arrays with a leading B axis and B-length prompt lists are
    sliced. Scalar metadata and constants are preserved. No device transfer or
    RGB copy happens here; the active microbatch is transferred by its owner.
    """
    if isinstance(value, Mapping):
        return {key: slice_batch(item, start, stop, batch_size) for key, item in value.items()}
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) and shape[0] == batch_size:
        return value[start:stop]
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return value[start:stop]
        items = [slice_batch(item, start, stop, batch_size) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


class _BatchFirstQAdapter(nn.Module):
    """Keep the observation axis first while DataParallel gathers Q outputs."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, features: Tensor, proprio: Tensor, actions: Tensor,
                indices: tuple[int, ...] | None = None) -> Tensor:
        pair = None if indices is None else torch.tensor(indices, device=features.device, dtype=torch.long)
        return self.module(features, proprio, actions, pair).movedim(0, 1)


class _EditorSampleAdapter(nn.Module):
    """Scatter master-sampled noise; no CUDA generator crosses device boundaries."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, features: Tensor, proprio: Tensor, reference: Tensor,
                epsilon: Tensor) -> tuple[Tensor, Tensor]:
        return self.module.sample(features, proprio, reference, None, epsilon=epsilon)


class ExpoLearner(nn.Module):
    """Independent action editor / ensemble / shared visual learner.

    ``select_actions(obs, base[B,N,C,D])`` is used in rollout AND backup.
    ``update_call(sample_batch, next_base_sampler, base_fm_callback)`` performs
    Q updates then driver-owned FM then one edit/temperature update.

    Batch fields: obs/next_obs dicts; actions normalized B,C,D; rewards already
    discounted B; continuations B; executed_steps B; valids optional B.
    Driver owns normalization, RGB/prompt/full-H replay, current VLA, and its
    optimizer/EMA/sampler RNG; those must be checkpointed beside this state.
    """

    STATE_VERSION = 2

    def __init__(self, config: ExpoConfig, device: str | torch.device = "cpu", seed: int = 42,
                 vision_encoder: nn.Module | None = None):
        super().__init__()
        config.validate()
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if config.parallel_devices > 1:
            if self.device.type != "cuda" or self.device.index != 0:
                raise ValueError("data parallel EXPO requires the master model on visible cuda:0")
            if torch.cuda.device_count() < config.parallel_devices:
                raise ValueError("not enough visible CUDA devices for EXPO data parallel")
        self.seed = int(seed)
        devices = [self.device.index] if self.device.type == "cuda" else []
        # Constructor reproducibility without disturbing another model's RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.vision_encoder = make_vision_encoder(config) if vision_encoder is None else vision_encoder
            self.critic = EnsembleQ(config)
            self.editor = EditActor(config)
            self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
            self.log_temperature = nn.Parameter(torch.tensor(math.log(config.initial_temperature)))
        self.to(self.device)
        self.target_critic.eval()
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.critic_optimizer = torch.optim.Adam(
            [*self.vision_encoder.parameters(), *self.critic.parameters()], lr=config.critic_lr)
        self.editor_optimizer = torch.optim.Adam(self.editor.parameters(), lr=config.actor_lr)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=config.temperature_lr)
        self.update_calls = 0
        self.critic_steps = 0
        self.editor_steps = 0
        self.temperature_steps = 0
        self.last_selection_q_indices: list[int] = []
        self.last_bootstrap_q_indices: list[int] = []

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def _observation(self, observation: Mapping[str, Tensor]) -> dict[str, Tensor]:
        images = observation["images"].to(self.device)
        proprio = observation["proprio"].to(self.device, dtype=torch.float32)
        if images.ndim != 5 or images.shape[1:3] != (self.config.num_views, 3):
            raise ValueError("images shape must be B,V,3,H,W")
        if proprio.shape != (images.shape[0], self.config.proprio_dim) or not torch.isfinite(proprio).all():
            raise ValueError("proprio must be finite normalized B,S")
        return {"images": images, "proprio": proprio}

    def _features(self, observation: Mapping[str, Tensor]) -> Tensor:
        features = self._forward(self.vision_encoder, observation["images"])
        if features.shape != (observation["images"].shape[0], self.config.image_latent_dim):
            raise ValueError("vision encoder output must be B,image_latent_dim")
        if not torch.isfinite(features).all():
            raise FloatingPointError("nonfinite image features")
        return features

    def _forward(self, module: nn.Module, *args: Any) -> Any:
        """Data parallel forward with method-specific, batch-first adapters.

        Optimizers own the original modules on cuda:0. Functional DataParallel
        replicates them per forward and automatically sums replica gradients.
        No persistent wrapper is registered, so checkpoint parameter keys stay
        unchanged. The small shared Q subset is metadata, never a scatter axis.
        """
        parallel = self.config.parallel_devices > 1
        if module is self.critic or module is self.target_critic:
            if not parallel:
                return module(*args)
            pair = None if len(args) < 4 or args[3] is None else tuple(int(index) for index in args[3].tolist())
            result = torch.nn.parallel.data_parallel(
                _BatchFirstQAdapter(module), (*args[:3], pair),
                device_ids=list(range(self.config.parallel_devices)), output_device=0)
            return result.movedim(0, 1)
        if module is self.editor:
            if not parallel:
                return module.sample(*args[:3], None, epsilon=args[3])
            return torch.nn.parallel.data_parallel(
                _EditorSampleAdapter(module), args,
                device_ids=list(range(self.config.parallel_devices)), output_device=0)
        if parallel:
            return torch.nn.parallel.data_parallel(module, args,
                                                  device_ids=list(range(self.config.parallel_devices)), output_device=0)
        return module(*args)

    def _sample_editor(self, features: Tensor, proprio: Tensor, reference: Tensor,
                       *, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        shape = (*reference.shape[:-2], self.config.flat_action_dim)
        epsilon = (torch.zeros(shape, device=self.device, dtype=features.dtype) if deterministic else
                   torch.randn(shape, device=self.device, dtype=features.dtype, generator=self.generator))
        return self._forward(self.editor, features, proprio, reference, epsilon)

    def _q_pair(self) -> Tensor:
        return torch.randperm(self.config.num_qs, device=self.device,
                              generator=self.generator)[:self.config.num_min_qs]

    @torch.no_grad()
    def _select_from_features(self, features: Tensor, proprio: Tensor, base_actions: Tensor,
                              pair: Tensor, *, deterministic_edits: bool = False) -> dict[str, Tensor]:
        """Candidate-axis microbatching; features already belong to one B slice."""
        base = base_actions.to(self.device, dtype=torch.float32)
        expected = (features.shape[0], self.config.n_base,
                    self.config.chunk_length, self.config.action_dim)
        if base.shape != expected or not torch.isfinite(base).all():
            raise ValueError(f"base must be finite normalized {expected}, got {tuple(base.shape)}")
        if self.config.n_edit:
            reference = base[:, :self.config.n_edit]
            edited_parts = []
            for start in range(0, self.config.n_edit, self.config.selection_candidate_microbatch_size):
                part = reference[:, start:start + self.config.selection_candidate_microbatch_size]
                delta, _ = self._sample_editor(features, proprio, part, deterministic=deterministic_edits)
                edited_parts.append(part + delta)
            candidates = torch.cat((base, *edited_parts), dim=1)
        else:
            candidates = base
        score_parts = []
        for start in range(0, candidates.shape[1], self.config.selection_candidate_microbatch_size):
            part = candidates[:, start:start + self.config.selection_candidate_microbatch_size]
            score_parts.append(self._forward(self.target_critic, features, proprio, part, pair).min(dim=0).values)
        scores = torch.cat(score_parts, dim=1)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("nonfinite candidate Q scores")
        index = scores.argmax(dim=1)
        actions = candidates[torch.arange(len(index), device=self.device), index]
        return {"actions": actions, "index": index, "candidate_actions": candidates,
                "scores": scores, "selection_q_indices": pair}

    @torch.no_grad()
    def select_actions(self, observation: Mapping[str, Any], base_actions: Tensor,
                       *, deterministic_edits: bool = False,
                       selection_q_indices: Tensor | None = None) -> dict[str, Tensor]:
        """Select all 8+8 candidates with bounded B/N work and one shared Q pair.

        The sampling schedule is part of the strict checkpoint configuration;
        changing microbatch sizes need not preserve bitwise random draws.
        """
        b = observation["images"].shape[0]
        if b < 1:
            raise ValueError("candidate selection needs a nonempty observation batch")
        pair = self._q_pair() if selection_q_indices is None else selection_q_indices.to(self.device)
        if (pair.shape != (self.config.num_min_qs,) or pair.dtype != torch.long
                or pair.unique().numel() != self.config.num_min_qs
                or (pair < 0).any() or (pair >= self.config.num_qs).any()):
            raise ValueError("selection_q_indices must be a distinct valid ensemble subset")
        expected = (b, self.config.n_base, self.config.chunk_length, self.config.action_dim)
        if base_actions.shape != expected:
            raise ValueError(f"base must have shape {expected}, got {tuple(base_actions.shape)}")
        self.last_selection_q_indices = pair.tolist()
        parts: list[dict[str, Tensor]] = []
        for start in range(0, b, self.config.selection_observation_microbatch_size):
            stop = min(start + self.config.selection_observation_microbatch_size, b)
            obs = self._observation(slice_batch(observation, start, stop, b))
            features = self._features(obs)
            parts.append(self._select_from_features(features, obs["proprio"], base_actions[start:stop],
                                                     pair, deterministic_edits=deterministic_edits))
        return {**{key: torch.cat([part[key] for part in parts], dim=0)
                   for key in ("actions", "index", "candidate_actions", "scores")},
                "selection_q_indices": pair}

    @torch.no_grad()
    def _polyak_critic(self) -> None:
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.lerp_(source, self.config.target_tau)

    @staticmethod
    def _grad_norm(parameters) -> float:
        # Infinity bound measures and rejects NaN/Inf without clipping gradients.
        return float(nn.utils.clip_grad_norm_(list(parameters), math.inf, error_if_nonfinite=True))

    def _critic_update(self, batch: Mapping[str, Any], next_base_sampler: Callable) -> dict[str, float]:
        b = batch["obs"]["images"].shape[0]
        if b < 1 or batch["next_obs"]["images"].shape[0] != b:
            raise ValueError("replay current/next observations must share a nonempty batch")
        if batch["actions"].shape != (b, self.config.chunk_length, self.config.action_dim):
            raise ValueError("replay actions must be normalized B,C,D")
        valids = batch.get("valids", torch.ones(b, device=self.device)).to(self.device, dtype=torch.float32).reshape(b)
        if not torch.isfinite(valids).all() or (valids < 0).any() or valids.sum() <= 0:
            raise ValueError("valids must be finite nonnegative with at least one valid transition")
        # One subset for selection and a separately sampled subset for backup,
        # shared by every observation/candidate shard of this global update.
        selection_pair, backup_pair = self._q_pair(), self._q_pair()
        self.last_selection_q_indices = selection_pair.tolist()
        self.last_bootstrap_q_indices = backup_pair.tolist()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=self.device)
        q_sum = torch.zeros((), device=self.device)
        target_sum = torch.zeros((), device=self.device)
        microbatches = 0
        for start in range(0, b, self.config.critic_microbatch_size):
            stop = min(start + self.config.critic_microbatch_size, b)
            part = slice_batch(batch, start, stop, b)
            obs, next_obs = self._observation(part["obs"]), self._observation(part["next_obs"])
            size = stop - start
            actions = part["actions"].to(self.device, dtype=torch.float32)
            rewards = part["rewards"].to(self.device, dtype=torch.float32).reshape(size)
            continuations = part["continuations"].to(self.device, dtype=torch.float32).reshape(size)
            steps = part["executed_steps"].to(self.device).reshape(size)
            if (steps > self.config.chunk_length).any() or (steps != steps.long()).any():
                raise ValueError("executed_steps must be integer 1..C")
            if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
                raise FloatingPointError("nonfinite replay action/reward")
            with torch.no_grad():
                # Raw RGB/prompt/state remain in the sliced driver observation.
                next_base = next_base_sampler(part["next_obs"])
                next_features = self._features(next_obs)
                selection = self._select_from_features(next_features, next_obs["proprio"],
                                                       next_base, selection_pair)
                next_q = self._forward(self.target_critic, next_features, next_obs["proprio"],
                                       selection["actions"], backup_pair).min(dim=0).values
                target = chunk_td_target(rewards, continuations, steps, next_q, self.config.discount)
            features = self._features(obs)
            values = self._forward(self.critic, features, obs["proprio"], actions)
            # Author formula is mean(valids * MSE), including invalid slots in
            # the denominator. Divide by GLOBAL Q*B, never a microbatch mean.
            loss = ((values - target[None, :]).square() * valids[None, start:stop]).sum() / (self.config.num_qs * b)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite critic loss")
            loss.backward()
            loss_sum += loss.detach()
            q_sum += values.detach().sum() / (self.config.num_qs * b)
            target_sum += target.sum() / b
            microbatches += 1
            # Release each shard's inference tensors and graph before the next.
            del features, values, loss, next_features, next_q, target, next_base, selection
        gradient = self._grad_norm([*self.vision_encoder.parameters(), *self.critic.parameters()])
        self.critic_optimizer.step()
        self._polyak_critic()
        self.critic_steps += 1
        return {"critic_loss": float(loss_sum), "critic_grad_norm": gradient,
                "q_mean": float(q_sum), "target_mean": float(target_sum),
                "critic_global_batch": float(b), "critic_microbatches": float(microbatches)}

    def _editor_temperature_update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        b = batch["obs"]["images"].shape[0]
        if b < 1 or batch["actions"].shape != (b, self.config.chunk_length, self.config.action_dim):
            raise ValueError("editor replay must contain normalized nonempty B,C,D actions")
        self.editor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        original_requires_grad = [parameter.requires_grad for parameter in self.critic.parameters()]
        self.critic.requires_grad_(False)
        loss_sum = torch.zeros((), device=self.device)
        q_sum = torch.zeros((), device=self.device)
        entropy = torch.zeros((), device=self.device)
        edit_norm = torch.zeros((), device=self.device)
        microbatches = 0
        try:
            for start in range(0, b, self.config.editor_microbatch_size):
                stop = min(start + self.config.editor_microbatch_size, b)
                part = slice_batch(batch, start, stop, b)
                obs = self._observation(part["obs"])
                reference = part["actions"].to(self.device, dtype=torch.float32)
                # No editor gradient into shared visual encoder.
                with torch.no_grad():
                    features = self._features(obs)
                delta, log_prob = self._sample_editor(features.detach(), obs["proprio"], reference)
                # ALL online heads mean; selection's min pair is not used here.
                q = self._forward(self.critic, features.detach(), obs["proprio"], reference + delta).mean(dim=0)
                loss = (self.config.entropy_scale * self.temperature.detach() * log_prob - q).sum() / b
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite editor loss")
                loss.backward()
                loss_sum += loss.detach()
                q_sum += q.detach().sum() / b
                entropy -= log_prob.detach().sum() / b
                edit_norm += delta.detach().flatten(1).norm(dim=1).sum() / b
                microbatches += 1
                del features, delta, log_prob, q, loss
            gradient = self._grad_norm(self.editor.parameters())
            self.editor_optimizer.step()
        finally:
            for parameter, was_trainable in zip(self.critic.parameters(), original_requires_grad):
                parameter.requires_grad_(was_trainable)
        self.editor_steps += 1
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss = self.temperature * (entropy - self.config.resolved_target_entropy)
        if not torch.isfinite(temperature_loss):
            raise FloatingPointError("nonfinite temperature loss")
        temperature_loss.backward()
        self._grad_norm([self.log_temperature])
        self.temperature_optimizer.step()
        self.temperature_steps += 1
        return {"editor_loss": float(loss_sum), "editor_grad_norm": gradient,
                "edit_q": float(q_sum), "entropy": float(entropy),
                "temperature_loss": float(temperature_loss.detach()), "temperature": float(self.temperature.detach()),
                "edit_norm": float(edit_norm), "editor_global_batch": float(b),
                "editor_microbatches": float(microbatches)}

    def update_call(self, sample_batch: Callable[[], Mapping[str, Any]], next_base_sampler: Callable,
                    base_fm_callback: Callable[[], Mapping[str, float]] | None = None) -> dict[str, float]:
        """Q x UTD -> driver FM x 1 -> edit/temperature x 1.

        Each sampled batch is global B; microbatches/replicas never increment
        optimizer, Polyak, or update counters. The driver FM callback likewise
        owns one global successful-data batch and one accumulated optimizer
        update, regardless of its internal microbatch size.

        Omitting FM is supported for named ablations / isolated core tests only;
        production EXPO-FT driver must pass a success-buffer FM callback.
        """
        last_batch = None
        metrics: dict[str, float] = {}
        for _ in range(self.config.critic_updates):
            last_batch = sample_batch()
            metrics.update(self._critic_update(last_batch, next_base_sampler))
        if base_fm_callback is not None:
            for key, value in base_fm_callback().items():
                metrics[f"base/{key}"] = float(value)
            metrics["base/fm_callback_called"] = 1.0
        else:
            metrics["base/fm_callback_called"] = 0.0
        metrics.update(self._editor_temperature_update(last_batch))
        self.update_calls += 1
        metrics.update({"update_calls": float(self.update_calls), "critic_steps": float(self.critic_steps),
                        "editor_steps": float(self.editor_steps), "temperature_steps": float(self.temperature_steps)})
        return metrics

    def get_extra_state(self) -> dict[str, Any]:
        """Included in ordinary nn.Module.state_dict alongside all parameters."""
        return {"version": self.STATE_VERSION, "config": asdict(self.config), "seed": self.seed,
                "rng_state": self.generator.get_state(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "editor_optimizer": self.editor_optimizer.state_dict(),
                "temperature_optimizer": self.temperature_optimizer.state_dict(),
                "update_calls": self.update_calls, "critic_steps": self.critic_steps,
                "editor_steps": self.editor_steps, "temperature_steps": self.temperature_steps,
                "last_selection_q_indices": self.last_selection_q_indices,
                "last_bootstrap_q_indices": self.last_bootstrap_q_indices}

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if state["version"] != self.STATE_VERSION or state["config"] != asdict(self.config):
            raise ValueError("EXPO checkpoint method/config differs; strict restore refused")
        self.seed = state["seed"]
        self.generator.set_state(state["rng_state"].cpu())
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.editor_optimizer.load_state_dict(state["editor_optimizer"])
        self.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
        for key in ("update_calls", "critic_steps", "editor_steps", "temperature_steps"):
            setattr(self, key, int(state[key]))
        self.last_selection_q_indices = list(state["last_selection_q_indices"])
        self.last_bootstrap_q_indices = list(state["last_bootstrap_q_indices"])
        self.target_critic.requires_grad_(False).eval()
