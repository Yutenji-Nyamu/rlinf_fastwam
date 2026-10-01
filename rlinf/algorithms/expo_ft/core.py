"""PyTorch EXPO-FT, pinned to pd-perry/expo-ft@023cf9cf.

Only the independent vision / action-Q / action editor live here. A driver must
provide current-VLA normalized proposals and success-episode flow-matching BC.
Images are RGB B,V,3,H,W; proprio and actions are already model-normalized.
The editor bounds only delta; it never squashes the combined base+delta action.

Source contracts: expo_ft.py:521-693 selection; :694-730 editor objective;
:775-804 independent target pair, no entropy in TD; :888-938 update order.
This port uses variable physical executed length K rather than fixed-C replay
windows, and torchvision GN ResNet50 rather than the authors' ResNetV2.
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
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    vision_kind: str = "torchvision_resnet50_groupnorm_weights_none"

    @property
    def flat_action_dim(self) -> int:
        return self.chunk_length * self.action_dim

    @property
    def resolved_target_entropy(self) -> float:
        # Original create(), adjust_target_entropy=False: -D/2, not -D.
        return -self.flat_action_dim / 2 if self.target_entropy is None else self.target_entropy

    def validate(self) -> None:
        for key in ("chunk_length", "action_dim", "proprio_dim", "num_views", "image_size",
                    "image_latent_dim", "proprio_latent_dim", "n_base", "num_qs", "num_min_qs", "critic_updates"):
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
    """Independent trainable visual tower shared by Q/editor, no downloads.

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
               generator: torch.Generator, *, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        state = self.state_projection(proprio)
        if reference_actions.ndim == 4:
            count = reference_actions.shape[1]
            image_features = image_features[:, None, :].expand(-1, count, -1)
            state = state[:, None, :].expand(-1, count, -1)
        inputs = torch.cat((image_features, state, reference_actions.flatten(-2)), dim=-1)
        mean, log_std = self.network(inputs).chunk(2, dim=-1)
        log_std = log_std.clamp(self.config.log_std_min, self.config.log_std_max)
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

    STATE_VERSION = 1

    def __init__(self, config: ExpoConfig, device: str | torch.device = "cpu", seed: int = 42,
                 vision_encoder: nn.Module | None = None):
        super().__init__()
        config.validate()
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.seed = int(seed)
        devices = [self.device.index] if self.device.type == "cuda" else []
        # Constructor reproducibility without disturbing another model's RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.vision_encoder = ResNetVisionEncoder(config) if vision_encoder is None else vision_encoder
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
        features = self.vision_encoder(observation["images"])
        if features.shape != (observation["images"].shape[0], self.config.image_latent_dim):
            raise ValueError("vision encoder output must be B,image_latent_dim")
        if not torch.isfinite(features).all():
            raise FloatingPointError("nonfinite image features")
        return features

    def _q_pair(self) -> Tensor:
        return torch.randperm(self.config.num_qs, device=self.device,
                              generator=self.generator)[:self.config.num_min_qs]

    @torch.no_grad()
    def select_actions(self, observation: Mapping[str, Tensor], base_actions: Tensor,
                       *, deterministic_edits: bool = False) -> dict[str, Tensor]:
        obs = self._observation(observation)
        base = base_actions.to(self.device, dtype=torch.float32)
        expected = (obs["images"].shape[0], self.config.n_base,
                    self.config.chunk_length, self.config.action_dim)
        if base.shape != expected or not torch.isfinite(base).all():
            raise ValueError(f"base must be finite normalized {expected}, got {tuple(base.shape)}")
        features = self._features(obs)
        if self.config.n_edit:
            reference = base[:, :self.config.n_edit]
            delta, _ = self.editor.sample(features, obs["proprio"], reference,
                                           self.generator, deterministic=deterministic_edits)
            candidates = torch.cat((base, reference + delta), dim=1)
        else:
            candidates = base
        pair = self._q_pair()
        self.last_selection_q_indices = pair.tolist()
        scores = self.target_critic(features, obs["proprio"], candidates, pair).min(dim=0).values
        if not torch.isfinite(scores).all():
            raise FloatingPointError("nonfinite candidate Q scores")
        index = scores.argmax(dim=1)
        actions = candidates[torch.arange(len(index), device=self.device), index]
        return {"actions": actions, "index": index, "candidate_actions": candidates,
                "scores": scores, "selection_q_indices": pair}

    @torch.no_grad()
    def _polyak_critic(self) -> None:
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.lerp_(source, self.config.target_tau)

    @staticmethod
    def _grad_norm(parameters) -> float:
        # Infinity bound measures and rejects NaN/Inf without clipping gradients.
        return float(nn.utils.clip_grad_norm_(list(parameters), math.inf, error_if_nonfinite=True))

    def _critic_update(self, batch: Mapping[str, Any], next_base_sampler: Callable) -> dict[str, float]:
        obs, next_obs = self._observation(batch["obs"]), self._observation(batch["next_obs"])
        b = obs["images"].shape[0]
        actions = batch["actions"].to(self.device, dtype=torch.float32)
        if actions.shape != (b, self.config.chunk_length, self.config.action_dim):
            raise ValueError("replay actions must be normalized B,C,D")
        rewards = batch["rewards"].to(self.device, dtype=torch.float32).reshape(b)
        continuations = batch["continuations"].to(self.device, dtype=torch.float32).reshape(b)
        steps = batch["executed_steps"].to(self.device).reshape(b)
        if (steps > self.config.chunk_length).any() or (steps != steps.long()).any():
            raise ValueError("executed_steps must be integer 1..C")
        valids = batch.get("valids", torch.ones(b, device=self.device)).to(self.device, dtype=torch.float32).reshape(b)
        if not torch.isfinite(valids).all() or (valids < 0).any() or valids.sum() <= 0:
            raise ValueError("valids must be finite nonnegative with at least one valid transition")
        if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
            raise FloatingPointError("nonfinite replay action/reward")
        with torch.no_grad():
            # Preserve driver-owned RGB/prompt/raw-state fields for VLA transforms.
            next_base = next_base_sampler(batch["next_obs"])
            selection = self.select_actions(next_obs, next_base)
            next_features = self._features(next_obs)
            # Separate random pair from selection, exactly as original :781.
            pair = self._q_pair()
            self.last_bootstrap_q_indices = pair.tolist()
            next_q = self.target_critic(next_features, next_obs["proprio"], selection["actions"], pair).min(dim=0).values
            target = chunk_td_target(rewards, continuations, steps, next_q, self.config.discount)
        self.critic_optimizer.zero_grad(set_to_none=True)
        features = self._features(obs)
        values = self.critic(features, obs["proprio"], actions)
        # Author implementation multiplies by valids then averages all slots.
        loss = ((values - target[None, :]).square() * valids[None, :]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite critic loss")
        loss.backward()
        gradient = self._grad_norm([*self.vision_encoder.parameters(), *self.critic.parameters()])
        self.critic_optimizer.step()
        self._polyak_critic()
        self.critic_steps += 1
        return {"critic_loss": float(loss.detach()), "critic_grad_norm": gradient,
                "q_mean": float(values.detach().mean()), "target_mean": float(target.mean())}

    def _editor_temperature_update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        obs = self._observation(batch["obs"])
        reference = batch["actions"].to(self.device, dtype=torch.float32)
        # No editor gradient into visual encoder, matching batch_encode stop-grad.
        with torch.no_grad():
            features = self._features(obs)
        self.editor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        original_requires_grad = [parameter.requires_grad for parameter in self.critic.parameters()]
        self.critic.requires_grad_(False)
        try:
            delta, log_prob = self.editor.sample(features.detach(), obs["proprio"], reference, self.generator)
            # ALL online Q heads mean for actor; do not use selection's min pair.
            q = self.critic(features.detach(), obs["proprio"], reference + delta).mean(dim=0)
            loss = (self.config.entropy_scale * self.temperature.detach() * log_prob - q).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite editor loss")
            loss.backward()
            gradient = self._grad_norm(self.editor.parameters())
            self.editor_optimizer.step()
        finally:
            for parameter, was_trainable in zip(self.critic.parameters(), original_requires_grad):
                parameter.requires_grad_(was_trainable)
        self.editor_steps += 1
        self.temperature_optimizer.zero_grad(set_to_none=True)
        entropy = -log_prob.detach().mean()
        temperature_loss = self.temperature * (entropy - self.config.resolved_target_entropy)
        if not torch.isfinite(temperature_loss):
            raise FloatingPointError("nonfinite temperature loss")
        temperature_loss.backward()
        self._grad_norm([self.log_temperature])
        self.temperature_optimizer.step()
        self.temperature_steps += 1
        return {"editor_loss": float(loss.detach()), "editor_grad_norm": gradient,
                "edit_q": float(q.detach().mean()), "entropy": float(entropy),
                "temperature_loss": float(temperature_loss.detach()), "temperature": float(self.temperature.detach()),
                "edit_norm": float(delta.detach().flatten(1).norm(dim=1).mean())}

    def update_call(self, sample_batch: Callable[[], Mapping[str, Any]], next_base_sampler: Callable,
                    base_fm_callback: Callable[[], Mapping[str, float]] | None = None) -> dict[str, float]:
        """Q x UTD -> driver FM x 1 -> edit/temperature x 1.

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
                "editor_steps": self.editor_steps, "temperature_steps": self.temperature_steps}

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
        self.target_critic.requires_grad_(False).eval()
