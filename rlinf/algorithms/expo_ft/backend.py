"""EXPO-FT bridges for the verified SZ2 legacy PyTorch OpenPI/RoboTwin source.

Uses one Pi0.5 instance for native FM training and current-base ODE inference.
The base factory, observation transforms and native env are existing RLinf APIs;
this module adds no physics/action-adapter changes. Q never backpropagates here.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _finite(tensor: torch.Tensor, label: str) -> torch.Tensor:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Non-finite {label}")
    return tensor


def _stat_identity(stat) -> tuple[int, int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _hash_stable_file(path: Path) -> tuple[dict, tuple[int, int, int, int, int]]:
    """Stream CPU SHA256, rejecting a file changed/replaced while it is read."""
    before = _stat_identity(path.stat())
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if _stat_identity(os.fstat(stream.fileno())) != before:
            raise RuntimeError(f"Starting-model file changed before hashing: {path}")
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if _stat_identity(os.fstat(stream.fileno())) != before:
            raise RuntimeError(f"Starting-model file changed during hashing: {path}")
    if _stat_identity(path.stat()) != before:
        raise RuntimeError(f"Starting-model file was replaced during hashing: {path}")
    return {"bytes": before[2], "sha256": digest.hexdigest()}, before


def _starting_model_files(model_dir: Path) -> tuple[str, list[Path]]:
    """Mirror the verified get_model weight precedence, also pin JSON metadata."""
    full_weights = model_dir / "model_state_dict" / "full_weights.pt"
    actor_weights = model_dir / "actor" / "model_state_dict" / "full_weights.pt"
    if full_weights.is_file():
        weight_format, weights = "full_weights.pt", [full_weights]
    elif actor_weights.is_file():
        weight_format, weights = "actor/full_weights.pt", [actor_weights]
    else:
        weight_format, weights = "safetensors", sorted(model_dir.glob("*.safetensors"))
        if not weights:
            raise FileNotFoundError(f"No starting-model safetensors in {model_dir}")
    # Safetensor index, config, tokenizer/preprocessor metadata are small JSON
    # files next to the actual checkpoint. Include all such metadata explicitly.
    metadata = sorted(model_dir.glob("*.json"))
    return weight_format, sorted(set(weights + metadata))


def _starting_model_manifest(model_dir: Path) -> tuple[dict, dict[Path, tuple]]:
    weight_format, files = _starting_model_files(model_dir)
    manifest = {"weight_format": weight_format, "files": {}}
    identities = {}
    for path in files:
        record, identities[path] = _hash_stable_file(path)
        manifest["files"][path.relative_to(model_dir).as_posix()] = record
    return manifest, identities


def _verify_model_files_unchanged(model_dir: Path, manifest: Mapping, identities: Mapping) -> None:
    """Reject identity/inventory drift between hashing and the factory load."""
    weight_format, files = _starting_model_files(model_dir)
    if weight_format != manifest["weight_format"] or set(files) != set(identities):
        raise RuntimeError("Starting-model checkpoint inventory changed during loading")
    for path, identity in identities.items():
        if _stat_identity(path.stat()) != identity:
            raise RuntimeError(f"Starting-model checkpoint changed during loading: {path}")


def stack_env_observations(observations: Sequence[Mapping[str, Any]]) -> dict:
    """Stack real B=1 native observations without dropping the prompt/cameras."""
    if not observations:
        raise ValueError("Empty observation batch")
    keys = set(observations[0])
    if any(set(obs) != keys for obs in observations):
        raise ValueError("Observation schema changed inside a batch")
    stacked = {}
    for key in keys:
        values = [obs[key] for obs in observations]
        if all(value is None for value in values):
            stacked[key] = None
        elif any(value is None for value in values):
            raise ValueError(f"Mixed missing/nonmissing camera: {key}")
        elif torch.is_tensor(values[0]):
            stacked[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], np.ndarray):
            stacked[key] = np.concatenate(values, axis=0)
        elif isinstance(values[0], (list, tuple)):
            stacked[key] = [item for value in values for item in value]
        else:
            raise TypeError(f"Unsupported native observation field {key}")
    return stacked


def clone_env_observation(obs: Mapping[str, Any]) -> dict:
    """Own a CPU copy so replay images/states cannot follow later env mutation."""
    result = {}
    for key, value in obs.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().clone()
        elif isinstance(value, np.ndarray):
            result[key] = value.copy()
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        elif value is None:
            result[key] = None
        else:
            raise TypeError(f"Unsupported native observation field {key}")
    return result


class _NativeBatchAdapter(torch.nn.Module):
    """Scatter a plain tensor observation dict; build native objects per replica.

    DataParallel gathers per-example values on device0 and sums replica grads.
    It does not pool VRAM or shard the starting model/Adam states.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, processed, noise_or_target, operation, action_weights=None):
        from openpi.models import model as openpi_model
        observation = openpi_model.Observation.from_dict(dict(processed))
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if operation in ('sample', 'sample_norm', 'compare5'):
                # Factory stores sample_actions as a bound instance attribute;
                # DP shallow-copies that attribute and would call the master.
                # Bind the class implementation to the actual local replica.
                result = type(self.model).sample_actions(self.model, observation, noise=noise_or_target,
                    mode='eval', compute_values=False, norm_enabled=operation == 'sample_norm',
                    num_steps_override=5 if operation == 'compare5' else None)
                return (result['actions'], result['norm_raw']) if operation == 'sample_norm' else result['actions']
            if operation == 'fm':
                # A [1] result avoids DataParallel's scalar gather special case.
                # Pinned OpenPiActionModel.sft_forward(use_rlt=False) returns
                # super().forward(observation, actions).mean(). Its device-only
                # next(self.parameters()) wrapper is incompatible with DP
                # replicas (their weights are differentiable tensor attrs).
                # Tensor scatter already places the native inputs on each card.
                from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
                if self.model.config.use_rlt:
                    raise RuntimeError('EXPO native FM must not enter RLT loss')
                if hasattr(self.model,'gradient_checkpointing_disable'):
                    self.model.gradient_checkpointing_disable()
                loss = PI0Pytorch.forward(self.model, observation, noise_or_target.float())
                if action_weights is not None:
                    loss = loss * action_weights.detach().to(loss.device)[..., None]
                return loss.mean().reshape(1)
        raise ValueError('Unknown native parallel operation')


class Pi05Backend:
    """Sidney π0.5 H50/C10/14D, same transforms, trainable action expert.

    The explicit training filter freezes the complete VLM prefix and trains the
    Gemma action expert plus native action/time projections, without LoRA. This
    is an EXPO-FT action-expert training variant based on the existing native
    `train_expert_only` API. Trainable parameters remain FP32 master values;
    CUDA autocast uses BF16 compute and the frozen prefix retains factory dtype.
    """

    HORIZON = 50
    CHUNK = 10
    ENV_DIM = 14
    MODEL_DIM = 32
    ODE_STEPS = 10
    TRAIN_PREFIXES = (
        "paligemma_with_expert.gemma_expert.",
        "action_in_proj.",
        "action_out_proj.",
        "time_mlp_in.",
        "time_mlp_out.",
    )

    def __init__(
        self,
        model_cfg: Any,
        *,
        device: str | torch.device = "cuda:0",
        lr: float = 2.5e-5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-10,
        clip_grad: float = 1.0,
        candidate_microbatch: int = 1,
        critic_camera_keys: Sequence[str] = ("main", "left_wrist", "right_wrist"),
        source_head: str | None = None,
        parallel_devices: int = 1,
        observation_microbatch: int = 64,
        fm_microbatch: int = 64,
        image_augmentation: bool = False,
    ):
        from omegaconf import OmegaConf, open_dict
        from rlinf.models.embodiment.openpi import get_model

        self.device = torch.device(device)
        self.parallel_devices = int(parallel_devices)
        self.observation_microbatch = int(observation_microbatch)
        self.fm_microbatch = int(fm_microbatch)
        if self.parallel_devices < 1 or self.observation_microbatch < 1 or self.fm_microbatch < 1:
            raise ValueError('Invalid parallel batch configuration')
        if self.device.type != 'cuda' or self.parallel_devices > torch.cuda.device_count():
            raise ValueError('Native backend requires the configured visible CUDA devices')
        if self.parallel_devices > 1 and self.device.index not in (None, 0):
            raise ValueError('DataParallel master must be local cuda0')
        self.image_augmentation = bool(image_augmentation)
        # Explicit human decision: no image augmentation, including the native
        # compute_loss(train=True) path. Keep image normalization/resize/pad,
        # FM noise/time sampling and all model training gradients unchanged.
        import openpi.models_pytorch.preprocessing_pytorch as preprocessing
        original = preprocessing.preprocess_observation_pytorch
        if getattr(original, '_expo_no_augmentation', False):
            if self.image_augmentation:
                raise RuntimeError('Incompatible preprocessing policy in this process')
        elif not self.image_augmentation:
            original_sha = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
            @functools.wraps(original)
            def no_image_augmentation(observation, *, train=False, **kwargs):
                return original(observation, train=False, **kwargs)
            no_image_augmentation._expo_no_augmentation = True
            no_image_augmentation._expo_original_sha256 = original_sha
            preprocessing.preprocess_observation_pytorch = no_image_augmentation
        self.preprocessing_source_sha256 = getattr(preprocessing.preprocess_observation_pytorch,
            '_expo_original_sha256', hashlib.sha256(inspect.getsource(original).encode()).hexdigest())
        # Same scoped H100/runtime compatibility fix as the successful
        # tools/pi05_dv50/run_inference.py:41. Disable only cuDNN SDPA; preserve
        # standard flash/efficient/math dispatch and ordinary cuDNN convolutions.
        torch.backends.cuda.enable_cudnn_sdp(False)
        if torch.backends.cuda.cudnn_sdp_enabled():
            raise RuntimeError("cuDNN SDPA compatibility policy was not applied")
        self.runtime_attention = {
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_sdp_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
            "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        }
        self.clip_grad = float(clip_grad)
        self.candidate_microbatch = int(candidate_microbatch)
        if self.candidate_microbatch < 1 or self.clip_grad <= 0:
            raise ValueError("Invalid candidate microbatch / base grad clipping")
        self.critic_camera_keys = tuple(critic_camera_keys)
        if not self.critic_camera_keys or set(self.critic_camera_keys) - {
            "main", "left_wrist", "right_wrist"
        }:
            raise ValueError("Unknown critic camera key")
        cfg = OmegaConf.create(
            OmegaConf.to_container(model_cfg, resolve=True)
            if OmegaConf.is_config(model_cfg) else model_cfg
        )
        if str(cfg.openpi.config_name) != "pi05_sidney_robotwin":
            raise ValueError("EXPO bridge currently requires pi05_sidney_robotwin")
        if bool(cfg.get("is_lora", False)):
            raise ValueError("This source contract uses native action-expert training, no LoRA")
        with open_dict(cfg):
            cfg.openpi.use_rlt = False
            cfg.openpi.use_dsrl = False
            cfg.openpi.is_nft = False
            cfg.openpi.add_value_head = False
            cfg.openpi.rtc_enabled = False
            cfg.openpi.train_expert_only = True
            cfg.openpi.action_horizon = self.HORIZON
            cfg.openpi.action_chunk = self.CHUNK
            cfg.openpi.action_env_dim = self.ENV_DIM
            cfg.openpi.num_steps = self.ODE_STEPS
            cfg.openpi.noise_method = "flow_ode"
            cfg.add_value_head = False
            cfg.is_lora = False
            cfg.num_action_chunks = self.CHUNK
            cfg.action_dim = self.ENV_DIM
        # Read the complete actual base checkpoint before the factory consumes
        # it. Paths/head/norm alone cannot identify a frozen VLA checkpoint.
        model_dir = Path(str(cfg.model_path)).expanduser().resolve(strict=True)
        if not model_dir.is_dir():
            raise NotADirectoryError(model_dir)
        norm_path = Path(str(cfg.openpi_data.norm_stats_path)).expanduser().resolve(strict=True)
        if not norm_path.is_file():
            raise FileNotFoundError(norm_path)
        starting_model_manifest, starting_model_identities = _starting_model_manifest(model_dir)
        norm_record, norm_identity = _hash_stable_file(norm_path)
        with open_dict(cfg):
            cfg.model_path = str(model_dir)
            cfg.openpi_data.norm_stats_path = str(norm_path)
        self.model_cfg = cfg
        self.model = get_model(cfg).to(self.device)
        if bool(getattr(self.model,'torch_compile_enabled',False)):
            raise RuntimeError('This replicated native API contract requires uncompiled per-replica methods')
        _verify_model_files_unchanged(model_dir, starting_model_manifest, starting_model_identities)
        if _stat_identity(norm_path.stat()) != norm_identity:
            raise RuntimeError("Normalization file changed during loading")
        expected = {
            "action_horizon": self.HORIZON,
            "action_chunk": self.CHUNK,
            "action_env_dim": self.ENV_DIM,
            "action_dim": self.MODEL_DIM,
            "num_steps": self.ODE_STEPS,
        }
        for key, value in expected.items():
            if int(getattr(self.model.config, key)) != value:
                raise ValueError(f"Base model contract differs: {key}")
        if not bool(getattr(self.model.config, "pi05", False)):
            raise ValueError("Base is not π0.5")
        self.trainable = {}
        for name, parameter in self.model.named_parameters():
            train = name.startswith(self.TRAIN_PREFIXES)
            parameter.requires_grad_(train)
            if train:
                parameter.data = parameter.data.to(dtype=torch.float32)
                self.trainable[name] = parameter
        if not self.trainable or not any(
            name.startswith("paligemma_with_expert.gemma_expert.")
            for name in self.trainable
        ):
            raise RuntimeError("Native action-expert parameter filter matched no expert")
        self.optimizer = torch.optim.AdamW(
            self.trainable.values(), lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
        )
        self.base_updates = 0
        self.parallel_adapter = _NativeBatchAdapter(self.model)
        self.inference_receipts = []
        self.contract = {
            "schema_version": 2,
            "source_head": source_head,
            "factory": "rlinf.models.embodiment.openpi.get_model",
            "model_path": str(model_dir),
            "starting_model_manifest": starting_model_manifest,
            "config_name": str(cfg.openpi.config_name),
            "norm_stats_sha256": norm_record["sha256"],
            "horizon": self.HORIZON, "chunk": self.CHUNK,
            "env_dim": self.ENV_DIM, "model_dim": self.MODEL_DIM,
            "ode_steps": self.ODE_STEPS,
            "training_filter": list(self.TRAIN_PREFIXES),
            "trainable_names": sorted(self.trainable),
            "critic_cameras": list(self.critic_camera_keys),
            "runtime_attention": self.runtime_attention,
            "parallel": {"devices": self.parallel_devices,
                "kind": "replicated_model_batch_scatter_grad_reduce",
                "observation_microbatch": self.observation_microbatch,
                "candidate_microbatch": self.candidate_microbatch,
                "fm_microbatch": self.fm_microbatch},
            "image_augmentation": self.image_augmentation,
            "preprocessing_source_sha256": self.preprocessing_source_sha256,
            "fm_api": "pinned-use_rlt-false-native-PI0Pytorch.forward.mean; tensor-scattered-device-inputs",
            "optimizer": {
                "lr": float(lr), "betas": list(betas), "eps": float(eps),
                "weight_decay": float(weight_decay), "clip_grad": self.clip_grad,
            },
        }
        self.contract_sha256 = hashlib.sha256(
            json.dumps(self.contract, sort_keys=True).encode()
        ).hexdigest()
        self.model.eval()

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @staticmethod
    def _compatible_obs(obs: Mapping[str, Any]) -> dict:
        adapted = dict(obs)
        # The native extracted env schema has no extra-view slot; the legacy
        # OpenPI obs_processor's optional slot is made explicit, with no image.
        adapted.setdefault("extra_view_images", None)
        adapted.setdefault("wrist_images", None)
        return adapted

    def _prepare(self, env_obs: Mapping[str, Any], actions=None) -> dict:
        repacked = self.model.obs_processor(self._compatible_obs(env_obs))
        if actions is not None:
            repacked["actions"] = torch.as_tensor(actions).detach().cpu().float()
        transformed = self.model.input_transform(repacked, transpose=False)
        return self.model.precision_processor(transformed)

    @staticmethod
    def _observation(processed: Mapping[str, Any]):
        from openpi.models import model as openpi_model
        return openpi_model.Observation.from_dict(dict(processed))

    def critic_observation(self, env_obs: Mapping[str, Any]) -> dict[str, Any]:
        """224 RGB views, base-normalized state, and owned native observation.

        Three views inherit the existing RoboTwin observation contract. The
        original EXPO-FT camera count is two; this explicit port choice keeps
        the base and critic cameras aligned and is included in the contract.
        """
        main = torch.as_tensor(env_obs["main_images"])
        wrist = env_obs.get("wrist_images")
        frames = {"main": main}
        if wrist is not None:
            wrist = torch.as_tensor(wrist)
            if wrist.ndim != 5 or wrist.shape[1] != 2:
                raise ValueError("Expected two native wrist cameras")
            frames["left_wrist"] = wrist[:, 0]
            frames["right_wrist"] = wrist[:, 1]
        views = []
        for camera in self.critic_camera_keys:
            if camera not in frames:
                raise ValueError(f"Missing required critic camera {camera}")
            frame = frames[camera]
            if frame.ndim != 4 or frame.shape[-1] != 3:
                raise ValueError("Native RGB must be [B,H,W,3]")
            frame = frame.permute(0, 3, 1, 2)
            if frame.dtype != torch.uint8:
                if not frame.is_floating_point() or frame.min() < 0 or frame.max() > 1:
                    raise ValueError("Critic RGB must be uint8 or float [0,1]")
                frame = (frame * 255).round().to(torch.uint8)
            if tuple(frame.shape[-2:]) != (224, 224):
                from openpi.shared import image_tools
                resized = image_tools.resize_with_pad_torch(
                    frame.permute(0, 2, 3, 1), 224, 224
                )
                # The pinned native helper squeezes a singleton image batch.
                # Restore that axis before converting NHWC back to NCHW.
                if resized.ndim == 3 and frame.shape[0] == 1:
                    resized = resized.unsqueeze(0)
                if resized.shape != (frame.shape[0], 224, 224, 3):
                    raise ValueError("Native critic resize changed batch or RGB shape")
                frame = resized.permute(0, 3, 1, 2).round().clamp(0, 255).to(torch.uint8)
            views.append(frame)
        images = torch.stack(views, dim=1).to(self.device)
        processed = self._prepare(env_obs)
        state = _finite(processed["state"][..., :self.ENV_DIM].float(), "critic state")
        return {
            "images": images, "proprio": state,
            "env_obs": clone_env_observation(env_obs),
        }

    def sample_normalized(
        self,
        env_obs: Mapping[str, Any],
        num_candidates: int = 8,
        *,
        generator: torch.Generator | None = None,
        signal_kind: str | None = None,
    ) -> torch.Tensor | dict:
        """Current-base candidates, with optional transient collection trace.

        U stores original noise until selection; Norm observes the same forwards.
        Neither path changes candidate RNG consumption or Q selection.
        """
        if signal_kind not in (None, 'ugrow_10_5', 'norm_residual_t5_l3'):
            raise ValueError('Unknown EXPO collection signal')
        operation = 'sample_norm' if signal_kind == 'norm_residual_t5_l3' else 'sample'
        norm_rows = []
        if num_candidates < 1:
            raise ValueError("No base candidates")
        from torch.utils._pytree import tree_map
        self.model.eval()
        processed = self._prepare(env_obs)
        batch_size = int(processed["state"].shape[0])
        rows = []
        with torch.inference_mode(), self._autocast():
            # Noise is drawn once in global B,N order, independent of splitting.
            noise_all = torch.randn(batch_size, num_candidates, self.HORIZON,
                self.MODEL_DIM, device=self.device, dtype=torch.float32, generator=generator)
            for bstart in range(0, batch_size, self.observation_microbatch):
                bend = min(batch_size, bstart + self.observation_microbatch)
                sub = tree_map(lambda x:x[bstart:bend] if torch.is_tensor(x) else x,processed)
                candidate_rows = []; candidate_norms = []
                for offset in range(0, num_candidates, self.candidate_microbatch):
                    count = min(self.candidate_microbatch, num_candidates - offset)
                    repeated = tree_map(lambda x:x.repeat_interleave(count,dim=0)
                        if torch.is_tensor(x) else x,sub)
                    noise = noise_all[bstart:bend,offset:offset+count].reshape(-1,self.HORIZON,self.MODEL_DIM)
                    if self.parallel_devices > 1:
                        raw = torch.nn.parallel.data_parallel(self.parallel_adapter,
                            (repeated,noise,operation),device_ids=list(range(self.parallel_devices)),output_device=0)
                    else:
                        raw = self.parallel_adapter(repeated,noise,operation)
                    if operation == 'sample_norm':
                        raw, norm = raw
                        candidate_norms.append(norm.reshape(bend-bstart,count,self.HORIZON))
                    raw = _finite(raw.float(), 'base model candidates')
                    if tuple(raw.shape) != ((bend-bstart)*count,self.HORIZON,self.MODEL_DIM):
                        raise ValueError(f'Unexpected base output shape {tuple(raw.shape)}')
                    candidate_rows.append(raw.reshape(bend-bstart,count,self.HORIZON,self.MODEL_DIM))
                rows.append(torch.cat(candidate_rows,dim=1))
                if candidate_norms:
                    norm_rows.append(torch.cat(candidate_norms,dim=1))
        self.inference_receipts.append({'B':batch_size,'N':num_candidates,'flat_batch':batch_size*num_candidates,
            'devices':self.parallel_devices,'observation_microbatch':self.observation_microbatch,
            'candidate_microbatch':self.candidate_microbatch})
        actions = torch.cat(rows, dim=0).detach()
        if signal_kind is None:
            return actions
        return dict(actions=actions, noise=noise_all.detach(), kind=signal_kind,
                    norm_raw=torch.cat(norm_rows, dim=0).detach() if norm_rows else None,
                    base_version=self.base_updates)

    @torch.no_grad()
    def selected_signal(self, env_obs, trace, selected_index):
        """Return [B,H] signal of the selected parent; edited j maps to base j.

        Only the selected U parent needs one comparison ODE5 plus prefix encode.
        The side solve restores CUDA/CPU/Python RNG, including unused ODE noise.
        """
        from rlinf.algorithms.ugrow_signal import compute_ugrow_signal
        actions = trace['actions']; b, n = actions.shape[:2]
        selected = torch.as_tensor(selected_index, device=actions.device).long()
        if selected.shape != (b,) or (selected < 0).any() or (selected >= 2*n).any():
            raise ValueError('Selected candidate does not have a valid base parent')
        parent = selected.remainder(n); rows = torch.arange(b, device=actions.device)
        if trace['kind'] == 'norm_residual_t5_l3':
            raw = trace['norm_raw'][rows, parent]
        else:
            processed = self._prepare(env_obs)
            python_rng = random.getstate()
            try:
                with torch.random.fork_rng(devices=list(range(self.parallel_devices))), torch.inference_mode():
                    comparison = self.parallel_adapter(processed, trace['noise'][rows, parent], 'compare5')
            finally:
                random.setstate(python_rng)
            raw = compute_ugrow_signal(actions[rows, parent], comparison)
        return raw.detach(), parent.detach()

    def decode(self, env_obs: Mapping[str, Any], normalized_chunk) -> torch.Tensor:
        """Decode only the selected normalized C10/14D chunk exactly once."""
        chunk = torch.as_tensor(normalized_chunk, device=self.device).detach().float()
        if chunk.ndim != 3 or tuple(chunk.shape[1:]) != (self.CHUNK, self.ENV_DIM):
            raise ValueError(f"Expected selected normalized [B,10,14], got {tuple(chunk.shape)}")
        _finite(chunk, "selected normalized action")
        processed = self._prepare(env_obs)
        if int(processed["state"].shape[0]) != int(chunk.shape[0]):
            raise ValueError("Action/observation batch mismatch")
        padded = torch.nn.functional.pad(chunk, (0, self.MODEL_DIM - self.ENV_DIM))
        decoded = self.model.output_transform(
            {"actions": padded, "state": processed["state"]}
        )["actions"].float()
        if tuple(decoded.shape) != tuple(chunk.shape):
            raise ValueError("Canonical decode did not preserve C10/14D")
        return _finite(decoded, "canonical action").detach().cpu()

    def encode_executed(self, env_obs: Mapping[str, Any], canonical_actions) -> torch.Tensor:
        """Encode real executed joint/gripper actions through the native pipeline.

        Any true length 1..H is accepted for replay; FM below requires a full H
        target. Padding here is only model dimension 14→32, never time padding.
        """
        actions = torch.as_tensor(canonical_actions).detach().cpu().float()
        if actions.ndim != 3 or actions.shape[-1] != self.ENV_DIM:
            raise ValueError("Executed actions must be canonical [B,K,14]")
        if not 1 <= actions.shape[1] <= self.HORIZON:
            raise ValueError("Executed action horizon outside 1..50")
        _finite(actions, "canonical replay target")
        normalized = self._prepare(env_obs, actions=actions)["actions"].float()
        if tuple(normalized.shape) != (actions.shape[0], actions.shape[1], self.MODEL_DIM):
            raise ValueError("Native action encoding changed the execution horizon")
        return _finite(normalized, "normalized replay target").detach()

    def fm_update(self, observations, canonical_full_actions, action_weights=None) -> dict[str, float]:
        """Native FM on genuine successful/demo H50 action windows."""
        obs = stack_env_observations(observations) if isinstance(observations, (list, tuple)) else observations
        actions = torch.as_tensor(canonical_full_actions).float()
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (self.HORIZON, self.ENV_DIM):
            raise ValueError("Base FM requires real full-H50 canonical actions, no invented tail")
        processed = self._prepare(obs, actions=actions)
        target = processed.pop("actions")
        return self.fm_update_native(processed, target, action_weights=action_weights)

    def fm_update_native(self, observation, normalized_padded_actions, action_weights=None) -> dict[str, float]:
        """Accept a native RLinf SFT dataset (Observation, H50×32 target) batch."""
        if not isinstance(observation, Mapping):
            raise TypeError('FM uses a tensor observation dict for native per-device construction')
        target = torch.as_tensor(normalized_padded_actions, device=self.device).float()
        if target.ndim != 3 or tuple(target.shape[1:]) != (self.HORIZON, self.MODEL_DIM):
            raise ValueError("Native FM target must be normalized [B,50,32]")
        _finite(target, "FM target")
        if action_weights is not None:
            action_weights = torch.as_tensor(action_weights, device=self.device).detach().float()
            if (action_weights.shape != target.shape[:2] or
                    not torch.isfinite(action_weights).all() or (action_weights < 0).any()):
                raise ValueError('FM weights must be finite nonnegative [B,H50]')
        self.model.train()
        self.model.paligemma_with_expert.paligemma.eval()
        self.optimizer.zero_grad(set_to_none=True)
        # Sampling every trainable tensor gives a bounded update receipt rather
        # than copying the full 300M expert before every one-step FM update.
        before = {
            name: parameter.detach().reshape(-1)[:64].clone()
            for name, parameter in self.trainable.items()
        }
        batch_size = int(target.shape[0]); total_loss = 0.0; forwards = 0
        from torch.utils._pytree import tree_map
        for start in range(0,batch_size,self.fm_microbatch):
            end=min(batch_size,start+self.fm_microbatch);size=end-start
            if self.parallel_devices > 1 and size % self.parallel_devices:
                raise ValueError('FM data parallel batch must divide evenly across configured cards')
            sub=tree_map(lambda x:x[start:end] if torch.is_tensor(x) else x,observation)
            weights = None if action_weights is None else action_weights[start:end]
            if self.parallel_devices > 1:
                values=torch.nn.parallel.data_parallel(self.parallel_adapter,(sub,target[start:end],'fm',weights),
                    device_ids=list(range(self.parallel_devices)),output_device=0)
            else:
                values=self.parallel_adapter(sub,target[start:end],'fm',weights)
            loss=values.mean();_finite(loss,'FM loss')
            (loss*(size/batch_size)).backward();total_loss+=float(loss.detach())*size/batch_size;forwards+=1
        frozen_with_grad = sum(
            parameter.grad is not None
            for name, parameter in self.model.named_parameters()
            if name not in self.trainable
        )
        if frozen_with_grad:
            raise RuntimeError("Base FM gradient reached a frozen prefix parameter")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.trainable.values(), self.clip_grad, error_if_nonfinite=True
        )
        if float(grad_norm) <= 0:
            raise RuntimeError("Native base FM produced no trainable gradient")
        self.optimizer.step()
        deltas = [
            float((parameter.detach().reshape(-1)[:64] - before[name]).abs().max())
            for name, parameter in self.trainable.items()
        ]
        self.base_updates += 1
        self.model.eval()
        return {
            "base_fm_loss": total_loss,
            "base_global_batch": float(batch_size),
            "base_fm_forwards": float(forwards),
            "base_parallel_devices": float(self.parallel_devices),
            "base_grad_norm": float(grad_norm),
            "base_sampled_parameter_delta_max": max(deltas),
            "base_changed_parameter_samples": float(sum(delta > 0 for delta in deltas)),
            "base_frozen_parameters_with_grad": float(frozen_with_grad),
            "base_updates": float(self.base_updates),
        }

    def state_dict(self) -> dict:
        return {
            "contract_sha256": self.contract_sha256,
            "contract": self.contract,
            "trainable_params": {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.trainable.items()
            },
            "optimizer": self.optimizer.state_dict(),
            "base_updates": self.base_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["contract_sha256"] != self.contract_sha256:
            raise ValueError("Base checkpoint source/model/norm/training contract differs")
        if set(state["trainable_params"]) != set(self.trainable):
            raise ValueError("Base checkpoint trainable schema differs")
        with torch.no_grad():
            for name, parameter in self.trainable.items():
                saved = state["trainable_params"][name]
                if saved.shape != parameter.shape or saved.dtype != torch.float32:
                    raise ValueError(f"Base checkpoint shape/master dtype differs: {name}")
                parameter.copy_(saved.to(self.device))
        self.optimizer.load_state_dict(state["optimizer"])
        self.base_updates = int(state["base_updates"])
        self.model.eval()


_RENDER_DEVICE_ENV = "RLINF_EXPO_SAPIEN_RENDER_DEVICE"


def bind_simulator_renderer(device: str = "cuda:0") -> dict:
    """Bind default SAPIEN3 scene systems inside this EXPO process only.

    Installed SAPIEN3.0.1 SapienRenderer accepts but ignores device kwargs.
    Engine.create_scene uses the Python Scene class; its default RenderSystem
    is therefore replaced by an explicit RenderSystem(device), keeping the
    identical CPU PhysX system. Explicitly supplied scene systems stay intact.
    The dedicated environment tag makes this patch recur on spawned children
    when their EXPO main module imports backend, without modifying SAPIEN files.
    """
    if device != "cuda:0":
        raise ValueError("This isolated renderer contract uses local cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_ids=visible.split(',')
    if not visible_ids or any(not item.startswith('GPU-') for item in visible_ids) or len(set(visible_ids))!=len(visible_ids):
        raise RuntimeError("Renderer requires explicit distinct GPU UUIDs in CUDA_VISIBLE_DEVICES")
    os.environ[_RENDER_DEVICE_ENV] = device
    import sapien
    from sapien.wrapper.scene import Scene

    existing = getattr(Scene, "_rlinf_expo_render_binding", None)
    if existing is not None:
        if existing["device"] != device or existing["visible_gpu_uuid"] != visible:
            raise RuntimeError("SAPIEN renderer is already bound to a different EXPO device")
        return dict(existing)
    if str(sapien.__version__) != "3.0.1" or sapien.Scene is not Scene:
        raise RuntimeError("SAPIEN version/Scene alias differs from the verified device API")
    original_init = Scene.__init__
    expected_source = (
        "    def __init__(self, systems=None):\n"
        "        if systems is None:\n"
        "            super().__init__(\n"
        "                [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()]\n"
        "            )\n"
        "        else:\n"
        "            super().__init__(systems)\n"
    )
    scene_source = inspect.getsource(original_init)
    if scene_source.strip() != expected_source.strip():
        raise RuntimeError("SAPIEN Scene initializer differs from the verified source")

    @functools.wraps(original_init)
    def explicit_device_init(self, systems=None):
        if systems is None:
            systems = [
                sapien.physx.PhysxCpuSystem(),
                sapien.render.RenderSystem(device),
            ]
        original_init(self, systems)

    receipt = {
        "sapien_version": str(sapien.__version__), "device": device,
        "visible_gpu_uuid": visible, "api": "Scene.default_systems.RenderSystem(device)",
        "render_gpu_uuid":visible_ids[0],
        "scene_source_sha256": hashlib.sha256(scene_source.encode()).hexdigest(),
        "explicit_systems_preserved": True,
    }
    Scene.__init__ = explicit_device_init
    Scene._rlinf_expo_render_binding = receipt
    return dict(receipt)


def create_robotwin_env(env_cfg, *, num_envs=1, seed_offset=0, render_device="cuda:0"):
    """Construct a new native VectorEnv without attaching/shared Ray changes."""
    from omegaconf import OmegaConf, open_dict
    render_binding = bind_simulator_renderer(render_device)
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    cfg = OmegaConf.create(
        OmegaConf.to_container(env_cfg, resolve=True)
        if OmegaConf.is_config(env_cfg) else env_cfg
    )
    if int(cfg.max_episode_steps) != 200:
        raise ValueError("EXPO smoke inherits the existing 200-action limit")
    with open_dict(cfg):
        cfg.auto_reset = False
        cfg.ignore_terminations = False
        cfg.group_size = 1
    env = RoboTwinEnv(
        cfg=cfg, num_envs=int(num_envs), seed_offset=int(seed_offset),
        total_num_processes=1, worker_info=None, record_metrics=True,
    )
    env.expo_renderer_binding = render_binding
    return env


# Native VectorEnv uses spawn. Its child reimports the EXPO runner's top-level
# backend import after inheriting only this dedicated process environment tag.
if os.environ.get(_RENDER_DEVICE_ENV):
    bind_simulator_renderer(os.environ[_RENDER_DEVICE_ENV])
