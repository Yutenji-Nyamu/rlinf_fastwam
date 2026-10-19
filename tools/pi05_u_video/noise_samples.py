"""Observe three extra noise samples without changing the executed main pass."""
import numpy as np
import torch


def numpy_exact(value):
    """Preserve the environment action dtype; only NumPy-unsupported bf16 casts."""
    value=value.detach().cpu()
    return (value.float() if value.dtype==torch.bfloat16 else value).numpy()


def repeated_observation(obs, copies):
    """Sample-major ordering: all B slots for sample1, then sample2, etc."""
    result = {}
    for key, value in obs.items():
        if value is None:
            result[key] = None
        elif torch.is_tensor(value):
            result[key] = torch.cat([value] * copies, dim=0)
        elif isinstance(value, np.ndarray):
            result[key] = np.concatenate([value] * copies, axis=0)
        elif isinstance(value, list):
            result[key] = value * copies
        else:
            raise TypeError(f"Unsupported observation field: {key}: {type(value)}")
    return result


def collect_noise_samples(model, obs, main_trace, main_actions, seed):
    """K=4: retain native B16 main output; evaluate extra three as one B48 batch.

    Batching the executed main pass itself would change its numerical path.
    Its action and global RNG are preserved; extra samples are observation only.
    """
    initial = main_trace["initial_noise_full"]
    batch, horizon, dim = initial.shape
    generator = torch.Generator(device=initial.device).manual_seed(int(seed))
    with torch.random.fork_rng(devices=[initial.device.index]), torch.inference_mode():
        noise = torch.randn((3 * batch, horizon, dim), device=initial.device,
                            dtype=initial.dtype, generator=generator)
        actions, result = model.predict_action_batch(
            repeated_observation(obs, 3), mode="eval", compute_values=False,
            num_steps=10, return_dvac_telemetry=True, record_signals=False,
            noise=noise)
        final = result["dvac_telemetry"]["final_model_action"]
        def stack(main, extra):
            extra = extra.reshape(3, batch, *extra.shape[1:]).transpose(0, 1)
            return numpy_exact(torch.cat([main[:, None], extra], dim=1))
        return {
            "noise4_initial_noise": stack(initial, noise),
            "noise4_model_actions": stack(main_trace["final_model_action"], final),
            "noise4_env_actions": stack(main_actions, actions),
            "noise4_seed": np.asarray(seed, dtype=np.int64),
        }
