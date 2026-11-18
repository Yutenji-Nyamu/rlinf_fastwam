import csv
import time
import torch


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def generate(teacher, obs, k, seed):
    n = len(obs['states'])
    group = {}
    for key, value in obs.items():
        if torch.is_tensor(value):
            group[key] = value.repeat_interleave(k, dim=0)
        elif isinstance(value, list):
            group[key] = [item for item in value for _ in range(k)]
        else:
            group[key] = value
    device = teacher.action_in_proj.weight.device
    shape = (n, teacher.config.action_horizon, teacher.config.action_dim)
    # A separate stream per candidate preserves candidate zero for every K.
    noise = torch.stack([torch.randn(shape, device=device,
        generator=torch.Generator(device=device).manual_seed(seed + i))
        for i in range(k)], dim=1).flatten(0, 1)
    actions, result = teacher.predict_action_batch(group, noise=noise, num_steps=10,
                                                    mode='eval', compute_values=False)
    model_actions = result['model_actions']
    assert actions.shape[0] == model_actions.shape[0] == n*k
    samples = model_actions.reshape(n, k, *model_actions.shape[1:])[..., :14]
    forward = {key: value[::k].contiguous() for key, value in result['forward_inputs'].items()
               if torch.is_tensor(value)}
    return actions[::k].contiguous(), model_actions[::k].contiguous(), samples, forward


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

