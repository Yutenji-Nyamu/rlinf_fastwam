"""CPU: merged odd/even Adam shards must produce the identical next update."""
import copy
import torch
from resume_two_rank import merge_optimizer_shards


def main():
    torch.manual_seed(123)
    parameters = [torch.nn.Parameter(torch.randn(n)) for n in (7, 8)]
    reference = torch.optim.AdamW(parameters, lr=5e-6)
    for parameter in parameters:
        parameter.grad = torch.randn_like(parameter)
    reference.step()
    full = reference.state_dict()
    shards = [copy.deepcopy(full), copy.deepcopy(full)]
    for index, row in full['state'].items():
        for key, value in row.items():
            if torch.is_tensor(value) and value.ndim:
                width = (value.numel() + 1) // 2
                padded = torch.nn.functional.pad(value, (0, width * 2 - value.numel()))
                for rank in (0, 1):
                    shards[rank]['state'][index][key] = padded[rank*width:(rank+1)*width].clone()
    copied = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    restored = torch.optim.AdamW(copied, lr=5e-6)
    merged, proof = merge_optimizer_shards(*shards, restored)
    restored.load_state_dict(merged)
    for a, b in zip(parameters, copied):
        gradient = torch.randn_like(a)
        a.grad, b.grad = gradient.clone(), gradient.clone()
    reference.step()
    restored.step()
    assert all(torch.equal(a, b) for a, b in zip(parameters, copied))
    broken = copy.deepcopy(shards[1])
    broken['state'][0]['step'] += 1
    try:
        merge_optimizer_shards(shards[0], broken, restored)
    except AssertionError:
        pass
    else:
        raise AssertionError('Unequal optimizer steps must reject')
    assert not torch.cuda.is_initialized()
    print('CPU_MERGE_TEST_OK: odd/even padding, exact next AdamW update, scalar mismatch rejection')


if __name__ == '__main__':
    main()
