# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Native WebSocket servers with private task state and shared immutable JAX weights."""

import asyncio
import copy
import sys
from pathlib import Path

from common import read, save
from policy_entry import install


def clone_model(base):
    # Native Model has only task/observation metadata plus its Policy. Policy.infer
    # mutates RNG and sample kwargs; those belong to each lane, not to the weights.
    model = copy.copy(base)
    model.policy = copy.copy(base.policy)
    model.policy._sample_kwargs = dict(base.policy._sample_kwargs)
    model.model = model.policy
    model.observation_window = None
    model._latest_env_idx_list = [0]
    return model


async def serve(deployment):
    d = Path(deployment)
    cfg = read(d / 'template.json')
    p = read(d / 'prepared.json')
    repo = Path(cfg['repo'])
    sys.path[:0] = [str(repo), str(repo / 'XPolicyLab')]
    from XPolicyLab.policy.Pi_05.model import Model
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    base = Model(cfg['policy'])
    # All calls using the shared JAX executable are serialized, as in the native
    # server. Simulators still run concurrently while another lane infers.
    lock = asyncio.Lock()
    servers = []
    models = []
    for i in range(p['lanes']):
        lane = d / f'lane-{i}'
        model = clone_model(base)
        install(model, lane / 'current-job.json', lane)
        server = PolicyServer(model, PolicyServerConfig(
            host='127.0.0.1', port=cfg['policy']['port'] + i))
        server._model_lock = lock
        models.append(model)
        servers.append(server)
    assert len({id(m.policy) for m in models}) == p['lanes']
    assert len({id(m.policy._sample_kwargs) for m in models}) == p['lanes']
    assert all(m.policy._model is base.policy._model for m in models)
    assert all(m.policy._sample_actions is base.policy._sample_actions for m in models)
    for server in servers:
        await server.start()
    save(d / 'policy-ready.json', {
        'lanes': p['lanes'], 'ports': [s.config.port for s in servers],
        'shared_weights': True, 'private_observation_kwargs_rng': True,
        'serialized_inference': True, 'batch_mode': 'native sequential',
    })
    try:
        await asyncio.Future()
    finally:
        await asyncio.gather(*(s.stop() for s in servers))


if __name__ == '__main__':
    asyncio.run(serve(sys.argv[1]))
