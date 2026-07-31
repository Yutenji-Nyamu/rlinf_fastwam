# Official QAM numerical oracle

This directory is an isolated fixture generator for the locked upstream source:

```text
https://github.com/ColinQiyangLi/qam
2726d767c9a0a7a46d49693f0391f73dc2cf58ac
```

The upstream repository is MIT-licensed; its commit and the hashes of every
executed source file are embedded in the fixture metadata.

It is not imported by production RLinf. In particular, do not install its JAX
dependencies into the shared π0/RLinf environment.

## Generate the fixture

On the server, create a fresh CPU-only environment without
`--system-site-packages`:

```bash
python3.11 -m venv /root/autodl-tmp/venvs/qam-oracle-2726d767
/root/autodl-tmp/venvs/qam-oracle-2726d767/bin/python -m pip install \
  -r tests/algorithms/qam/oracle/requirements.lock.txt
JAX_PLATFORMS=cpu \
  /root/autodl-tmp/venvs/qam-oracle-2726d767/bin/python \
  tests/algorithms/qam/oracle/export_official_fixture.py \
  --source /root/autodl-tmp/oracles/qam-2726d767 \
  --output tests/algorithms/qam/oracle/qam_official_2726d767_v1.npz
```

The exporter refuses a different or tracked-dirty upstream commit. It uses the
official `QAMAgent` implementation with a small deterministic problem:

- batch `B=2`;
- action dimension `D=4`;
- AM flow steps `K=3`;
- ten independent Q heads;
- two hidden layers of width eight.

The output contains only numeric NumPy arrays. JSON metadata, parameter paths,
source hashes, and the full installed distribution list are UTF-8 encoded into
`uint8` arrays, so this check is valid:

```python
with np.load(path, allow_pickle=False) as fixture:
    ...
```

The fixture records all PRNG keys and sampled tensors needed to replay the
official loss, the FM and AM intermediates, terminal-Q gradient and reverse
adjoints, raw/clipped parameter gradients, one combined Optax Adam step, and
the official pre-update-parameter EMA targets. Normal production tests only
read the committed `.npz`; they must not import JAX or regenerate it.

The direct pins follow the lower-bound era declared by upstream. The exporter
also records every resolved transitive distribution. The fixture is accepted
only after the fresh environment imports the locked source and the exporter
finishes its internal consistency checks.

`resolved-freeze.txt` records the exact 39-package environment that generated
the committed fixture; `requirements.lock.txt` intentionally lists only the
eight direct pins used to construct that fresh environment.
