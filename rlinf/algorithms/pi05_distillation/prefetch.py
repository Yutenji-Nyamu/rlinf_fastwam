# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only, ordered lookahead without duplicating the in-memory dataset."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import numpy as np


def training_indices(data, start, stop, global_batch, micro, rank, world, seed):
    """Preserve the existing per-step/rank/microbatch order, including resume."""
    if global_batch % (world * micro):
        raise ValueError("Global batch must divide evenly across ranks/microbatches")
    per_epoch = (len(data) + global_batch - 1) // global_batch
    last_epoch, permutation = None, None
    for step in range(start, stop):
        epoch, offset = divmod(step, per_epoch)
        if hasattr(data, "indices"):
            indices = data.indices(step, global_batch, seed)
        else:
            if last_epoch != epoch:
                permutation = np.random.default_rng(seed + epoch).permutation(len(data))
                last_epoch = epoch
            indices = np.resize(permutation, per_epoch * global_batch)[
                offset * global_batch : (offset + 1) * global_batch
            ]
        local = indices[rank::world]
        for m in range(global_batch // (world * micro)):
            yield local[m * micro : (m + 1) * micro]


def decoded_batches(data, indices, *, decode_workers=0, prefetch_batches=0):
    """Yield identical CPU batches; lookahead never touches model/global RNG.

    Close this iterator when stopping early. One producer shares compressed RAM
    records; decoder threads only unpack images. Exceptions propagate to caller.
    """
    if decode_workers < 0 or not 0 <= prefetch_batches <= 2:
        raise ValueError("Use nonnegative decoder workers and 0-2 prefetched batches")
    with ExitStack() as stack:
        decoder = (
            stack.enter_context(ThreadPoolExecutor(decode_workers))
            if decode_workers > 1
            else None
        )

        def load(batch_indices):
            return data.batch(batch_indices, executor=decoder)

        source = iter(indices)
        if not prefetch_batches:
            for batch_indices in source:
                yield load(batch_indices)
            return
        producer = stack.enter_context(ThreadPoolExecutor(1))
        pending = deque()
        for _ in range(prefetch_batches):
            batch_indices = next(source, None)
            if batch_indices is None:
                break
            pending.append(producer.submit(load, batch_indices))
        try:
            while pending:
                batch = pending.popleft().result()
                batch_indices = next(source, None)
                if batch_indices is not None:
                    pending.append(producer.submit(load, batch_indices))
                yield batch
                del batch
        finally:
            for future in pending:
                future.cancel()
