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

"""CPU loader contract: exact tensors/order, stateless resume, error propagation."""

import io
import random
from contextlib import closing

import numpy as np
import pytest
import torch
from PIL import Image

from rlinf.algorithms.pi05_distillation.data import balanced_indices, decode_records
from rlinf.algorithms.pi05_distillation.prefetch import decoded_batches, training_indices


class TinyWindows:
    horizon = 2

    def __init__(self):
        self.row = {
            "states": np.arange(12 * 14, dtype=np.float32).reshape(12, 14),
            "actions": np.arange(12 * 14, dtype=np.float32).reshape(12, 14) / 3,
        }
        rng = np.random.default_rng(91)
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            images = []
            for _ in range(12):
                buf = io.BytesIO()
                Image.fromarray(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)).save(
                    buf, format="JPEG"
                )
                images.append({"bytes": buf.getvalue()})
            self.row["observation.images." + name] = images

    def __len__(self):
        return 11

    def batch(self, indices, *, executor=None):
        return decode_records(
            ((self.row, int(i), f"instruction-{i}") for i in indices),
            self.horizon,
            executor,
        )


def assert_batch_equal(a, b):
    assert torch.equal(a[1], b[1])
    for key in a[0]:
        if torch.is_tensor(a[0][key]):
            assert torch.equal(a[0][key], b[0][key])
        else:
            assert a[0][key] == b[0][key]


@pytest.mark.parametrize("prefetch", [0, 1, 2])
def test_tensors_order_rng_and_early_close(prefetch):
    data = TinyWindows()
    ids = [np.array([5, 1, 5]), np.array([10, 2, 0]), np.array([3, 7, 4])]
    expected = [data.batch(i) for i in ids]
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    with closing(decoded_batches(data, ids, decode_workers=4, prefetch_batches=prefetch)) as batches:
        for actual, reference in zip(batches, expected):
            assert_batch_equal(actual, reference)
    assert random.getstate() == python_rng
    current = np.random.get_state()
    assert current[0] == numpy_rng[0] and current[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(current[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    with closing(decoded_batches(data, ids, decode_workers=4, prefetch_batches=prefetch)) as batches:
        assert_batch_equal(next(batches), expected[0])


def test_sampler_matches_existing_loop_and_resume_for_both_scopes():
    class Balanced:
        def __len__(self):
            return 23

        def indices(self, step, size, seed):
            return balanced_indices([7, 5, 11], step, size, seed)

    for data in [TinyWindows(), Balanced()]:
        for rank in [0, 1]:
            whole = list(training_indices(data, 0, 5, 8, 2, rank, 2, 73))
            per_epoch = (len(data) + 7) // 8
            expected = []
            for step in range(5):
                epoch, offset = divmod(step, per_epoch)
                if hasattr(data, "indices"):
                    ids = data.indices(step, 8, 73)
                else:
                    perm = np.random.default_rng(73 + epoch).permutation(len(data))
                    ids = np.resize(perm, per_epoch * 8)[offset * 8 : (offset + 1) * 8]
                local = ids[rank::2]
                expected.extend([local[:2], local[2:]])
            np.testing.assert_array_equal(whole, expected)
            np.testing.assert_array_equal(
                list(training_indices(data, 2, 5, 8, 2, rank, 2, 73)), whole[4:]
            )


def test_decode_failure_is_not_swallowed():
    data = TinyWindows()
    data.row["observation.images.cam_high"][3] = {"bytes": b"invalid image"}
    with closing(decoded_batches(data, [[1], [3]], decode_workers=4, prefetch_batches=1)) as batches:
        next(batches)
        with pytest.raises(OSError):
            next(batches)
