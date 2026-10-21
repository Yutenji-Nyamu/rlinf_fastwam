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

"""Full H50 windows from the already accepted, canonical Clean50 parquet data."""

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


class DemoWindows:
    """Read each small episode once; retain compressed real RGB, not one-file churn."""

    def __init__(self, root: str, horizon: int = 50):
        self.root, self.horizon = Path(root), horizon
        self.prompts = {
            int(r["task_index"]): r["task"]
            for r in (
                json.loads(s)
                for s in (self.root / "meta/tasks.jsonl").read_text().splitlines()
                if s
            )
        }
        self.episodes, self.windows = [], []
        for path in sorted(self.root.glob("data/*/episode_*.parquet")):
            table = pq.read_table(path)
            row = {
                k: table[k].to_pylist()
                for k in (
                    "observation.state",
                    "action",
                    "task_index",
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                )
            }
            row["path"] = str(path)
            row["actions"] = np.asarray(row.pop("action"), dtype=np.float32)
            row["states"] = np.asarray(row.pop("observation.state"), dtype=np.float32)
            if row["actions"].shape != (table.num_rows, 14):
                raise ValueError(f"Expected absolute 14D Aloha actions: {path}")
            ep = len(self.episodes)
            self.episodes.append(row)
            for start in range(max(0, table.num_rows - horizon + 1)):
                if len(set(row["task_index"][start : start + horizon])) != 1:
                    raise ValueError("H50 window crosses an instruction boundary")
                self.windows.append((ep, start))
        if len(self.episodes) != 50 or not self.windows:
            raise ValueError(
                f"Expected complete Clean50; got {len(self.episodes)} episodes"
            )

    def __len__(self):
        return len(self.windows)

    def batch(self, indices):
        """Decode requested real RGB frames, with no temporal/frame substitution."""
        states, actions, prompts, views = [], [], [], []
        keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        for i in indices:
            ep, start = self.windows[int(i)]
            row = self.episodes[ep]
            imgs = []
            for key in keys:
                item = row[f"observation.images.{key}"][start]
                with Image.open(io.BytesIO(item["bytes"])) as image:
                    imgs.append(np.asarray(image.convert("RGB")))
            views.append(np.stack(imgs))
            states.append(row["states"][start])
            actions.append(row["actions"][start : start + self.horizon])
            prompts.append(self.prompts[int(row["task_index"][start])])
        rgb = torch.from_numpy(np.stack(views))
        return {
            "main_images": rgb[:, 0],
            "wrist_images": rgb[:, 1:],
            "states": torch.from_numpy(np.stack(states)),
            "task_descriptions": prompts,
        }, torch.from_numpy(np.stack(actions))
