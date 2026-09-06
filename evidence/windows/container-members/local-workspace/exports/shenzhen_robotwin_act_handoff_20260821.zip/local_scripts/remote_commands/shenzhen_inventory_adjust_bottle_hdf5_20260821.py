from pathlib import Path
import re

import h5py
import numpy as np

from XPolicyLab.utils.process_data import decode_image_bit


ROOT = Path("/data/chenyiteng/projects/robotwin-native/RoboTwin/data/demo_clean/adjust_bottle/aloha_agilex")
DATA_DIR = ROOT / "data"
EXPECTED_EPISODES = 50
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
FIELDS = {
    "state/left_arm_joint_states": 6,
    "state/left_ee_joint_states": 1,
    "state/right_arm_joint_states": 6,
    "state/right_ee_joint_states": 1,
    "action/left_arm_joint_states": 6,
    "action/left_ee_joint_states": 1,
    "action/right_arm_joint_states": 6,
    "action/right_ee_joint_states": 1,
}
PATTERN = re.compile(r"episode_(\d{7})\.hdf5$")

assert DATA_DIR.is_dir(), f"missing data directory: {DATA_DIR}"
all_hdf5 = sorted(DATA_DIR.glob("*.hdf5"))
bad_names = [path.name for path in all_hdf5 if PATTERN.fullmatch(path.name) is None]
assert not bad_names, f"non-canonical episode names: {bad_names}"

indexed = sorted((int(PATTERN.fullmatch(path.name).group(1)), path) for path in all_hdf5)
ids = [episode_id for episode_id, _ in indexed]
assert ids == list(range(len(ids))), f"episode IDs are not continuous from zero: {ids}"
assert len(indexed) == EXPECTED_EPISODES, f"expected 50 episodes, found {len(indexed)}"

source_bytes = sum(path.stat().st_size for _, path in indexed)
total_frames = 0
largest_episode_frames = 0
first_camera_shapes = None

for episode_id, path in indexed:
    with h5py.File(path, "r") as h5:
        if episode_id == 0:
            print("=== episode_0000000 tree ===")

            def show(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"{name}/")

            h5.visititems(show)
            print(f"file attrs: {dict(h5.attrs)}")

        for key, last_dim in FIELDS.items():
            assert key in h5, f"{path.name}: missing {key}"
            ds = h5[key]
            assert ds.ndim == 2 and ds.shape[-1] == last_dim, f"{path.name}: bad {key} {ds.shape}"
            assert np.issubdtype(ds.dtype, np.number), f"{path.name}: nonnumeric {key} {ds.dtype}"

        frames = h5["state/left_arm_joint_states"].shape[0]
        assert frames > 0, f"{path.name}: zero frames"
        for key, last_dim in FIELDS.items():
            assert h5[key].shape == (frames, last_dim), f"{path.name}: shape mismatch {key} {h5[key].shape}"

        decoded_shapes = {}
        for camera in CAMERAS:
            key = f"vision/{camera}/colors"
            assert key in h5, f"{path.name}: missing {key}"
            ds = h5[key]
            assert ds.ndim >= 1 and ds.shape[0] == frames, f"{path.name}: frame mismatch {key} {ds.shape}"
            image = np.asarray(decode_image_bit(ds[0]))
            assert image.ndim == 3 and image.shape[-1] == 3, f"{path.name}: bad decoded {key} {image.shape}"
            decoded_shapes[camera] = {
                "stored_shape": tuple(ds.shape),
                "stored_dtype": str(ds.dtype),
                "decoded_first_frame": tuple(image.shape),
                "decoded_dtype": str(image.dtype),
            }

        if first_camera_shapes is None:
            first_camera_shapes = decoded_shapes
        total_frames += frames
        largest_episode_frames = max(largest_episode_frames, frames)
        print(f"{path.name}: frames={frames}")

image_bytes_per_frame = 3 * 640 * 480 * 3
vector_bytes_per_frame = 2 * 14 * np.dtype(np.float32).itemsize
act_payload_bytes = total_frames * (image_bytes_per_frame + vector_bytes_per_frame)
largest_episode_payload = largest_episode_frames * (image_bytes_per_frame + vector_bytes_per_frame)

print("=== summary ===")
print(f"root: {ROOT}")
print(f"episodes: {len(indexed)}, continuous IDs: 0..{len(indexed) - 1}")
print(f"total_frames: {total_frames}")
print(f"largest_episode_frames: {largest_episode_frames}")
print("packed_state_action_dim: 14 / 14")
print(f"source_hdf5_gib: {source_bytes / 2**30:.3f}")
print(f"first_episode_cameras: {first_camera_shapes}")
print(f"act_uncompressed_payload_lower_bound_gib: {act_payload_bytes / 2**30:.3f}")
print(f"largest_episode_payload_lower_bound_gib: {largest_episode_payload / 2**30:.3f}")
print("estimate excludes HDF5 metadata and process-time duplication")
