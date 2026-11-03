"""Read native object poses in each environment's local frame after settling."""

import numpy as np


def audit_objects(env, n, group, *, raise_on_mismatch=True):
    manager = env.scene_manager.layout_manager
    poses = []

    def array(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float64).reshape(-1)

    for i in range(n):
        row = {}
        counts = {}
        for name, kind in manager.instance_type_by_env[i].items():
            if kind not in ("rigid", "articulation"):
                continue
            p, q = manager.get_instance_pose(env_idx=i, inst_name=name, relative=True)
            # Native _generate_object_paths appends a global unique counter.
            # Match layout load order within each asset, not that runtime ID.
            asset = (kind, *manager.decode_inst_name(name))
            index = counts.get(asset, 0)
            counts[asset] = index + 1
            row[(*asset, index)] = (array(p), array(q))
        poses.append(row)
    position_error = 0.0
    rotation_error = 0.0
    bad_slots = []
    for start in range(0, n, group):
        reference = poses[start]
        group_position = group_rotation = 0.0
        for row in poses[start : start + group]:
            if set(row) != set(reference):
                raise RuntimeError("GRPO group has different object instances")
            for name, (p, q) in row.items():
                p0, q0 = reference[name]
                group_position = max(group_position, float(np.abs(p - p0).max()))
                # q and -q encode the same rotation.
                cosine = min(
                    1.0, float(abs(q @ q0) / (np.linalg.norm(q) * np.linalg.norm(q0)))
                )
                group_rotation = max(
                    group_rotation, float(2 * np.arccos(cosine) * 180 / np.pi)
                )
        position_error = max(position_error, group_position)
        rotation_error = max(rotation_error, group_rotation)
        if group_position > 0.005 or group_rotation > 2:
            bad_slots.extend(range(start, start + group))
    result = {
        "object_count": len(poses[0]),
        "position_max_m": position_error,
        "rotation_max_deg": rotation_error,
        "bad_slots": bad_slots,
    }
    # Numerical settling tolerance, much tighter than native instability checks
    # (4 cm / 30 degrees). Refuse substantially different group initial states.
    if bad_slots and raise_on_mismatch:
        raise RuntimeError(f"GRPO group object state mismatch: {result}")
    return result
