import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


event = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
event.Reload()
for tag in event.Tags().get("scalars", []):
    if any(
        needle in tag
        for needle in (
            "rlt_dvac",
            "success",
            "grad_norm",
            "actor_loss",
            "critic_loss",
            "weighted_q",
            "weighted_bc",
            "bc_loss",
            "q_pi",
            "action_ref",
            "actor_weight",
            "replay_age",
        )
    ):
        values = event.Scalars(tag)
        if values:
            print(f"{tag}\t{len(values)}\t{values[-1].step}\t{values[-1].value:.9g}")
