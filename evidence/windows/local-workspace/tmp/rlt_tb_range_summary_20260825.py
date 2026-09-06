import statistics
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


path = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
tags = sys.argv[4:]
event = EventAccumulator(path, size_guidance={"scalars": 0})
event.Reload()
available = set(event.Tags().get("scalars", []))
for tag in tags:
    if tag not in available:
        print(f"{tag}\tMISSING")
        continue
    selected = [item for item in event.Scalars(tag) if start <= item.step <= end]
    if not selected:
        print(f"{tag}\tEMPTY")
        continue
    values = [float(item.value) for item in selected]
    print(
        "\t".join(
            (
                tag,
                str(len(values)),
                str(selected[0].step),
                str(selected[-1].step),
                f"{statistics.fmean(values):.9g}",
                f"{statistics.median(values):.9g}",
                f"{min(values):.9g}",
                f"{max(values):.9g}",
                f"{values[-1]:.9g}",
            )
        )
    )
