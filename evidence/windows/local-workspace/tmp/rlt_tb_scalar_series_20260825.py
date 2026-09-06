import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


event = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0})
event.Reload()
for tag in sys.argv[2:]:
    for item in event.Scalars(tag):
        print(f"{tag}\t{item.step}\t{float(item.value):.9g}")
