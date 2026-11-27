
import tonic
import numpy as np
from pathlib import Path

print("=" * 50)
print("DVS128 GESTURE DATASET")
print("=" * 50)

data_path = "./data/dvs_gesture"

Path(data_path).mkdir(parents=True, exist_ok=True)

try:
    print("\nLoading training dataset...")
    print("The first download may take some time.")

    dataset = tonic.datasets.DVSGesture(
        save_to=data_path,
        train=True
    )

    print("\nDataset loaded successfully!")

    print("Number of training samples:", len(dataset))

    # Inspect first sample
    events, label = dataset[0]

    print("\nFIRST SAMPLE DETAILS")
    print("-" * 30)

    print("Gesture label:", label)
    print("Number of events:", len(events))
    print("Event data type:", events.dtype)

    print("Event fields:", events.dtype.names)

    if len(events) > 0:

        print("\nFirst five events:")
        print(events[:5])

        print("\nSensor coordinate ranges:")

        print(
            "X:",
            np.min(events["x"]),
            "to",
            np.max(events["x"])
        )

        print(
            "Y:",
            np.min(events["y"]),
            "to",
            np.max(events["y"])
        )

        print(
            "Polarity:",
            np.unique(events["p"])
        )

        print(
            "Timestamp:",
            np.min(events["t"]),
            "to",
            np.max(events["t"])
        )

    print("\n" + "=" * 50)
    print("DVS128 DATASET CHECK SUCCESSFUL")
    print("=" * 50)

except Exception as error:

    print("\nDATASET LOADING FAILED")

    print("Error type:", type(error).__name__)
    print("Error message:", error)
