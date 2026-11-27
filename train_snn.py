
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from snn_model import MotionSNN


# ----------------------------------
# 1. Reproducibility
# ----------------------------------

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ----------------------------------
# 2. Generate simulated event data
# ----------------------------------

def generate_sample(label, rng):

    time_steps = 20
    height = 16
    width = 16
    size = 3

    # Choose motion direction
    if label == 0:
        # Horizontal
        dx = int(rng.choice([-1, 1]))
        dy = 0

    elif label == 1:
        # Vertical
        dx = 0
        dy = int(rng.choice([-1, 1]))

    else:
        # Diagonal
        dx = int(rng.choice([-1, 1]))
        dy = int(rng.choice([-1, 1]))

    # Choose valid starting position
    if dx == 1:
        x0 = int(rng.integers(1, 8))
    elif dx == -1:
        x0 = int(rng.integers(7, 14))
    else:
        x0 = int(rng.integers(1, 14))

    if dy == 1:
        y0 = int(rng.integers(1, 8))
    elif dy == -1:
        y0 = int(rng.integers(7, 14))
    else:
        y0 = int(rng.integers(1, 14))

    events = np.zeros(
        (time_steps, 2, height, width),
        dtype=np.float32
    )

    previous_frame = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for t in range(time_steps):

        # Move one pixel every three time steps
        movement = t // 3

        x = x0 + movement * dx
        y = y0 + movement * dy

        frame = np.zeros(
            (height, width),
            dtype=np.float32
        )

        # Draw moving square
        frame[
            y:y + size,
            x:x + size
        ] = 1.0

        # Detect brightness increases and decreases
        on_events = (
            frame > previous_frame
        ).astype(np.float32)

        off_events = (
            previous_frame > frame
        ).astype(np.float32)

        events[t, 0] = on_events
        events[t, 1] = off_events

        previous_frame = frame

    # Flatten into 512 features per time step
    return events.reshape(time_steps, -1)


# ----------------------------------
# 3. Build complete dataset
# ----------------------------------

rng = np.random.default_rng(SEED)

samples = []
labels = []

class_names = [
    "Horizontal",
    "Vertical",
    "Diagonal"
]

for label in range(3):

    for _ in range(100):

        sample = generate_sample(label, rng)

        samples.append(sample)
        labels.append(label)

X = np.array(samples, dtype=np.float32)
y = np.array(labels, dtype=np.int64)

print("\nDataset generated")
print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

print("Samples per class:")

for label, name in enumerate(class_names):
    print(name, ":", np.sum(y == label))


# ----------------------------------
# 4. Split dataset
# ----------------------------------

indices = np.arange(len(y))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Save the exact dataset split for Day 5
np.savez_compressed(
    "data/synthetic_motion_events.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    class_names=np.array(class_names)
)


# ----------------------------------
# 5. Create PyTorch DataLoader
# ----------------------------------

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


# ----------------------------------
# 6. Initialize SNN
# ----------------------------------

model = MotionSNN(
    input_size=512,
    hidden_size=128,
    num_classes=3
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ----------------------------------
# 7. Train network
# ----------------------------------

epochs = 20

loss_history = []

print("\nStarting SNN training...\n")

for epoch in range(epochs):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        spk_out, mem_out, hidden_spikes = model(batch_x)

        # Sum output membrane activity over time
        logits = mem_out.sum(dim=0)

        # Calculate loss
        loss = criterion(logits, batch_y)

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        # Prevent excessively large gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        # Track metrics
        total_loss += loss.item() * batch_y.size(0)

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.size(0)

    average_loss = total_loss / total
    train_accuracy = 100 * correct / total

    loss_history.append(average_loss)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Training Accuracy: {train_accuracy:.2f}%"
    )


# ----------------------------------
# 8. Save trained model
# ----------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": 512,
        "hidden_size": 128,
        "num_classes": 3,
        "num_steps": 20,
        "class_names": class_names,
        "seed": SEED
    },
    "models/snn_model.pth"
)


# ----------------------------------
# 9. Plot training loss
# ----------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("SNN Training Loss")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/training_loss.png",
    dpi=150
)

plt.show()


# ----------------------------------
# 10. Final summary
# ----------------------------------

print("\nTraining finished.")

print("Final training loss:", loss_history[-1])

print("Model saved:")
print("models/snn_model.pth")

print("Dataset saved:")
print("data/synthetic_motion_events.npz")

print("Loss graph saved:")
print("results/training_loss.png")
