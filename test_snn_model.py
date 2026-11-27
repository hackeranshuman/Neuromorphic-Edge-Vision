
import torch

from snn_model import MotionSNN

# Reproducibility
torch.manual_seed(42)

# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Create model
model = MotionSNN().to(device)

# Generate random test input
batch_size = 4
time_steps = 20
input_features = 512

x = torch.rand(
    batch_size,
    time_steps,
    input_features,
    device=device
) * 2.0

# Forward pass
spk_out, mem_out, spk_hidden = model(x)

print("\nForward pass successful")

print("Input shape:", tuple(x.shape))
print("Output spike shape:", tuple(spk_out.shape))
print("Output membrane shape:", tuple(mem_out.shape))
print("Hidden spike shape:", tuple(spk_hidden.shape))

# Count output spikes
output_spike_counts = spk_out.sum(dim=0)

print("\nOutput spike counts:")
print(output_spike_counts.detach().cpu())

# Test backpropagation
loss = mem_out[-1].pow(2).mean()

model.zero_grad()
loss.backward()

gradient = model.fc1.weight.grad

print("\nGradient test")
print("Loss:", loss.item())
print("Gradient exists:", gradient is not None)

if gradient is not None:
    print(
        "Gradient norm:",
        gradient.norm().item()
    )

    print(
        "Gradient is finite:",
        torch.isfinite(gradient).all().item()
    )

    print(
        "Gradient reaches first layer:",
        gradient.abs().sum().item() > 0
    )

print("\nSNN forward and backward test finished")
