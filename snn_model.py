
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class MotionSNN(nn.Module):

    def __init__(
        self,
        input_size=512,
        hidden_size=128,
        num_classes=3
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Surrogate gradient function
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Input to hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Hidden LIF neurons
        self.lif1 = snn.Leaky(
            beta=0.9,
            threshold=1.0,
            spike_grad=spike_grad,
            reset_mechanism="zero"
        )

        # Hidden to output layer
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # Output LIF neurons
        self.lif2 = snn.Leaky(
            beta=0.9,
            threshold=1.0,
            spike_grad=spike_grad,
            reset_mechanism="zero"
        )

    def forward(self, x):

        # Expected input: [batch, time, features]
        if x.ndim != 3:
            raise ValueError(
                "Input must have shape [batch, time, features]"
            )

        batch_size, num_steps, num_features = x.shape

        if num_features != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"received {num_features}"
            )

        # Initialize membrane potentials
        mem1 = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype
        )

        mem2 = torch.zeros(
            batch_size,
            self.num_classes,
            device=x.device,
            dtype=x.dtype
        )

        # Store outputs over time
        spk_out_rec = []
        mem_out_rec = []
        spk_hidden_rec = []

        # Process one time step at a time
        for t in range(num_steps):

            # First fully connected layer
            current1 = self.fc1(x[:, t, :])

            # Hidden LIF layer
            spk1, mem1 = self.lif1(current1, mem1)

            # Second fully connected layer
            current2 = self.fc2(spk1)

            # Output LIF layer
            spk2, mem2 = self.lif2(current2, mem2)

            # Record activity
            spk_hidden_rec.append(spk1)
            spk_out_rec.append(spk2)
            mem_out_rec.append(mem2)

        return (
            torch.stack(spk_out_rec),
            torch.stack(mem_out_rec),
            torch.stack(spk_hidden_rec)
        )
