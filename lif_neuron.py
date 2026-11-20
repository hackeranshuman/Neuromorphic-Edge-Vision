
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# LIF neuron parameters
# -----------------------------

beta = 0.9
threshold = 1.0
time_steps = 100

# Input current
current = np.zeros(time_steps)
current[10:40] = 0.35

# Neuron variables
membrane_potential = 0.0

voltages = np.zeros(time_steps)
spikes = np.zeros(time_steps)

# -----------------------------
# LIF simulation
# -----------------------------

for t in range(time_steps):

    # Integrate input and apply leak
    new_potential = (
        beta * membrane_potential + current[t]
    )

    # Check threshold
    if new_potential >= threshold:

        spikes[t] = 1

        # Hard reset
        membrane_potential = 0.0

    else:
        membrane_potential = new_potential

    voltages[t] = new_potential

# -----------------------------
# Visualization
# -----------------------------

fig, ax = plt.subplots(
    3, 1,
    figsize=(12, 9),
    sharex=True
)

# Input current
ax[0].plot(current)
ax[0].set_title("Input Current")
ax[0].set_ylabel("Current")

# Membrane potential before reset
ax[1].plot(voltages, label="Membrane Potential")
ax[1].axhline(
    y=threshold,
    linestyle="--",
    label="Threshold"
)
ax[1].set_title("LIF Membrane Potential")
ax[1].set_ylabel("Voltage")
ax[1].legend()

# Spike train
ax[2].stem(
    np.arange(time_steps),
    spikes
)
ax[2].set_title("Generated Spikes")
ax[2].set_xlabel("Time Steps")
ax[2].set_ylabel("Spike")

plt.tight_layout()
plt.show()

print("Total spikes:", int(np.sum(spikes)))
print("Beta:", beta)
print("Threshold:", threshold)
