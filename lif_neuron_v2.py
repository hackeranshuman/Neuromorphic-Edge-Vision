
import numpy as np
import matplotlib.pyplot as plt

# 1. Neuron parameters
beta = 0.9
threshold = 1.0
time_steps = 100

# 2. Input current
current = np.zeros(time_steps)
current[10:40] = 0.35

# 3. Initialize neuron variables
membrane_potential = 0.0

voltage_before_reset = np.zeros(time_steps)
voltage_after_reset = np.zeros(time_steps)
spikes = np.zeros(time_steps)

# 4. LIF simulation
for t in range(time_steps):

    # Integration and leakage
    new_potential = (
        beta * membrane_potential + current[t]
    )

    # Record potential before reset
    voltage_before_reset[t] = new_potential

    # Threshold check
    if new_potential >= threshold:

        spikes[t] = 1

        # Hard reset
        membrane_potential = 0.0

    else:
        membrane_potential = new_potential

    # Record actual internal potential
    voltage_after_reset[t] = membrane_potential


# 5. Calculate results
spike_times = np.where(spikes == 1)[0]

print("Total spikes:", len(spike_times))
print("Spike times:", spike_times.tolist())

# 6. Visualize results
fig, ax = plt.subplots(4, 1, figsize=(12, 12))

ax[0].plot(current)
ax[0].set_title("Input Current")
ax[0].set_ylabel("Current")
ax[0].grid()

ax[1].plot(voltage_before_reset, label="Before reset")
ax[1].axhline(
    threshold,
    linestyle="--",
    label="Threshold"
)
ax[1].set_title("Membrane Potential Before Reset")
ax[1].set_ylabel("Voltage")
ax[1].legend()
ax[1].grid()

ax[2].step(
    range(time_steps),
    voltage_after_reset,
    where="post"
)
ax[2].set_title("Internal Membrane Potential After Reset")
ax[2].set_ylabel("Voltage")
ax[2].grid()

ax[3].stem(
    np.arange(time_steps),
    spikes
)
ax[3].set_title("Spike Train")
ax[3].set_xlabel("Time Steps")
ax[3].set_ylabel("Spike")
ax[3].grid()

plt.tight_layout()

plt.savefig("lif_results.png", dpi=150)
plt.show()

# 7. Verify expected result
expected_spikes = [13, 17, 21, 25, 29, 33, 37]

assert spike_times.tolist() == expected_spikes

print("LIF simulation verification passed!")
