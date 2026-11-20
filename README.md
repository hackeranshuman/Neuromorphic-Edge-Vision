# Event-Driven Spiking Neural Networks (SNNs) for Neuromorphic Edge Vision

An experimental implementation of **Spiking Neural Networks (SNNs)** for event-driven visual processing, exploring biologically inspired computation and energy-efficient neural architectures for edge devices.

## Overview

Traditional Artificial Neural Networks (ANNs) process information using continuous numerical activations. In contrast, Spiking Neural Networks communicate through discrete spikes over time, making them particularly interesting for event-driven computation and neuromorphic hardware.

This project explores how Leaky Integrate-and-Fire (LIF) neurons and surrogate-gradient learning can be used to build a compact SNN for visual motion classification.

The initial prototype focuses on event generation, temporal information processing, neuron dynamics, and classification.

## Objectives

- Understand and implement biologically inspired LIF neurons.
- Generate event-based visual representations.
- Develop a compact SNN architecture using PyTorch and snnTorch.
- Train the network using surrogate-gradient learning.
- Evaluate classification performance and spike activity.
- Investigate the potential of SNNs for neuromorphic edge vision.

## Project Architecture

The planned processing pipeline:

```text
Visual Input
     |
     v
Event Generation
     |
     v
Event Representation
     |
     v
Input Layer
     |
     v
LIF Hidden Layer
     |
     v
Output Layer
     |
     v
Motion Classification
     |
     v
Performance Evaluation
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | Neural network framework |
| snnTorch | Spiking neural networks |
| NumPy | Numerical computations |
| Matplotlib | Visualization |
| Tonic | Neuromorphic datasets |
| Scikit-learn | Model evaluation |

## Neuron Model

The project uses the **Leaky Integrate-and-Fire (LIF)** neuron model.

The membrane potential evolves over time according to:

\[
U[t] = \beta U[t-1] + I[t]
\]

A neuron generates a spike when its membrane potential reaches the threshold:

\[
S[t] =
\begin{cases}
1 & U[t] \geq U_{threshold}\\
0 & \text{otherwise}
\end{cases}
\]

Where:

- \(U[t]\): Membrane potential at time \(t\).
- \(\beta\): Membrane decay factor.
- \(I[t]\): Input current.
- \(S[t]\): Output spike.

Surrogate gradients are used to enable gradient-based training despite the non-differentiable spike-generation function.

## Dataset

The project explores the DVS128 Gesture dataset, a neuromorphic vision dataset containing event-camera recordings of human gestures.

If real dataset loading is unavailable during initial development, a clearly labelled synthetic moving-shape event dataset will be used for prototyping and testing.

**Important:** Synthetic events are simulated visual events, not recordings from a physical event camera.

## Evaluation Metrics

The implementation aims to evaluate:

- Classification accuracy
- Confusion matrix
- Spike count and firing activity
- Membrane potential dynamics
- Inference time
- Computational activity

Energy efficiency and latency improvements will only be reported if supported by actual measurements and a suitable baseline.

## Project Structure

```text
SNN-Edge-Vision/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── results/
│   ├── plots/
│   └── metrics/
│
├── src/
│   ├── lif_neuron.py
│   ├── event_generator.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── check_environment.py
├── test_dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure represents the intended organization; files will be added as implementation progresses.*

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify the environment:

```bash
python check_environment.py
```

## Current Development Status

**Development stage: Initial prototype**

| Component | Status |
|---|---|
| Environment setup | In progress |
| Dataset integration | Pending verification |
| LIF neuron implementation | Initial implementation |
| Event generation | Planned |
| SNN architecture | Planned |
| Model training | Planned |
| Model evaluation | Planned |
| Performance benchmarking | Planned |

## Research Direction

The broader motivation is to investigate whether event-driven neural computation can provide useful advantages for visual processing under resource constraints.

This project is an educational and experimental exploration of neuromorphic computing. Claims regarding energy savings, hardware acceleration, and latency improvements require further implementation and benchmarking.

## Future Improvements

- Integration with real event-camera hardware.
- Exploration of different neuron models.
- Comparison between ANN and SNN architectures.
- Optimization for edge deployment.
- Investigation of neuromorphic hardware platforms.

## Author

**Anshuman Singh**

B.Tech Computer Science  
GLA University

**Area of Interest:** Machine Learning, Neuromorphic Computing, Spiking Neural Networks, and Edge AI.

---

*This project is under active development. Implementation details, experimental results, and evaluation metrics will be updated as development progresses.*
