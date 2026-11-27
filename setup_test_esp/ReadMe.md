# Phase 1: Pipeline Mock Validation

**Project:** Event-Driven SNN for Neuromorphic Edge Vision  
**Hardware:** ESP32-WROOM-32  
**Status:** Completed

## Overview
Established an end-to-end host-to-edge inference pipeline using synthetic event data to validate serial communication and embedded LIF neuron execution.

## Implementation

- **Synthetic Input:** Generated 256 random binary events representing a 16×16 input grid.
- **Data Encoding:** Packed binary events into 32-byte serial packets.
- **Weight Generation:** Generated 256 random floating-point weights (`-0.5 to +0.5`) in `weights.h`.
- **Serial Communication:** Implemented bidirectional Python–ESP32 communication using PySerial.
- **Edge Inference:** Developed an ESP32-based C++ pipeline for bit unpacking, weighted accumulation and LIF neuron processing.
- **Input Rate:** Approximately 2 FPS.

## Validation

- Successful ESP32 firmware compilation and deployment.
- Serial packet reception and binary event decoding.
- Weight integration and LIF neuron execution.
- Runtime monitoring through serial diagnostics.
- No memory-related crashes observed during testing.

## Design Decisions

- Synthetic data for initial pipeline validation.
- Compact bit-packed event representation.
- Random weights to simulate synaptic connections.
- Local neuron computation on ESP32.
- Serial diagnostics for runtime verification.

## Outcome

Successfully validated the Phase 1 mock inference pipeline, establishing the foundation for real-time vision input and trained SNN integration.

**Next Phase:** Webcam-based event generation and integration of trained SNN parameters.
