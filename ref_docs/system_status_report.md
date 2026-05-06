# DopplerSim System Status Report
**Date:** May 1, 2026
**Status:** Operational - Specialized Dataset Generation Active

## Executive Summary
DopplerSim has been optimized for high-fidelity dataset generation targeting the VS13 vehicle speed estimation task. The system currently supports two primary generation modes designed to bridge the gap between simulation and real-world acoustic data. All generated datasets are exported with standardized annotations and deterministic splits for model training and validation.

## Core Generation Modes

### 1. VS13-Compatible Generation
This mode generates pure synthetic audio clips using the DopplerSim physics engine, strictly adhering to the VS13 dataset standards.
- **Goal**: Create large-scale, clean synthetic datasets with precise kinematic ground truth.
- **Output Path**: `D:\Antigravity\vs13-model\ExtendedSimulatedData`
- **Features**: 
    - Generates 10-second uniform audio clips.
    - Labels include speed (km/h) and Time of Closest Approach (TCA) in seconds.
    - Uses high-precision internal conversions for simulation fidelity.

### 2. Real-Simulated Mixed Audio Generation
This mode aligns simulated audio generation with real-world metadata extracted from the VS13 dataset, allowing for "matched" pairs of real and synthetic data.
- **Goal**: Facilitate domain adaptation and performance benchmarking against real-world captures.
- **Output Path**: `D:\Antigravity\vs13-model\SimulatedData`
- **Features**: 
    - Uses VS13 metadata as the "Source of Truth" for speed and vehicle types.
    - Exports to a flat, vehicle-specific hierarchy.
    - Implements deterministic 80/20 train/valid splits per vehicle.

## Generated Datasets Overview

The following datasets have been generated and are stored in their respective directories:

### Vehicle Categories
Data has been generated for the following 6 vehicle types, consistent across both modes:
- **Kia Sportage**
- **Nissan Qashqai**
- **Peugeot 3008**
- **Peugeot 307**
- **Renault Scenic**
- **VW Passat**

### Data Structure & Content
Each vehicle folder contains:
- **Audio Files (`.wav`)**: 10-second clips sampled at 22,050 Hz (mono/stereo depending on config).
- **Annotation Files (`.txt`)**: Simple text files containing two values: `[Speed_kmh] [TCA_s]`.
    - *Example*: `100.00 5.00` (100 km/h at 5 seconds).
- **Split Configuration**: `Train_valid_split.txt` which defines the exact files assigned to training and validation sets to ensure repeatable experiments.

## Summary of Storage Paths

| Dataset Type | Physical Path | Description |
| :--- | :--- | :--- |
| **Extended Simulated** | `D:\Antigravity\vs13-model\ExtendedSimulatedData` | Full synthetic suite for baseline training. |
| **Simulated** | `D:\Antigravity\vs13-model\SimulatedData` | Real-world aligned clips for validation/matching. |

---
