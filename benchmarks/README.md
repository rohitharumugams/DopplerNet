# DopplerSim Initial Benchmark Suite (B1-B10)

This directory contains the unified generation and evaluation suite for the ten initial benchmarks described in the DopplerSim paper.

## Core Utility: `benchmark_suite.py`

The suite handles data generation, metadata extraction, and automated scoring for the following benchmarks:

### Initial Benchmarks (Implemented)
- **B1: Speed Estimation**: Predict velocity (mps).
- **B2: Direction-of-Travel**: Approach/Recede/Lateral classification.
- **B3: Distance-of-Closest-Approach**: Estimate CPA distance (m).
- **B4: Trajectory Shape**: Straight/Parabola/Bezier classification.
- **B5: Time-to-Event**: Predict seconds to CPA (CPA Time control).
- **B6: Motion State Segmentation**: Frame-level state labeling (`segmentation_mask.npy`).
- **B8: Multi-Object Disentanglement**: Resolve multiple concurrent sources (Num Sources control).
- **B9: Crossing/Interaction**: Detect intersection events (Crossing toggle).
- **B10: Source Identity**: Vehicle type recognition under motion.

*Note: B7 (Acceleration/Deceleration) is currently excluded from the selection suite.*

## Usage

### 1. Web Interface (Recommended)
The simulator now features a dedicated **Benchmark Mode** section in the Batch Generation tab.
- Select specific benchmarks (B1-B10).
- Configure parameters like `CPA Window` (B6) or `Number of Sources` (B8).
- The system automatically generates physically consistent samples and logs them to `dataset.csv`.

### 2. Manual Generation
To generate a verification batch using the CLI:
```bash
python benchmarks/benchmark_suite.py --generate --num_samples 5
```
**Testing Phase Adjustments:**
- **Audibility**: Source audio is normalized to 1.0 peak before processing to ensure all samples are audible (fixes silent/blank audio issues).
- **Proximity**: Vehicles are constrained to a ±100m CPA distance for clear Doppler shifts during manual checking.

### 2. Run Evaluation
To run the full B1-B10 evaluation suite:
```bash
python benchmarks/benchmark_suite.py
```

## Directory Structure: `test_output/`
- `dataset.csv`: Unified ground truth with `path_plot` and `spectrogram_plot` mappings.
- `dataset_summary.txt`: Statistical distribution of the generated batch.
- `test_batch/audio_clips/sample_XXXXXXX/`:
  - `(test_N_){metadata}.wav`: Generated Doppler audio (normalized).
  - `(test_N_){metadata}.png`: **Primary Trajectory Graph** (physical path).
  - `(test_N_){metadata}_spectrogram.png`: Frequency sweep (CQT/STFT).
  - `*.npy`: Raw feature arrays for model training.

The naming convention `(test_N_)` allows for quick manual reference while the metadata suffix provides instantaneous parameter lookup without opening the CSV.
