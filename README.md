# Doppler Effect Simulator (DopplerNet)

A Flask-based system for generating realistic Doppler-shifted vehicle audio clips for research purposes, dataset creation, and acoustic modeling experimentation. Supports batch, overlapping, and single-clip generation using straight-line, parabolic, and Bezier paths with physically accurate Doppler and distance attenuation.

---

## Features

### Core Capabilities
- **Realistic Doppler Shift**: Simulation using acoustic wave physics with sample-level resampling at SR = 44,100 Hz.
- **Multiple Vehicle Trajectories**:
  - **Straight Line**: Standard pass-by with configurable closest point of approach (CPA).
  - **Parabolic**: Curved path simulation.
  - **Bezier Curve**: Complex multi-point cubic trajectories.
- **Batch Overlap (Busy Road)**: Simulate multiple vehicles with staggered starts and lane offsets to create complex acoustic scenes.
- **Drone Support**: specialized support for drone sound libraries and flight dynamics.
- **Adaptive Physics**: Physically correct 1/R spherical spreading for distance-based amplitude shaping.
- **Automated Visualizations**: Generates path plots and spectrograms for every audio clip generated.

### UI Functionality
- **Multi-Mode Web Interface**:
  - **Batch Generation**: Large scale dataset creation with randomized or user-defined parameters.
  - **Batch Overlap**: Complex scene generation for multi-target tracking research.
  - **Spectrograms**: dedicated tool for analyzing sound files and generating visual high-resolution spectrograms.
  - **Single Clip**: Instant preview mode for parameter tuning.
- **Vehicle Management**: Library upload, validation (3.0s duration check), and live management.
- **Real-time Plotting**: Live preview of vehicle paths on a canvas.

---

## Project Structure

```
DopplerNet/
├── static/
│   ├── batch_outputs/       # Organized batch results (Audio + Metadata + Plots)
│   ├── single_outputs/      # Single-clip preview results
│   ├── vehicle_sounds/      # User-uploaded car/vehicle sound files
│   ├── drone_sounds/        # specialized drone sound library
│   └── spectrograms/        # Transient spectrogram outputs
├── templates/
│   └── index_batch.html     # unified Web UI
├── app_batch.py             # Main Flask backend + API endpoints
├── audio_utils.py           # Doppler, amplitude, and resampling utilities
├── graphs.py                # shared plotting logic for paths and spectrograms
├── straight_line.py         # Straight line Doppler model logic
├── parabola.py              # Parabolic path logic
├── bezier.py                # Bezier curve logic
├── sampler_state.json       # Persistent state for cyclic parameter sampling
├── generation_progress.json # Live progress tracking for long batches
└── requirements.txt         # Project dependencies
```

---

## Installation

### Prerequisites
- Python 3.9+
- pip and virtual environment support

### Setup

```bash
# Clone the repository
git clone https://github.com/rohitharumugams/doppler-batch-generation.git
cd DopplerNet

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
python3 app_batch.py
```
The server will start at: `http://localhost:5050`

---

## Operational Modes

### 1. Batch Generation
Used to create large-scale datasets for machine learning.
1. Select path types (Multiple can be selected).
2. Choose sound source (Cars/Drones).
3. Set ranges for speed, distance, and angle.
4. Define total clips and distribution mode (Automatic or Manual).
5. **Output**: Folders containing WAVs, metadata.json, path plots, and spectrograms.

### 2. Batch Overlap (Busy Road Simulation)
Simulates realistic environments with multiple vehicles.
1. Define number of scenes.
2. Set range of vehicles per scene.
3. Configure lane width and maximum stagger (delay between vehicle starts).
4. **Output**: A "mixed_audio.wav" per scene along with individual vehicle tracks and a combined path plot.

### 3. Spectrogram Analyze
Analysis tool for sound libraries.
- Upload any audio file to generate a high-quality spectrogram.
- View and analyze the frequency distribution of vehicle sounds before generation.

### 4. Single Clip
Instant simulator for testing specific parameters.
- Control every aspect of a single vehicle's path.
- Play and download results immediately.

---

## Physics Model

The Doppler shift formula applied per-sample:
```
f_obs(t) = f_src * (c / (c - v_rel(t)))
```
where:
- `c` = 343 m/s (speed of sound)
- `v_rel(t)` = radial velocity component relative to the observer

Distance-based attenuation (1/R law):
```
A(t) = 1 / max(distance(t), 1)
```

---

## Authors
**Rohith Arumugam Suresh** & **Seetharam Killivalavan**  
Computer Science and Engineering  
Sri Sivasubramaniya Nadar College of Engineering  
*Research Interns, Carnegie Mellon University*

## Acknowledgments
Carnegie Mellon University, Language Technologies Institute  
Professor Bhiksha Raj and Bradley Warren for research guidance.
