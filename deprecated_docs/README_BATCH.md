# Doppler Effect Batch Simulator

Professional-grade automated Doppler effect audio generation system for machine learning datasets.

## Features

###  Vehicle Sound Management
- Upload custom vehicle audio clips (3 ± 0.5 seconds)
- Persistent storage in `/static/vehicle_sounds/`
- Support for WAV, MP3, OGG, FLAC formats
- Preview and delete functionality

###  Advanced Randomization
- **Master "Randomize All" toggle** - One-click full randomization
- **Individual parameter randomization** - Fine-grained control
- **Hardcoded sensible ranges** for realistic physics
- **Manual override** - Specify exact parameter ranges

###  Motion Path Types
1. **Straight Line** - Linear motion with configurable angle
2. **Parabolic** - Curved trajectory with adjustable curvature
3. **Bezier Curve** - Complex paths with 4 control points

###  Configurable Parameters

#### Always Configurable
- **Speed** (m/s): Vehicle velocity
  - Default ranges: Car (15-50), Train (20-55), Drone (5-30)
- **Distance** (m): Perpendicular distance from observer (5-100m)
- **Duration** (s): Audio clip length (3-8s)
- **Angle** (degrees): Path angle for straight line motion (-45° to 45°)

#### Path-Specific
- **Parabola**: Curvature coefficient (a), Height offset (h)
- **Bezier**: 4 control points (P0, P1, P2, P3) with x,y coordinates

###  Batch Generation Modes

#### Auto-Split Mode
- Specify total clips
- Automatically distributes evenly across vehicles and paths
- Handles remainders intelligently

#### Manual Distribution Mode
- Precise control over clip distribution
- Per-vehicle count specification
- Per-path count specification
- Real-time validation (sums must equal total)

###  Output Options
- **WAV** (Recommended) - Lossless, ideal for ML training
- **MP3** - Compressed, smaller file size
- Configurable sample rate and bitrate
- Custom output folder path

###  Comprehensive Logging

Each batch generates:
1. **metadata.json** - Complete parameter record for every clip
2. **generation_log.txt** - Success/failure status for each clip
3. **statistics.txt** - Aggregate statistics and distributions

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app_batch.py
```

## Usage

### 1. Upload Vehicle Sounds
```
- Click "Upload Vehicle Sound" area
- Select audio file (3 ± 0.5 seconds)
- Enter vehicle name (e.g., "car", "train", "drone")
- Upload
```

### 2. Configure Parameters

#### Quick Start - Full Randomization
```
1. Toggle "RANDOMIZE ALL" ON
2. Set total clips (e.g., 100)
3. Select output format (WAV recommended)
4. Click "Generate Batch"
```

#### Advanced - Custom Configuration
```
1. Toggle individual parameters as needed
2. For manual parameters, set min/max ranges
3. Choose distribution mode:
   - Auto: Equal distribution
   - Manual: Specify exact counts
4. Generate batch
```

### 3. Review Output

Batch output structure:
```
static/batch_outputs/
 batch_20241121_153045/
     audio_clips/
    �    car_straight_25mps_15m_20241121_0001.wav
    �    train_parabola_40mps_30m_20241121_0002.wav
    �    ...
     metadata.json
     generation_log.txt
     statistics.txt
```

## Output File Naming Convention

```
{vehicle}_{path}_{speed}mps_{distance}m_{batch_id}_{index}.{format}

Examples:
- car_straight_25mps_15m_20241121_0001.wav
- drone_bezier_18mps_45m_20241121_0002.wav
- train_parabola_35mps_22m_20241121_0003.wav
```

## Metadata Structure

```json
{
  "batch_id": "20241121_153045",
  "generation_time": "2024-11-21T15:30:45",
  "total_clips_generated": 100,
  "configuration": {
    "output": {"format": "wav"},
    "batch": {"total_clips": 100, "mode": "auto"}
  },
  "clips": [
    {
      "filename": "car_straight_25mps_15m_0001.wav",
      "vehicle": "car",
      "path_type": "straight",
      "parameters": {
        "speed": 25.3,
        "distance": 15.2,
        "duration": 5.1,
        "angle": 12.5
      },
      "freq_ratio_range": {"min": 0.85, "max": 1.15}
    }
  ]
}
```

## Statistics Summary

```
Vehicle Distribution:
  car: 35 clips (35.0%)
  train: 33 clips (33.0%)
  drone: 32 clips (32.0%)

Path Type Distribution:
  straight: 34 clips (34.0%)
  parabola: 33 clips (33.0%)
  bezier: 33 clips (33.0%)

Speed Statistics:
  Min: 10.2 m/s
  Max: 49.8 m/s
  Mean: 30.5 m/s
  Median: 31.2 m/s
```

## Default Parameter Ranges

### Speed (m/s)
- Car: 15-50 (54-180 km/h)
- Train: 20-55 (72-198 km/h)
- Drone: 5-30 (18-108 km/h)
- Motorcycle: 10-45 (36-162 km/h)

### Other Parameters
- Distance: 5-100 m
- Duration: 3-8 seconds
- Angle: -45° to 45°
- Bezier coordinates: -150 to 150 m
- Parabola curvature: -0.1 to 0.1
- Parabola height: 10-50 m

## Validation & Error Handling

The system automatically validates:
-  Audio file format and duration
-  Parameter ranges (positive values, physical limits)
-  Distribution sums (must equal total in manual mode)
-  Disk space availability
-  Vehicle sound availability

## Performance

### Tested Configurations
- **Small batch**: 10-50 clips (~30 seconds)
- **Medium batch**: 100-500 clips (~2-5 minutes)
- **Large batch**: 1000-5000 clips (~20-30 minutes)
- **Maximum**: 10,000 clips per batch

### Recommendations for Supercomputer Use
- Use WAV format for training data
- Enable parallel processing (modify backend)
- Use SSD storage for faster I/O
- Monitor disk space (5-10 MB per clip)

## API Endpoints

```
POST /api/upload_vehicle      - Upload vehicle sound
GET  /api/list_vehicles        - List uploaded vehicles
DELETE /api/delete_vehicle/:id - Delete vehicle sound
POST /api/batch_generate       - Generate batch
```

## Troubleshooting

### Vehicle Upload Fails
- Check audio duration (must be 3 ± 0.5 seconds)
- Verify file format (WAV, MP3, OGG, FLAC)
- Ensure unique vehicle name

### Distribution Validation Error
- In manual mode, sums must equal total clips
- Check both vehicle AND path distributions

### Generation Fails
- Ensure at least one vehicle is uploaded
- Check parameter ranges are valid (min < max)
- Verify sufficient disk space

## Technical Details

### Physics Model
- Accurate Doppler shift calculation
- Sound speed: 343 m/s at 20°C
- Frequency ratio: f'/f = c/(c - vr)
- Inverse distance amplitude law

### Audio Processing
- Sample rate: 22,050 Hz
- Bit depth: 16-bit (WAV)
- Processing: Spectral time-stretching
- Smoothing: Savitzky-Golay filter

## Future Enhancements

- [ ] Parallel batch processing
- [ ] Resume interrupted batches
- [ ] Export configuration presets
- [ ] Visualization of parameter distributions
- [ ] Advanced filtering options
- [ ] Automatic dataset splitting (train/val/test)

## License

MIT License

## Contact

For issues or questions, please contact the development team.
