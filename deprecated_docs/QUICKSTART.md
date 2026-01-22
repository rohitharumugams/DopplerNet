# Quick Start Guide - Doppler Effect Batch Simulator

## Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Application
```bash
python3 app_batch.py
```

### 3. Open Browser
```
http://localhost:5050
```

## First Batch Generation (3 minutes)

### Step 1: Upload Vehicle Sounds
```
1. Click the upload area
2. Select audio file (must be 3Â±0.5 seconds)
3. Enter name: "car"
4. Click outside to upload
5. Repeat for "train", "drone", etc.
```

### Step 2: Quick Generate
```
1. Toggle "RANDOMIZE ALL" ON
2. Set "Total Clips" to 10 (for testing)
3. Keep output format as "WAV (Recommended)"
4. Click "Generate Batch"
```

### Step 3: Check Output
```
Navigate to: static/batch_outputs/batch_YYYYMMDD_HHMMSS/
- audio_clips/ folder contains all WAV files
- metadata.json has all parameters
- generation_log.txt shows success/failure
- statistics.txt has distribution info
```

## Example Configurations

### Small Test Batch (10 clips)
```
- Total Clips: 10
- Randomize All: ON
- Output: WAV
- Time: ~10 seconds
```

### Medium Training Set (100 clips)
```
- Total Clips: 100
- Distribution: Auto-split
- Randomize All: ON
- Output: WAV
- Time: ~2 minutes
```

### Large Dataset (1000 clips)
```
- Total Clips: 1000
- Distribution: Manual
  - Car: 400
  - Train: 350
  - Drone: 250
- Speed: Custom range (20-60 m/s)
- Output: WAV
- Time: ~20 minutes
```

### Production Dataset (5000+ clips)
```
- Run on supercomputer
- Multiple batches recommended
- Use manual distribution for balance
- WAV format for ML training
- Monitor disk space (~50GB for 5000 clips)
```

## File Naming Convention

Generated files follow this pattern:
```
{vehicle}_{path}_{speed}mps_{distance}m_{timestamp}_{index}.wav

Example:
car_straight_25mps_15m_20241121_0001.wav
  ‚      ‚       ‚      ‚        ‚       Clip index
  ‚      ‚       ‚      ‚         Batch timestamp
  ‚      ‚       ‚       Distance parameter
  ‚      ‚        Speed parameter
  ‚       Path type
   Vehicle type
```

## Common Use Cases

### 1. Balanced Training Dataset
```
Goal: 1000 clips, evenly distributed
Steps:
  - Upload 3 vehicles (car, train, drone)
  - Toggle "RANDOMIZE ALL" ON
  - Total Clips: 1000
  - Mode: Auto-split
  - Generate
Result: 333-334 clips per vehicle, evenly across paths
```

### 2. Speed-Focused Dataset
```
Goal: Vary speeds, fixed distance
Steps:
  - Randomize Speed: ON
  - Randomize Distance: OFF
    - Min: 15m, Max: 15m (fixed)
  - Generate
Result: All clips at 15m distance, varying speeds
```

### 3. Path-Specific Dataset
```
Goal: Only straight-line motion
Steps:
  - Randomize Path: OFF
  - Select: Straight only
  - Generate
Result: All clips use straight-line motion
```

### 4. Multi-Angle Dataset
```
Goal: Test different approach angles
Steps:
  - Select: Straight path only
  - Randomize Angle: ON (or set custom range)
  - Generate
Result: Clips with varying angles (-45Â° to 45Â°)
```

## Tips for Supercomputer Use

### Optimize for Speed
```python
# In app_batch.py, add parallel processing:
from multiprocessing import Pool

def generate_parallel(configs, num_workers=8):
    with Pool(num_workers) as pool:
        results = pool.map(generate_single_clip, configs)
    return results
```

### Batch Splitting
```
Instead of 10,000 clips in one batch:
- Run 10 batches of 1,000 clips
- Easier to manage
- Faster recovery from failures
- Better progress tracking
```

### Storage Management
```bash
# Check disk space before generating
df -h

# Estimate needed space
# WAV: ~10MB per 5-second clip
# For 1000 clips: ~10GB
# For 10,000 clips: ~100GB
```

## Troubleshooting

### "Audio duration must be 3Â±0.5 seconds"
- Check your audio file length
- Trim or extend to 2.5-3.5 seconds
- Use audio editor (Audacity, etc.)

### "Vehicle distribution sum must equal total"
- In Manual mode, counts must match
- Example: Total=100, Car=40, Train=30, Drone=30 
- Example: Total=100, Car=40, Train=35, Drone=30  (=105)

### Generation stuck or slow
- Check system resources (CPU, RAM)
- Reduce batch size
- Use faster storage (SSD)
- Close other applications

### No vehicles showing up
- Check `static/vehicle_sounds/` directory
- Verify files are .wav format
- Re-upload if necessary

## Next Steps

1. **Test with small batch** (10 clips) to verify setup
2. **Review outputs** (audio quality, metadata accuracy)
3. **Scale up gradually** (100  1000  10000)
4. **Customize parameters** for your specific use case
5. **Integrate with ML pipeline** using metadata.json

## Support

For detailed documentation, see README_BATCH.md

For issues:
1. Check generation_log.txt for errors
2. Review metadata.json for configuration
3. Verify system requirements
4. Contact development team
