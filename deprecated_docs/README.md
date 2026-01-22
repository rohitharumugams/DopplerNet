# Doppler Effect Batch Simulator

**Professional-grade automated audio generation system for machine learning datasets**

---

##  Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start application
python app_batch.py

# 3. Open browser
http://localhost:5050

# 4. Upload vehicle sounds, configure, generate!
```

**First-time users**: See [QUICKSTART.md](QUICKSTART.md) for detailed 5-minute setup guide.

---

##  What's Included

### Core Application
- **app_batch.py** - Flask backend with batch generation engine
- **index_batch.html** - Professional web interface (no explanations, expert-focused)
- **Physics modules** - Straight line, parabola, and Bezier curve motion
- **Audio processing** - Doppler shift application and quality enhancement

### Documentation
- **README_BATCH.md** - Comprehensive technical documentation
- **QUICKSTART.md** - Getting started in 5 minutes
- **CONFIG_EXAMPLES.md** - 10 ready-to-use configurations
- **PROJECT_SUMMARY.md** - Complete feature overview
- **ARCHITECTURE.txt** - System architecture diagrams
- **DEPLOYMENT_CHECKLIST.md** - Production deployment guide

---

##  Key Features

###  Vehicle Sound Management
- Upload custom audio (3 Â± 0.5 seconds)
- Persistent storage, preview, delete
- Support WAV, MP3, OGG, FLAC

###  Advanced Randomization
- Master "Randomize All" toggle
- Individual parameter controls
- Sensible defaults + manual override

###  Three Motion Paths
- **Straight Line** - Linear motion with angle
- **Parabolic** - Curved trajectory
- **Bezier Curve** - Complex paths (4 control points)

###  Flexible Configuration
- **Auto Mode** - Equal distribution
- **Manual Mode** - Precise control
- **Validate** - Real-time error checking

###  Output Options
- WAV (Recommended) or MP3
- Custom save paths
- Comprehensive metadata logging

###  Complete Logging
- metadata.json - All parameters
- generation_log.txt - Success/failure tracking
- statistics.txt - Aggregate analysis

---

##  Output Structure

```
static/batch_outputs/
 batch_20241121_153045/
     audio_clips/
    ‚    car_straight_25mps_15m_20241121_0001.wav
    ‚    train_parabola_40mps_30m_20241121_0002.wav
    ‚    drone_bezier_18mps_45m_20241121_0003.wav
     metadata.json         # Complete parameter record
     generation_log.txt    # Success/failure log
     statistics.txt        # Distribution statistics
```

---

##  Use Cases

1. **ML Training** - Generate labeled datasets (1000-10000 clips)
2. **Algorithm Testing** - Controlled test sets with known parameters
3. **Research** - Study Doppler effects with varying conditions
4. **Validation** - Compare synthetic vs. real-world audio

---

##  Performance

| Batch Size | Time | Storage (WAV) |
|------------|------|---------------|
| 10 clips | ~10 sec | ~100 MB |
| 100 clips | ~2 min | ~1 GB |
| 1000 clips | ~20 min | ~10 GB |
| 10000 clips | ~3 hours | ~100 GB |

---

##  Default Parameter Ranges

```
Speed (m/s):
  Car:        15-50  (54-180 km/h)
  Train:      20-55  (72-198 km/h)
  Drone:      5-30   (18-108 km/h)
  
Distance:     5-100 m
Duration:     3-8 seconds
Angle:        -45Â° to 45Â°
```

---

##  Documentation Guide

| File | Purpose | Read When |
|------|---------|-----------|
| **README.md** (this) | Overview | First look |
| **QUICKSTART.md** | Fast setup | Getting started |
| **README_BATCH.md** | Full docs | Need details |
| **CONFIG_EXAMPLES.md** | Examples | Configuring batches |
| **ARCHITECTURE.txt** | System design | Understanding internals |
| **PROJECT_SUMMARY.md** | Feature list | What's included |
| **DEPLOYMENT_CHECKLIST.md** | Production | Deploying to server |

---

##  Getting Started Paths

### Path 1: Quick Test (5 minutes)
1. Read QUICKSTART.md
2. Generate 10 test clips
3. Review outputs

### Path 2: Production Setup (30 minutes)
1. Read README_BATCH.md
2. Follow DEPLOYMENT_CHECKLIST.md
3. Generate larger batches

### Path 3: Custom Configuration (1 hour)
1. Review CONFIG_EXAMPLES.md
2. Design your configuration
3. Test and scale up

---

##  Example Workflow

```bash
# 1. Upload vehicle sounds
# (Use web interface at http://localhost:5050)

# 2. Configure batch (Web UI or API)
{
  "batch": {"total_clips": 100, "mode": "auto"},
  "output": {"format": "wav"},
  "vehicles": {"randomize": true},
  "paths": {"randomize": true},
  "speed": {"randomize": true}
}

# 3. Generate
# Click "Generate Batch" button

# 4. Review outputs
cd static/batch_outputs/batch_20241121_153045/
ls audio_clips/  # WAV files
cat metadata.json  # Parameters
cat statistics.txt  # Distribution
```

---

##  Technical Highlights

### Physics Accuracy
- Doppler shift: f'/f = c/(c - vr)
- Sound speed: 343 m/s at 20Â°C
- Radial velocity calculation
- Inverse distance amplitude

### Audio Quality
- Sample rate: 22,050 Hz
- 16-bit WAV output
- Spectral time-stretching
- Savitzky-Golay smoothing

---

##  Pro Tips

1. **Always use WAV for ML training** (lossless)
2. **Start small** (10-100 clips) to verify setup
3. **Check generation_log.txt** for errors
4. **Use auto-split** for balanced datasets
5. **Use manual mode** for specific distributions

---

##  Troubleshooting

### Common Issues

**Vehicle upload fails**
- Check duration (3Â±0.5 seconds)
- Verify format (WAV/MP3/OGG/FLAC)

**Distribution validation error**
- Ensure sums equal total (manual mode)
- Check both vehicle AND path distributions

**Generation slow**
- Reduce batch size
- Use SSD storage
- Close other applications

**More solutions**: See DEPLOYMENT_CHECKLIST.md

---

##  System Requirements

### Minimum
- Python 3.8+
- 4 GB RAM
- 50 GB disk space

### Recommended
- Python 3.10+
- 16 GB RAM
- 500 GB SSD
- 8-core CPU

---

##  Contributing

This is a research tool. Feedback and improvements welcome!

---

##  License

MIT License

---

##  Next Steps

1.  Read QUICKSTART.md
2.  Generate first batch (10 clips)
3.  Review outputs and metadata
4.  Scale up to production size
5.  Integrate with your ML pipeline

---

**Built for researchers who need professional-grade Doppler effect datasets.**

For detailed information on any topic, see the documentation files listed above.
