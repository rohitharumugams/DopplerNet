# Configuration Examples

This file contains example configurations for common use cases.

## Example 1: Fully Random Small Batch

```json
{
  "output": {
    "format": "wav",
    "path": "static/batch_outputs/"
  },
  "vehicles": {
    "randomize": true,
    "selected": []
  },
  "paths": {
    "randomize": true,
    "selected": []
  },
  "speed": {
    "randomize": true,
    "min": null,
    "max": null
  },
  "distance": {
    "randomize": true,
    "min": null,
    "max": null
  },
  "duration": {
    "randomize": true,
    "min": null,
    "max": null
  },
  "angle": {
    "randomize": true,
    "min": null,
    "max": null
  },
  "batch": {
    "total_clips": 10,
    "mode": "auto"
  }
}
```

**Result**: 10 clips with all parameters randomized, evenly split across available vehicles and paths.

---

## Example 2: Fixed Distance, Variable Speed

```json
{
  "output": {
    "format": "wav"
  },
  "vehicles": {
    "randomize": true
  },
  "paths": {
    "randomize": true
  },
  "speed": {
    "randomize": true,
    "min": 10,
    "max": 60
  },
  "distance": {
    "randomize": false,
    "min": 20,
    "max": 20
  },
  "duration": {
    "randomize": true
  },
  "batch": {
    "total_clips": 100,
    "mode": "auto"
  }
}
```

**Result**: 100 clips, all at 20m distance, varying speeds between 10-60 m/s.

---

## Example 3: Only Straight-Line Motion

```json
{
  "output": {
    "format": "wav"
  },
  "vehicles": {
    "randomize": true
  },
  "paths": {
    "randomize": false,
    "selected": ["straight"]
  },
  "speed": {
    "randomize": true
  },
  "distance": {
    "randomize": true
  },
  "duration": {
    "randomize": true
  },
  "angle": {
    "randomize": true,
    "min": -45,
    "max": 45
  },
  "batch": {
    "total_clips": 200,
    "mode": "auto"
  }
}
```

**Result**: 200 clips, all straight-line motion, varying angles from -45° to 45°.

---

## Example 4: Manual Distribution - Unbalanced Dataset

```json
{
  "output": {
    "format": "wav"
  },
  "vehicles": {
    "randomize": false,
    "selected": ["car", "train", "drone"]
  },
  "paths": {
    "randomize": false,
    "selected": ["straight", "parabola", "bezier"]
  },
  "speed": {
    "randomize": true
  },
  "distance": {
    "randomize": true
  },
  "duration": {
    "randomize": true
  },
  "batch": {
    "total_clips": 1000,
    "mode": "manual",
    "distribution": {
      "vehicles": {
        "car": 500,
        "train": 300,
        "drone": 200
      },
      "paths": {
        "straight": 400,
        "parabola": 350,
        "bezier": 250
      }
    }
  }
}
```

**Result**: 1000 clips with custom distribution. More car clips (500), fewer drone clips (200). More straight-line clips (400).

---

## Example 5: High-Speed Scenario

```json
{
  "output": {
    "format": "wav"
  },
  "vehicles": {
    "randomize": false,
    "selected": ["train", "car"]
  },
  "paths": {
    "randomize": true
  },
  "speed": {
    "randomize": false,
    "min": 40,
    "max": 70
  },
  "distance": {
    "randomize": true,
    "min": 10,
    "max": 50
  },
  "duration": {
    "randomize": false,
    "min": 4,
    "max": 6
  },
  "batch": {
    "total_clips": 500,
    "mode": "auto"
  }
}
```

**Result**: 500 clips focused on high-speed scenarios (40-70 m/s), closer distances (10-50m), medium duration (4-6s).

---

## Example 6: Low-Speed Urban Scenario

```json
{
  "output": {
    "format": "wav"
  },
  "vehicles": {
    "randomize": false,
    "selected": ["car", "motorcycle"]
  },
  "paths": {
    "randomize": false,
    "selected": ["straight"]
  },
  "speed": {
    "randomize": false,
    "min": 5,
    "max": 25
  },
  "distance": {
    "randomize": false,
    "min": 3,
    "max": 15
  },
  "duration": {
    "randomize": false,
    "min": 5,
    "max": 7
  },
  "angle": {
    "randomize": false,
    "min": -15,
    "max": 15
  },
  "batch": {
    "total_clips": 300,
    "mode": "auto"
  }
}
```

**Result**: 300 clips simulating urban traffic. Low speeds (5-25 m/s), close distances (3-15m), small angles (-15° to 15°).

---

## Example 7: Aerial Drone Dataset

```json
{
  "output": {
    "format": "wav"
  },
  "vehicles": {
    "randomize": false,
    "selected": ["drone"]
  },
  "paths": {
    "randomize": false,
    "selected": ["parabola", "bezier"]
  },
  "speed": {
    "randomize": false,
    "min": 5,
    "max": 30
  },
  "distance": {
    "randomize": false,
    "min": 5,
    "max": 80
  },
  "duration": {
    "randomize": false,
    "min": 3,
    "max": 8
  },
  "parabola": {
    "randomize": true
  },
  "bezier": {
    "randomize": true
  },
  "batch": {
    "total_clips": 400,
    "mode": "auto"
  }
}
```

**Result**: 400 clips of drone audio with curved/complex paths, appropriate speed ranges for drones.

---

## Example 8: MP3 Compressed Dataset

```json
{
  "output": {
    "format": "mp3",
    "path": "static/batch_outputs/"
  },
  "vehicles": {
    "randomize": true
  },
  "paths": {
    "randomize": true
  },
  "speed": {
    "randomize": true
  },
  "distance": {
    "randomize": true
  },
  "duration": {
    "randomize": true
  },
  "batch": {
    "total_clips": 1000,
    "mode": "auto"
  }
}
```

**Result**: 1000 clips in MP3 format for reduced file size (good for quick testing, not recommended for ML training).

---

## Example 9: Training-Validation Split

### Training Set (80%)
```json
{
  "batch": {
    "total_clips": 8000,
    "mode": "auto"
  },
  "output": {
    "path": "static/batch_outputs/train/"
  }
  // ... rest of config
}
```

### Validation Set (20%)
```json
{
  "batch": {
    "total_clips": 2000,
    "mode": "auto"
  },
  "output": {
    "path": "static/batch_outputs/val/"
  }
  // ... rest of config (same parameters)
}
```

**Result**: Proper train/validation split for ML training.

---

## Example 10: Maximum Diversity

```json
{
  "output": {
    "format": "wav"
  },
  "vehicles": {
    "randomize": true
  },
  "paths": {
    "randomize": true
  },
  "speed": {
    "randomize": false,
    "min": 5,
    "max": 70
  },
  "distance": {
    "randomize": false,
    "min": 3,
    "max": 100
  },
  "duration": {
    "randomize": false,
    "min": 3,
    "max": 8
  },
  "angle": {
    "randomize": false,
    "min": -45,
    "max": 45
  },
  "batch": {
    "total_clips": 5000,
    "mode": "auto"
  }
}
```

**Result**: 5000 clips with maximum parameter diversity. Wide speed range (5-70 m/s), wide distance range (3-100m), all durations and angles.

---

## How to Use These Configurations

### Method 1: Copy-Paste to Browser Console
```javascript
// In browser console (F12)
const config = { /* paste config here */ };
fetch('/api/batch_generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(config)
});
```

### Method 2: Save as JSON file and use API
```bash
# Save config to file
cat > config.json << EOF
{ /* paste config */ }
EOF

# Send to API
curl -X POST http://localhost:5050/api/batch_generate \
  -H "Content-Type: application/json" \
  -d @config.json
```

### Method 3: Use Web Interface
Simply configure the UI to match the desired settings.

---

## Parameter Cheat Sheet

### Speed Ranges by Vehicle Type
```
Car:        15-50 m/s  (54-180 km/h)
Train:      20-55 m/s  (72-198 km/h)
Drone:      5-30 m/s   (18-108 km/h)
Motorcycle: 10-45 m/s  (36-162 km/h)
```

### Distance Guidelines
```
Close:      3-15 m   (street level)
Medium:     15-50 m  (near field)
Far:        50-100 m (far field)
```

### Duration Guidelines
```
Short:   3-4 s   (quick pass)
Medium:  4-6 s   (standard)
Long:    6-8 s   (extended)
```

### Angle Guidelines (Straight Line)
```
Perpendicular: 0°      (crosses in front)
Diagonal:      ±15-30° (angled approach)
Sharp angle:   ±30-45° (extreme angle)
```

---

## Tips for Choosing Configurations

1. **For ML Training**: Use maximum diversity (Example 10)
2. **For Testing**: Use small batches with fixed parameters
3. **For Specific Scenarios**: Use constrained ranges (Examples 5-7)
4. **For Balanced Datasets**: Use auto-split mode
5. **For Custom Distributions**: Use manual mode (Example 4)

6. **Always use WAV** for ML training (lossless)
7. **Use MP3** only for demos or testing (lossy)

8. **Start small** (10-100 clips) to verify configuration
9. **Scale up gradually** once verified
10. **Check disk space** before large batches (~10 MB per clip)
