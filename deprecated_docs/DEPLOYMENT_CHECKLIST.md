# Deployment Checklist - Doppler Effect Batch Simulator

## Pre-Deployment

### Environment Setup
- [ ] Python 3.8+ installed
- [ ] pip package manager available
- [ ] Virtual environment created (recommended)
  ```bash
  python -m venv venv
  source venv/bin/activate  # Linux/Mac
  # or
  venv\Scripts\activate  # Windows
  ```

### Dependencies Installation
- [ ] Install required packages
  ```bash
  pip install -r requirements.txt
  ```
- [ ] Verify installations
  ```bash
  python -c "import flask, numpy, soundfile, librosa, scipy; print('All imports successful')"
  ```

### Directory Structure
- [ ] Create necessary directories
  ```bash
  mkdir -p static/vehicle_sounds
  mkdir -p static/batch_outputs
  mkdir -p templates
  ```
- [ ] Verify write permissions
  ```bash
  touch static/vehicle_sounds/test.txt && rm static/vehicle_sounds/test.txt
  touch static/batch_outputs/test.txt && rm static/batch_outputs/test.txt
  ```

### File Verification
- [ ] All Python files present:
  - `app_batch.py`
  - `audio_utils.py`
  - `straight_line.py`
  - `parabola.py`
  - `bezier.py`
- [ ] Template file present:
  - `templates/index_batch.html`
- [ ] Documentation present:
  - `README_BATCH.md`
  - `QUICKSTART.md`
  - `CONFIG_EXAMPLES.md`
  - `PROJECT_SUMMARY.md`
  - `ARCHITECTURE.txt`

## Testing Phase

### Initial Test
- [ ] Start application
  ```bash
  python app_batch.py
  ```
- [ ] Verify server starts without errors
- [ ] Check output shows:
  ```
  === Batch Doppler Simulator Starting ===
  * Running on http://0.0.0.0:5050
  ```

### Web Interface Test
- [ ] Open browser to http://localhost:5050
- [ ] Verify page loads correctly
- [ ] Check all UI elements visible:
  - Vehicle upload area
  - Parameter controls
  - Master "Randomize All" toggle
  - Generate button

### Vehicle Upload Test
- [ ] Prepare test audio (3±0.5 seconds)
- [ ] Upload vehicle sound
- [ ] Verify success message
- [ ] Check file in `static/vehicle_sounds/`
- [ ] Verify vehicle appears in list
- [ ] Test delete functionality

### Small Batch Test (Critical)
- [ ] Set total clips to 10
- [ ] Enable "Randomize All"
- [ ] Click "Generate Batch"
- [ ] Wait for completion
- [ ] Verify output folder created
- [ ] Check all 10 audio files generated
- [ ] Verify metadata.json exists and valid
- [ ] Check generation_log.txt for errors
- [ ] Review statistics.txt

### Medium Batch Test
- [ ] Generate 100 clips
- [ ] Verify successful completion
- [ ] Check file sizes reasonable (~10MB per clip)
- [ ] Spot-check audio quality (listen to random files)
- [ ] Verify metadata completeness

## Production Deployment

### Server Configuration (if deploying to server)
- [ ] Set appropriate host/port in app_batch.py
  ```python
  app.run(debug=False, host='0.0.0.0', port=5050)
  ```
- [ ] Disable debug mode for production
- [ ] Consider using production WSGI server (gunicorn, uwsgi)
  ```bash
  pip install gunicorn
  gunicorn -w 4 -b 0.0.0.0:5050 app_batch:app
  ```

### Security Considerations
- [ ] Set max upload file size
- [ ] Add rate limiting for API endpoints
- [ ] Implement authentication (if needed)
- [ ] Configure CORS properly for your domain
- [ ] Set up HTTPS (if exposed to internet)

### Performance Optimization (for large batches)
- [ ] Enable multiprocessing (modify app_batch.py)
- [ ] Use SSD storage for faster I/O
- [ ] Increase file descriptors limit if needed
  ```bash
  ulimit -n 4096
  ```
- [ ] Monitor memory usage during generation

### Monitoring Setup
- [ ] Set up logging to file
- [ ] Configure log rotation
- [ ] Monitor disk space
  ```bash
  df -h
  ```
- [ ] Track generation success rate

### Backup Strategy
- [ ] Regular backups of vehicle_sounds/
- [ ] Archive completed batches if needed
- [ ] Document backup procedures

## Supercomputer Deployment

### Environment Module Loading (if applicable)
- [ ] Load required modules
  ```bash
  module load python/3.10
  module load scipy
  module load ffmpeg  # for audio processing
  ```

### Resource Allocation
- [ ] Request appropriate resources
  ```bash
  # Example SLURM submission
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=8
  #SBATCH --mem=32GB
  #SBATCH --time=24:00:00
  ```

### Parallel Processing Setup
- [ ] Modify app_batch.py for multiprocessing
- [ ] Test with small batch first
- [ ] Scale up gradually

### Storage Considerations
- [ ] Use scratch storage for temporary files
- [ ] Move completed batches to permanent storage
- [ ] Clean up old files regularly

## Post-Deployment Validation

### Functional Tests
- [ ] Test all three path types (straight, parabola, bezier)
- [ ] Test manual distribution mode
- [ ] Test parameter range overrides
- [ ] Test with different vehicles
- [ ] Test MP3 output option
- [ ] Verify graph toggle works

### Data Quality Checks
- [ ] Listen to sample outputs
- [ ] Check spectrograms for Doppler effect
- [ ] Verify metadata accuracy
- [ ] Confirm parameter distributions match config

### Performance Benchmarks
- [ ] Time 10 clips: _____ seconds
- [ ] Time 100 clips: _____ seconds
- [ ] Time 1000 clips: _____ minutes
- [ ] Memory usage at peak: _____ GB
- [ ] Disk space per 1000 clips: _____ GB

### Documentation Review
- [ ] Update README with deployment-specific info
- [ ] Document any custom configurations
- [ ] Note any known issues or limitations
- [ ] Provide contact info for support

## Common Issues & Solutions

### Issue: "Module not found" errors
**Solution**: Reinstall requirements
```bash
pip install --upgrade -r requirements.txt
```

### Issue: "Permission denied" when saving files
**Solution**: Check directory permissions
```bash
chmod -R 755 static/
```

### Issue: Application won't start
**Solution**: Check port availability
```bash
lsof -i :5050
# If port in use, kill process or change port
```

### Issue: Vehicle upload fails
**Solution**: 
- Check audio duration (must be 3±0.5 sec)
- Verify file format (WAV/MP3/OGG/FLAC)
- Ensure disk space available

### Issue: Batch generation hangs
**Solution**:
- Check system resources (RAM, CPU)
- Reduce batch size
- Review generation_log.txt for errors

### Issue: Out of memory
**Solution**:
- Reduce concurrent processing
- Process in smaller batches
- Increase system RAM allocation

## Maintenance Tasks

### Daily
- [ ] Check disk space usage
- [ ] Review generation logs for errors
- [ ] Verify backup completion

### Weekly
- [ ] Clean up old batch outputs (if needed)
- [ ] Review system performance
- [ ] Update documentation if needed

### Monthly
- [ ] Update dependencies
  ```bash
  pip list --outdated
  pip install --upgrade <package>
  ```
- [ ] Review and optimize performance
- [ ] Archive old batches

## Success Criteria

Deployment is successful when:
-  Application starts without errors
-  Web interface accessible and functional
-  Vehicle upload/management works
-  Small batch (10 clips) generates successfully
-  Medium batch (100 clips) completes in reasonable time
-  All output files created correctly
-  Audio quality is satisfactory
-  Metadata is complete and accurate
-  No memory leaks or performance degradation
-  Documentation is clear and accessible

## Support & Contact

For issues or questions:
1. Check generation_log.txt for error details
2. Review metadata.json for configuration issues
3. Consult README_BATCH.md for detailed documentation
4. Review CONFIG_EXAMPLES.md for configuration guidance
5. Contact development team: [your contact info]

---

**Date Deployed**: _______________
**Deployed By**: _______________
**Environment**: _______________
**Notes**: _______________
