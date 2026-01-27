import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import os
import json
import uuid
import time
from datetime import datetime
import random
import soundfile as sf
import librosa
import traceback

# NEW: for saving path graphs
import matplotlib
matplotlib.use('Agg')  # non-GUI backend for servers
import matplotlib.pyplot as plt

# NEW: for spectrograms
SPECTROGRAM_FOLDER = os.path.join('static', 'spectrograms')
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)
import librosa.display

from audio_utils import (
    load_original_audio,
    apply_doppler_to_audio_fixed,
    normalize_amplitudes,
    save_audio,
    SR
)

from straight_line import calculate_straight_line_doppler
from parabola import calculate_parabola_doppler
from bezier import calculate_bezier_doppler

# NEW: Import scipy for validation
from scipy.ndimage import gaussian_filter1d

app = Flask(__name__)
CORS(app)

# Directory structure
UPLOAD_FOLDER = 'static/vehicle_sounds'
DRONE_SOUNDS_FOLDER = 'static/drone_sounds'  # Fallback directory
OUTPUT_FOLDER = 'static/batch_outputs'
SINGLE_OUTPUT_FOLDER = 'static/single_outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DRONE_SOUNDS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SINGLE_OUTPUT_FOLDER, exist_ok=True)

# Default ranges for randomization
# NOTE: enforce constraints:
# - parabola_a is positive so it opens upwards
# - Bezier y will be sampled positive in code
# - angle limited to [-45, 45]
# ============================================================
# GLOBAL SAMPLER CACHE (PER BATCH)
# ============================================================
SAMPLERS = {}

# ============================================================
# GLOBAL GENERATION STATE (CONTINUOUS BATCHING)
# ============================================================

SAMPLER_STATE_FILE = "sampler_state.json"
PROGRESS_FILE = "generation_progress.json"


def save_sampler_state():
    state = {}
    for key, sampler in SAMPLERS.items():
        state[key] = {
            "low": sampler.low,
            "high": sampler.high,
            "step": sampler.step,
            "offset": sampler.offset,
            "k": sampler.k
        }
    with open(SAMPLER_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_sampler_state():
    if not os.path.exists(SAMPLER_STATE_FILE):
        return
    with open(SAMPLER_STATE_FILE, "r") as f:
        state = json.load(f)

    for key, s in state.items():
        sampler = CyclicIntegerSampler(s["low"], s["high"])
        sampler.step = s["step"]
        sampler.offset = s["offset"]
        sampler.k = s["k"]
        SAMPLERS[key] = sampler


def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {"total_target": 0, "generated_so_far": 0}
    with open(PROGRESS_FILE, "r") as f:
        return json.load(f)


def save_progress(total_target, generated_so_far):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(
            {
                "total_target": total_target,
                "generated_so_far": generated_so_far
            },
            f,
            indent=2
        )

DEFAULT_RANGES = {
    'speed': {
        'car': (15, 50),
        'train': (20, 55),
        'drone': (5, 30),
        'motorcycle': (10, 45),
        'default': (10, 50)
    },
    'distance': (5, 100),
    # duration range is now unused (we force 10s everywhere),
    # but we keep it for reference
    'duration': (3, 8),
    'angle': (-45, 45),
    # for Bezier: allow negative x, but y will be kept positive in code
    'bezier_coords': (-150, 150),
    # positive only so parabola opens towards +y
    'parabola_a': (5, 20),       # will be divided by 10000 => 0.0005 to 0.0020
    'parabola_h': (10, 50)       # always positive height
}


# ============================================================
# PATH VALIDATION FUNCTIONS
# ============================================================

def compute_road_boundaries(all_paths, lane_width, include_opposite=False):
    """
    Compute road boundaries (upper, lower, and centerline) based on all vehicle paths.
    
    Args:
        all_paths: List of (x, y) tuples for each vehicle
        lane_width: Total width of the road
        include_opposite: Whether the road has opposite-direction traffic
        
    Returns:
        x_common: Common x-coordinates for boundaries
        y_upper: Upper road boundary
        y_lower: Lower road boundary
        y_centerline: Center line (median)
    """
    if not all_paths:
        return None, None, None, None
    
    # Find x range
    x_min = min(np.min(x) for x, y in all_paths)
    x_max = max(np.max(x) for x, y in all_paths)
    
    # Common x grid
    x_common = np.linspace(x_min, x_max, 300)
    
    # Interpolate all paths onto common x-grid
    y_interpolated = []
    for x, y in all_paths:
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        y_interp = np.interp(x_common, x_sorted, y_sorted, 
                            left=y_sorted[0], right=y_sorted[-1])
        y_interpolated.append(y_interp)
    
    # Calculate centerline (split into two balanced groups)
    avg_y_per_path = [np.mean(y) for y in y_interpolated]
    
    if len(y_interpolated) > 1:
        sorted_indices = np.argsort(avg_y_per_path)
        num_cars = len(sorted_indices)
        split_idx = num_cars // 2
        
        lower_indices = sorted_indices[:split_idx]
        upper_indices = sorted_indices[split_idx:]
        
        y_lower_max = np.max([y_interpolated[i] for i in lower_indices], axis=0)
        y_upper_min = np.min([y_interpolated[i] for i in upper_indices], axis=0)
        
        y_centerline = (y_lower_max + y_upper_min) / 2
    else:
        y_centerline = np.mean(y_interpolated, axis=0)
    
    # Smooth centerline
    y_centerline_smooth = gaussian_filter1d(y_centerline, sigma=10)
    
    # Compute adaptive lane width
    y_min_env = np.min(y_interpolated, axis=0)
    y_max_env = np.max(y_interpolated, axis=0)
    actual_spread = np.max(y_max_env - y_min_env)
    effective_lane_width = max(lane_width, actual_spread + 6.0)
    
    # Road boundaries
    y_upper = y_centerline_smooth + effective_lane_width / 2
    y_lower = y_centerline_smooth - effective_lane_width / 2
    
    # Apply the y-shift transformation used in plotting (from the code)
    y_shift = 7.5
    y_upper += y_shift
    y_lower += y_shift
    y_centerline_smooth += y_shift
    
    return x_common, y_upper, y_lower, y_centerline_smooth


def check_path_violations(path_x, path_y, x_common, y_upper, y_lower, y_centerline, 
                         tolerance=0.5):
    """
    Check if a single path violates road boundaries or crosses the median.
    
    Args:
        path_x, path_y: Vehicle path coordinates
        x_common: Common x-coordinates for boundaries
        y_upper, y_lower: Road boundaries
        y_centerline: Center line (median)
        tolerance: Allowed tolerance for violations (in meters)
        
    Returns:
        dict with violation information
    """
    # Interpolate boundaries to path's x-coordinates
    y_upper_interp = np.interp(path_x, x_common, y_upper)
    y_lower_interp = np.interp(path_x, x_common, y_lower)
    y_center_interp = np.interp(path_x, x_common, y_centerline)
    
    violations = {
        'has_violation': False,
        'upper_boundary_violations': [],
        'lower_boundary_violations': [],
        'median_crossings': [],
        'summary': ''
    }
    
    # Check upper boundary violations
    upper_violations = path_y > (y_upper_interp + tolerance)
    if np.any(upper_violations):
        violation_indices = np.where(upper_violations)[0]
        max_violation = np.max(path_y[upper_violations] - y_upper_interp[upper_violations])
        violations['has_violation'] = True
        violations['upper_boundary_violations'] = [
            {
                'index': int(idx),
                'x': float(path_x[idx]),
                'y': float(path_y[idx]),
                'boundary_y': float(y_upper_interp[idx]),
                'violation_distance': float(path_y[idx] - y_upper_interp[idx])
            }
            for idx in violation_indices[::10]  # Sample every 10th point
        ]
        violations['summary'] += f"Upper boundary violated by {max_violation:.2f}m. "
    
    # Check lower boundary violations
    lower_violations = path_y < (y_lower_interp - tolerance)
    if np.any(lower_violations):
        violation_indices = np.where(lower_violations)[0]
        max_violation = np.max(y_lower_interp[lower_violations] - path_y[lower_violations])
        violations['has_violation'] = True
        violations['lower_boundary_violations'] = [
            {
                'index': int(idx),
                'x': float(path_x[idx]),
                'y': float(path_y[idx]),
                'boundary_y': float(y_lower_interp[idx]),
                'violation_distance': float(y_lower_interp[idx] - path_y[idx])
            }
            for idx in violation_indices[::10]
        ]
        violations['summary'] += f"Lower boundary violated by {max_violation:.2f}m. "
    
    # Check median crossings (detect when path crosses from one side to other)
    y_relative_to_center = path_y - y_center_interp
    sign_changes = np.diff(np.sign(y_relative_to_center))
    crossings = np.where(sign_changes != 0)[0]
    
    if len(crossings) > 0:
        violations['has_violation'] = True
        violations['median_crossings'] = [
            {
                'index': int(idx),
                'x': float(path_x[idx]),
                'y_before': float(path_y[idx]),
                'y_after': float(path_y[idx + 1]),
                'centerline_y': float(y_center_interp[idx])
            }
            for idx in crossings
        ]
        violations['summary'] += f"Median crossed {len(crossings)} time(s). "
    
    if not violations['has_violation']:
        violations['summary'] = "No violations detected."
    
    return violations


def validate_scene_paths(scenes_data, lane_width=4.0, include_opposite=False, 
                        tolerance=0.5, y_shift=7.5):
    """
    Validate all vehicle paths in a scene for boundary and median violations.
    
    Args:
        scenes_data: List of (path_type, params, vehicle_name) tuples
        lane_width: Total road width
        include_opposite: Whether road has opposite traffic
        tolerance: Allowed tolerance for violations (meters)
        y_shift: Y-axis transformation applied to paths (from plotting code)
        
    Returns:
        dict with validation results for all vehicles
    """
    # Compute all vehicle paths
    all_paths = []
    for path_type, params, vehicle_name in scenes_data:
        x, y, _ = compute_path_points(path_type, params, n_points=200)
        # Apply the same y_shift transformation as in plotting
        y_shifted = y + y_shift
        all_paths.append((x, y_shifted))
    
    # Compute road boundaries
    x_common, y_upper, y_lower, y_centerline = compute_road_boundaries(
        all_paths, lane_width, include_opposite
    )
    
    # Validate each vehicle path
    results = {
        'scene_valid': True,
        'total_vehicles': len(scenes_data),
        'vehicles_with_violations': 0,
        'vehicle_results': []
    }
    
    for i, ((x, y), (path_type, params, vehicle_name)) in enumerate(zip(all_paths, scenes_data)):
        violations = check_path_violations(x, y, x_common, y_upper, y_lower, 
                                          y_centerline, tolerance)
        
        vehicle_result = {
            'vehicle_id': i + 1,
            'vehicle_name': vehicle_name,
            'path_type': path_type,
            'violations': violations
        }
        
        if violations['has_violation']:
            results['scene_valid'] = False
            results['vehicles_with_violations'] += 1
        
        results['vehicle_results'].append(vehicle_result)
    
    return results


def save_validation_report(validation_results, output_dir, scene_id):
    """Save validation results to JSON and text files."""
    # Save JSON
    validation_file = os.path.join(output_dir, f"validation.json")
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Save human-readable report
    report_file = os.path.join(output_dir, f"validation.txt")
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"PATH VALIDATION REPORT - Scene {scene_id}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total Vehicles: {validation_results['total_vehicles']}\n")
        f.write(f"Scene Valid: {'YES' if validation_results['scene_valid'] else 'NO'}\n")
        f.write(f"Vehicles with Violations: {validation_results['vehicles_with_violations']}\n")
        f.write("=" * 70 + "\n\n")
        
        for vehicle_result in validation_results['vehicle_results']:
            vid = vehicle_result['vehicle_id']
            vname = vehicle_result['vehicle_name']
            vpath = vehicle_result['path_type']
            violations = vehicle_result['violations']
            
            status = "VALID" if not violations['has_violation'] else "INVALID"
            f.write(f"Vehicle {vid}: {vname} ({vpath}) - {status}\n")
            
            if violations['has_violation']:
                f.write(f"  Summary: {violations['summary']}\n")
                
                if violations['upper_boundary_violations']:
                    f.write(f"  - Upper boundary violations: "
                          f"{len(violations['upper_boundary_violations'])} points\n")
                
                if violations['lower_boundary_violations']:
                    f.write(f"  - Lower boundary violations: "
                          f"{len(violations['lower_boundary_violations'])} points\n")
                
                if violations['median_crossings']:
                    f.write(f"  - Median crossings: {len(violations['median_crossings'])}\n")
            
            f.write("\n")
    
    return validation_file, report_file


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template('index_batch.html')


@app.route('/api/upload_vehicle', methods=['POST'])
def upload_vehicle():
    """Upload vehicle audio file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        vehicle_name = request.form.get('vehicle_name', 'unnamed')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate audio file
        if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            return jsonify({'error': 'Invalid audio format. Use WAV, MP3, OGG, or FLAC'}), 400

        # Save temporarily to check duration
        temp_path = os.path.join(UPLOAD_FOLDER, f'temp_{uuid.uuid4()}.wav')
        file.save(temp_path)

        # Load and check duration
        try:
            audio, sr = librosa.load(temp_path, sr=SR, mono=True)
            duration = len(audio) / SR

            if not (2.5 <= duration <= 3.5):
                os.remove(temp_path)
                return jsonify({'error': f'Audio duration must be 3±0.5 seconds. Got {duration:.2f}s'}), 400

            # Save with proper name
            safe_name = "".join(c for c in vehicle_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            filename = f'{safe_name}.wav'
            final_path = os.path.join(UPLOAD_FOLDER, filename)

            # Convert to WAV format
            sf.write(final_path, audio, SR)
            os.remove(temp_path)

            return jsonify({
                'success': True,
                'filename': filename,
                'vehicle_name': safe_name,
                'duration': duration
            })

        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': f'Failed to process audio: {str(e)}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/list_vehicles', methods=['GET'])
def list_vehicles():
    """List all vehicle sounds from static/vehicle_sounds and static/drone_sounds"""
    try:
        # Optional filter by source type
        source_filter = request.args.get('source', 'all')  # 'vehicle', 'drone', or 'all'
        
        vehicles = []
        
        # Scan both directories
        folders_to_scan = [
            (UPLOAD_FOLDER, 'vehicle'),
            (DRONE_SOUNDS_FOLDER, 'drone')
        ]
        
        for folder, source_type in folders_to_scan:
            # Skip if filtering and this source doesn't match
            if source_filter != 'all' and source_filter != source_type:
                continue
                
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    if filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                        filepath = os.path.join(folder, filename)
                        try:
                            audio, sr = librosa.load(filepath, sr=SR, mono=True)
                            duration = len(audio) / SR
                            # Remove any audio extension
                            vehicle_name = filename
                            for ext in ['.wav', '.mp3', '.ogg', '.flac', '.WAV', '.MP3', '.OGG', '.FLAC']:
                                vehicle_name = vehicle_name.replace(ext, '')
                            vehicles.append({
                                'name': vehicle_name,
                                'filename': filename,
                                'duration': round(duration, 2),
                                'source': source_type,
                                'folder': folder
                            })
                        except Exception:
                            pass

        return jsonify({'vehicles': vehicles})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete_vehicle/<filename>', methods=['DELETE'])
def delete_vehicle(filename):
    """Delete a vehicle sound"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True})
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch_generate', methods=['POST'])
def batch_generate():
    """Generate batch of Doppler simulations"""
    try:
        config = request.get_json()

        start_time = time.time()

        # Validate configuration
        validation_error = validate_batch_config(config)
        if validation_error:
            return jsonify({'error': validation_error}), 400

        # Base output root (respect UI "Save Path" if provided)
        base_output_root = config.get('output', {}).get('path', OUTPUT_FOLDER)
        os.makedirs(base_output_root, exist_ok=True)

        # Create batch directory
        custom_name = config.get('batch', {}).get('name', '').strip()
        if custom_name:
            # Sanitize custom name
            safe_batch_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_batch_name = safe_batch_name.replace(' ', '_')
            batch_id = safe_batch_name
            batch_dir = os.path.join(base_output_root, batch_id)
        else:
            batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_dir = os.path.join(base_output_root, f'batch_{batch_id}')

        audio_dir = os.path.join(batch_dir, 'audio_clips')
        os.makedirs(audio_dir, exist_ok=True)

        # ------------------------------------------------------------
        # BATCH GENERATION - USE EXACT NUMBER FROM INPUT
        # ------------------------------------------------------------
        # Get the exact number of clips requested by the user
        total_clips = int(config['batch']['total_clips'])
        
        # Reset samplers for fresh uniform state space coverage each batch
        SAMPLERS.clear()
        save_progress(total_clips, 0)


        # Get distribution (per-vehicle & per-path counts)
        distribution = calculate_distribution(config, total_clips)
        vehicle_dist = distribution['vehicles']
        path_dist = distribution['paths']

        # Build flat lists of vehicles and paths with exact counts
        vehicle_list = []
        for v, count in vehicle_dist.items():
            vehicle_list.extend([v] * int(count))

        path_list = []
        for p, count in path_dist.items():
            path_list.extend([p] * int(count))

        # Helper to fix length to exactly total_clips
        def fix_list(lst, allowed_values):
            lst = list(lst)
            if not allowed_values:
                return lst
            if not lst:
                # If empty, fill with the first allowed value
                return [allowed_values[0]] * total_clips
            if len(lst) > total_clips:
                return lst[:total_clips]
            if len(lst) < total_clips:
                while len(lst) < total_clips:
                    lst.append(random.choice(allowed_values))
            return lst

        vehicle_list = fix_list(vehicle_list, list(vehicle_dist.keys()))
        path_list = fix_list(path_list, list(path_dist.keys()))

        # Now both lists have length == total_clips
        random.shuffle(vehicle_list)
        random.shuffle(path_list)

        clips_metadata = []
        generation_log = []
        clip_index = 1

        for i in range(total_clips):
            vehicle_name = vehicle_list[i]
            path_type = path_list[i]

            try:
                # Generate random parameters (respecting constraints)
                params = generate_random_parameters(config, vehicle_name, path_type)

                # Generate audio + save path plot
                result = generate_single_clip(
                    vehicle_name, path_type, params,
                    audio_dir, batch_id, clip_index, config
                )

                clips_metadata.append(result)
                generation_log.append(f"Generated clip {clip_index}/{total_clips}: {result['filename']}")
                print(f"Generated clip {clip_index}/{total_clips}")  # Print to terminal without filename
                save_progress(total_clips, clip_index)
                clip_index += 1

            except Exception as e:
                error_message = f"Error generating clip {clip_index}/{total_clips}: {str(e)}"
                generation_log.append(error_message)
                print(error_message)  # Print error to terminal
                save_progress(total_clips, clip_index)
                continue

        # Save metadata
        metadata_file = os.path.join(batch_dir, f'metadata_{batch_id}.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'batch_id': batch_id,
                'config': config,
                'clips': clips_metadata,
                'total_generated': len(clips_metadata),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        # Save log
        log_file = os.path.join(batch_dir, f'generation_log_{batch_id}.txt')
        with open(log_file, 'w') as f:
            f.write('\n'.join(generation_log))

        # Generate statistics
        stats_text = generate_statistics(clips_metadata, config)
        stats_file = os.path.join(batch_dir, f'statistics_{batch_id}.txt')
        with open(stats_file, 'w') as f:
            f.write(stats_text)
        
        elapsed_time = time.time() - start_time
        formatted_time = f"{elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
        print(f"Batch generation finished in {formatted_time}")

        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'total_generated': len(clips_metadata),
            'elapsed_time': elapsed_time,
            'formatted_time': formatted_time,
            'batch_directory': batch_dir,
            'metadata_file': metadata_file,
            'log_file': log_file,
            'stats_file': stats_file
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch_overlap_generate', methods=['POST'])
def batch_overlap_generate():
    """Generate scenes with multiple overlapping vehicles (busy road simulation) WITH VALIDATION"""
    try:
        config = request.get_json()
        start_time = time.time()
        
        # Root output directory
        base_output_root = config.get('output', {}).get('path', OUTPUT_FOLDER)
        custom_name = config.get('batch', {}).get('name', 'overlap_batch').strip()
        if not custom_name:
            custom_name = f"overlap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        safe_root_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        root_dir = os.path.join(base_output_root, safe_root_name)
        os.makedirs(root_dir, exist_ok=True)

        num_datasets = int(config.get('batch', {}).get('total_scenes', 10))
        vehicle_min = int(config.get('overlap', {}).get('vehicle_min', 1))
        vehicle_max = int(config.get('overlap', {}).get('vehicle_max', 20))
        lane_width = float(config.get('overlap', {}).get('lane_width', 4.0))
        max_stagger = float(config.get('overlap', {}).get('max_stagger', 5.0))
        include_opposite = config.get('overlap', {}).get('include_opposite', False)
        
        # NEW: Get validation settings from config (with defaults)
        enable_validation = config.get('validation', {}).get('enabled', True)
        validation_tolerance = float(config.get('validation', {}).get('tolerance', 0.5))
        
        selected_vehicles = config.get('vehicles', {}).get('selected', [])
        if not selected_vehicles:
            return jsonify({'error': 'No vehicles selected'}), 400
            
        selected_paths = config.get('paths', {}).get('selected', ['straight'])
        
        path_mixing_mode = config.get('overlap', {}).get('path_mixing_mode', 'same')
        
        SAMPLERS.clear()
        save_progress(num_datasets, 0)
        
        # NEW: Track validation statistics
        validation_stats = {
            'total_scenes': 0,
            'valid_scenes': 0,
            'invalid_scenes': 0,
            'total_vehicles': 0,
            'valid_vehicles': 0,
            'invalid_vehicles': 0
        }
        
        for scene_idx in range(1, num_datasets + 1):
            scene_id = f"{scene_idx:04d}"
            scene_dir = os.path.join(root_dir, scene_id)
            os.makedirs(scene_dir, exist_ok=True)
            
            num_vehicles = random.randint(vehicle_min, vehicle_max)
            
            # Determine path logic for this scene
            scene_path_mixing = path_mixing_mode
            if scene_path_mixing == 'both':
                scene_path_mixing = random.choice(['same', 'mixed'])
                
            scene_path_type = None
            if scene_path_mixing == 'same':
                scene_path_type = random.choice(selected_paths)
            
            clips_metadata = []
            clips_with_delays = []
            scene_paths_data = [] 
            
            for v_idx in range(1, num_vehicles + 1):
                vehicle_name = random.choice(selected_vehicles)
                
                # Assign path for this specific vehicle
                if scene_path_mixing == 'same':
                    path_type = scene_path_type
                else:
                    path_type = random.choice(selected_paths)
                
                # Base parameters for this vehicle
                params = generate_random_parameters(config, vehicle_name, path_type, force_symmetric=True)
                
                # CRITICAL: lane_width is the TOTAL road width for ALL vehicles
                # Randomly distribute vehicles within this total width (not evenly spaced)
                # If include_opposite is True, split lanes: 
                # Bottom half (negative offset) for forward traffic (left to right)
                # Top half (positive offset) for reverse traffic (right to left)
                is_opposite = False
                if include_opposite:
                    # Randomly decide direction (roughly equal spread)
                    is_opposite = (v_idx % 2 == 0)
                    if is_opposite:
                        lane_offset = random.uniform(0, lane_width / 2)
                    else:
                        lane_offset = random.uniform(-lane_width / 2, 0)
                else:
                    lane_offset = random.uniform(-lane_width / 2, lane_width / 2)
                
                # For reverse traffic, we need to flip the direction
                if is_opposite:
                    if path_type == 'straight':
                        params['angle'] = 180  # Opposite direction
                    elif path_type == 'parabola':
                        params['speed'] = -params['speed']  # Move right to left
                    elif path_type == 'bezier':
                        # Reverse the path by swapping endpoints and control points
                        params['x0'], params['x3'] = params['x3'], params['x0']
                        params['x1'], params['x2'] = params['x2'], params['x1']
                
                # For all path types, we want them to be as flat as possible
                # so the road width constraint is just about the y-position, not path curvature
                
                if path_type == 'straight':
                    # Force angle to 0 to keep the path horizontal at constant height
                    params['angle'] = 0
                    # Set distance to the lane offset (this is the y-coordinate)
                    # Add a base offset to ensure minimum distance from observer
                    params['distance'] = max(1.0, 5.0 + lane_offset)
                    
                elif path_type == 'parabola':
                    # Allow parabola to have some curvature, but only within this vehicle's lane slice
                    # Each vehicle gets lane_width/num_vehicles of vertical space for its curve
                    vehicle_lane_slice = lane_width / num_vehicles if num_vehicles > 1 else lane_width
                    
                    span = params['speed'] * params['duration']
                    x_max = span / 2.0
                    
                    # Limit curve to 80% of the vehicle's lane slice to leave some spacing
                    max_curve_height = vehicle_lane_slice * 0.8
                    max_a = max_curve_height / (x_max ** 2) if x_max > 0 else 0.0001
                    
                    # Use the original 'a' but clamp it to stay within bounds
                    params['a'] = min(params['a'], max_a)
                    
                    # Set center height to the lane offset with base offset
                    params['h'] = max(1.0, 5.0 + lane_offset)
                    params['distance'] = params['h']
                    
                elif path_type == 'bezier':
                    # Keep the bezier curve shape but constrain to this vehicle's lane slice
                    # Each vehicle gets lane_width/num_vehicles of vertical space
                    vehicle_lane_slice = lane_width / num_vehicles if num_vehicles > 1 else lane_width
                    
                    # Get the original y-coordinates (they have variation for curve shape)
                    y_coords = [params['y0'], params['y1'], params['y2'], params['y3']]
                    y_min = min(y_coords)
                    y_max = max(y_coords)
                    current_span = y_max - y_min
                    
                    # Limit curve to 80% of the vehicle's lane slice
                    max_curve_height = vehicle_lane_slice * 0.8
                    
                    if current_span > max_curve_height:
                        # Scale down the curve to fit within allowed height
                        scale_factor = max_curve_height / current_span
                        y_center = (y_min + y_max) / 2
                        params['y0'] = y_center + (params['y0'] - y_center) * scale_factor
                        params['y1'] = y_center + (params['y1'] - y_center) * scale_factor
                        params['y2'] = y_center + (params['y2'] - y_center) * scale_factor
                        params['y3'] = y_center + (params['y3'] - y_center) * scale_factor
                    
                    # Now shift the entire curve to the vehicle's lane position
                    base_height = 5.0 + lane_offset
                    y_coords_new = [params['y0'], params['y1'], params['y2'], params['y3']]
                    y_center_new = sum(y_coords_new) / 4
                    offset = base_height - y_center_new
                    
                    params['y0'] = max(1.0, params['y0'] + offset)
                    params['y1'] = max(1.0, params['y1'] + offset)
                    params['y2'] = max(1.0, params['y2'] + offset)
                    params['y3'] = max(1.0, params['y3'] + offset)
                
                delay = random.uniform(0, max_stagger)
                
                # Generate audio
                audio_arr, freq_ratios, amplitudes = get_doppler_audio_array(vehicle_name, path_type, params)
                
                # Save individual audio
                v_filename = f"vehicle_{v_idx:02d}_{vehicle_name}.wav"
                v_audio_path = os.path.join(scene_dir, v_filename)
                save_audio(audio_arr, v_audio_path)
                
                # NEW: Auto-generate spectrogram for individual car
                v_spec_path = v_audio_path.replace('.wav', '_spec.png')
                save_spectrogram_to_file(audio_arr, SR, f"Car {v_idx}: {vehicle_name}", v_spec_path)
                
                clips_with_delays.append((audio_arr, delay))
                scene_paths_data.append((path_type, params, vehicle_name))
                
                clips_metadata.append({
                    'id': v_idx,
                    'vehicle': vehicle_name,
                    'filename': v_filename,
                    'spectrogram': os.path.basename(v_spec_path),
                    'delay_s': delay,
                    'parameters': params
                })
            
            # NEW: VALIDATE PATHS BEFORE SAVING
            if enable_validation:
                validation_results = validate_scene_paths(
                    scenes_data=scene_paths_data,
                    lane_width=lane_width,
                    include_opposite=include_opposite,
                    tolerance=validation_tolerance,
                    y_shift=7.5
                )
                
                # Save validation reports
                save_validation_report(validation_results, scene_dir, scene_id)
                
                # Update statistics
                validation_stats['total_scenes'] += 1
                validation_stats['total_vehicles'] += validation_results['total_vehicles']
                validation_stats['valid_vehicles'] += (
                    validation_results['total_vehicles'] - 
                    validation_results['vehicles_with_violations']
                )
                validation_stats['invalid_vehicles'] += validation_results['vehicles_with_violations']
                
                if validation_results['scene_valid']:
                    validation_stats['valid_scenes'] += 1
                    print(f"✓ Scene {scene_id}: All {num_vehicles} vehicle paths valid")
                else:
                    validation_stats['invalid_scenes'] += 1
                    print(f"✗ Scene {scene_id}: {validation_results['vehicles_with_violations']}/{num_vehicles} "
                          f"vehicles have violations")
            else:
                print(f"Generated scene {scene_idx}/{num_datasets} (validation disabled)")
                
            # Mix and save combined audio
            mixed_audio = mix_audio_clips(clips_with_delays)
            mixed_audio_path = os.path.join(scene_dir, "mixed_audio.wav")
            save_audio(mixed_audio, mixed_audio_path)
            
            # NEW: Auto-generate spectrogram for mixed scene
            mixed_spec_path = os.path.join(scene_dir, "mixed_audio_spec.png")
            save_spectrogram_to_file(mixed_audio, SR, f"Mixed Scene: {scene_id} ({num_vehicles} cars)", mixed_spec_path)
            
            # Save combined plot
            save_combined_path_plot(scene_paths_data, scene_dir, "scene", lane_width=lane_width, include_opposite=include_opposite)
            
            # Save metadata (with validation results if enabled)
            metadata = {
                'scene_id': scene_id,
                'num_vehicles': num_vehicles,
                'path_type': scene_path_type if scene_path_mixing == 'same' else 'mixed',
                'vehicles': clips_metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            if enable_validation:
                metadata['validation'] = {
                    'enabled': True,
                    'scene_valid': validation_results['scene_valid'],
                    'vehicles_with_violations': validation_results['vehicles_with_violations']
                }
            
            with open(os.path.join(scene_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
                
            save_progress(num_datasets, scene_idx)

        elapsed_time = time.time() - start_time
        formatted_time = f"{elapsed_time:.2f}s"
        
        # NEW: Print validation summary
        if enable_validation:
            print("\n" + "=" * 70)
            print("BATCH VALIDATION SUMMARY")
            print("=" * 70)
            print(f"Total Scenes: {validation_stats['total_scenes']}")
            print(f"Valid Scenes: {validation_stats['valid_scenes']} "
                  f"({validation_stats['valid_scenes']/max(validation_stats['total_scenes'],1)*100:.1f}%)")
            print(f"Invalid Scenes: {validation_stats['invalid_scenes']} "
                  f"({validation_stats['invalid_scenes']/max(validation_stats['total_scenes'],1)*100:.1f}%)")
            print()
            print(f"Total Vehicles: {validation_stats['total_vehicles']}")
            print(f"Valid Vehicles: {validation_stats['valid_vehicles']} "
                  f"({validation_stats['valid_vehicles']/max(validation_stats['total_vehicles'],1)*100:.1f}%)")
            print(f"Invalid Vehicles: {validation_stats['invalid_vehicles']} "
                  f"({validation_stats['invalid_vehicles']/max(validation_stats['total_vehicles'],1)*100:.1f}%)")
            print("=" * 70)
            
            # Save batch validation summary
            summary_file = os.path.join(root_dir, "validation_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("BATCH VALIDATION SUMMARY\n")
                f.write("=" * 70 + "\n")
                f.write(f"Total Scenes: {validation_stats['total_scenes']}\n")
                f.write(f"Valid Scenes: {validation_stats['valid_scenes']} "
                       f"({validation_stats['valid_scenes']/max(validation_stats['total_scenes'],1)*100:.1f}%)\n")
                f.write(f"Invalid Scenes: {validation_stats['invalid_scenes']} "
                       f"({validation_stats['invalid_scenes']/max(validation_stats['total_scenes'],1)*100:.1f}%)\n")
                f.write(f"\nTotal Vehicles: {validation_stats['total_vehicles']}\n")
                f.write(f"Valid Vehicles: {validation_stats['valid_vehicles']} "
                       f"({validation_stats['valid_vehicles']/max(validation_stats['total_vehicles'],1)*100:.1f}%)\n")
                f.write(f"Invalid Vehicles: {validation_stats['invalid_vehicles']} "
                       f"({validation_stats['invalid_vehicles']/max(validation_stats['total_vehicles'],1)*100:.1f}%)\n")
        
        response_data = {
            'success': True,
            'batch_id': safe_root_name,
            'root_directory': root_dir,
            'total_generated': num_datasets,
            'elapsed_time': elapsed_time,
            'formatted_time': formatted_time
        }
        
        if enable_validation:
            response_data['validation'] = validation_stats
        
        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_spectrogram', methods=['POST'])
def generate_spectrogram():
    """Generate a spectrogram PNG for a given vehicle sound"""
    try:
        config = request.get_json()
        vehicle_name = config.get('vehicle_name')
        source = config.get('source', 'all')

        if not vehicle_name:
            return jsonify({'error': 'No vehicle name provided'}), 400

        # Find vehicle file
        vehicle_file = None
        folders_to_check = []
        if source == 'vehicle' or source == 'car':
            folders_to_check = [UPLOAD_FOLDER]
        elif source == 'drone':
            folders_to_check = [DRONE_SOUNDS_FOLDER]
        else:
            folders_to_check = [UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER]

        for folder in folders_to_check:
            for ext in ['.wav', '.mp3', '.ogg', '.flac']:
                test_path = os.path.join(folder, f'{vehicle_name}{ext}')
                if os.path.exists(test_path):
                    vehicle_file = test_path
                    break
            if vehicle_file:
                break

        if not vehicle_file:
            return jsonify({'error': f"Vehicle sound '{vehicle_name}' not found"}), 404

        # Load audio
        y, sr = librosa.load(vehicle_file, sr=SR)

        # Save to PNG
        file_id = f"{vehicle_name}_{int(time.time())}"
        plot_filename = f"spectrogram_{file_id}.png"
        plot_path = os.path.join(SPECTROGRAM_FOLDER, plot_filename)
        
        save_spectrogram_to_file(y, sr, f'Spectrogram: {vehicle_name}', plot_path)

        return jsonify({
            'success': True,
            'spectrogram_url': f'/static/spectrograms/{plot_filename}'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_generate_spectrogram', methods=['POST'])
def upload_generate_spectrogram():
    """Upload an audio file and generate a spectrogram"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save temporarily
        temp_filename = f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
        temp_path = os.path.join(SPECTROGRAM_FOLDER, temp_filename)
        file.save(temp_path)

        # Load and generate
        y, sr = librosa.load(temp_path, sr=SR)
        
        plot_filename = f"spectrogram_{int(time.time())}.png"
        plot_path = os.path.join(SPECTROGRAM_FOLDER, plot_filename)
        
        save_spectrogram_to_file(y, sr, f'Spectrogram: {file.filename}', plot_path)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'success': True,
            'spectrogram_url': f'/static/spectrograms/{plot_filename}'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get current generation progress"""
    return jsonify(load_progress())


@app.route('/simulate', methods=['POST'])
def simulate_single():
    """
    Single-clip Doppler simulation endpoint for the single-clip UI.
    Returns a WAV file blob that the frontend plays directly.
    """
    try:
        # Basic inputs from form
        path_type = request.form.get('path', 'straight')
        vehicle_type = request.form.get('vehicle_type', 'car')

        # FORCE all single-clip simulations to 10 seconds
        duration = 10.0

        # Use lower-case name to match uploaded vehicle files (car.wav, train.wav, etc.)
        vehicle_name = vehicle_type.lower()

        # Common parameters
        params = {
            'duration': duration
        }

        # Path-specific parameters (manual mode – you control signs here)
        if path_type == 'straight':
            speed = float(request.form.get('speed', 20.0))
            h = float(request.form.get('h', 10.0))       # closest distance
            angle = float(request.form.get('angle', 0.0))

            params['speed'] = speed
            params['distance'] = h
            params['angle'] = angle

        elif path_type == 'parabola':
            speed = float(request.form.get('speed', 15.0))
            a = float(request.form.get('a', 0.1))
            h = float(request.form.get('h', 10.0))

            params['speed'] = speed
            params['a'] = a
            params['h'] = h
            # store something reasonable for filename/stats distance
            params['distance'] = h

        elif path_type == 'bezier':
            speed = float(request.form.get('speed', 20.0))

            params['speed'] = speed
            params['x0'] = float(request.form.get('x0', -30))
            params['y0'] = float(request.form.get('y0', 20))
            params['x1'] = float(request.form.get('x1', -10))
            params['y1'] = float(request.form.get('y1', 5))
            params['x2'] = float(request.form.get('x2', 10))
            params['y2'] = float(request.form.get('y2', 5))
            params['x3'] = float(request.form.get('x3', 30))
            params['y3'] = float(request.form.get('y3', 20))
            # nominal distance just for filename
            params['distance'] = 10.0

        else:
            return jsonify({'error': f'Unknown path type: {path_type}'}), 400

        # Minimal config reused from batch code
        config = {
            'output': {'format': 'wav'}
        }

        single_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        index = 1

        result = generate_single_clip(
            vehicle_name=vehicle_name,
            path_type=path_type,
            params=params,
            output_dir=SINGLE_OUTPUT_FOLDER,
            batch_id=single_id,
            index=index,
            config=config
        )

        file_path = os.path.join(SINGLE_OUTPUT_FOLDER, result['filename'])
        if not os.path.exists(file_path):
            return jsonify({'error': 'Audio generation failed - output file not created'}), 500

        return send_file(file_path, mimetype='audio/wav')

    except FileNotFoundError as e:
        return jsonify({'error': f'Audio file not found: {str(e)}'}), 404
    except ValueError as e:
        return jsonify({'error': f'Invalid parameter value: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Simulation error: {str(e)}'}), 500


def validate_batch_config(config):
    """Validate batch configuration (GLOBAL target only)"""
    batch = config.get('batch', {})
    total_clips = batch.get('total_clips')

    if not total_clips or total_clips < 1:
        return "Total clips must be at least 1"

    vehicles = config.get('vehicles', {}).get('selected', [])
    if not vehicles:
        return "No vehicles selected"

    paths = config.get('paths', {}).get('selected', [])
    if not paths:
        return "No path types selected"

    # ❗ DO NOT validate distribution totals anymore
    # because batching is continuous

    return None


def calculate_distribution(config, current_batch_size):
    """Calculate vehicle and path distribution for THIS batch"""
    total_clips = current_batch_size
    mode = config['batch'].get('mode', 'auto')

    if mode == 'manual':
        return config['batch']['distribution']

    vehicles = config['vehicles']['selected']
    paths = config['paths']['selected']

    clips_per_vehicle = total_clips // len(vehicles)
    clips_per_path = total_clips // len(paths)

    vehicle_dist = {v: clips_per_vehicle for v in vehicles}
    path_dist = {p: clips_per_path for p in paths}

    for i in range(total_clips % len(vehicles)):
        vehicle_dist[vehicles[i]] += 1

    for i in range(total_clips % len(paths)):
        path_dist[paths[i]] += 1

    return {
        'vehicles': vehicle_dist,
        'paths': path_dist
    }


# ============================================================
# FAST INTEGER CYCLIC SAMPLER (O(1), FULL COVERAGE)
# ============================================================

class CyclicIntegerSampler:
    def __init__(self, low, high, seed=None):
        self.low = int(low)
        self.high = int(high)
        self.range = self.high - self.low + 1
        self.k = 0

        if self.range <= 1:
            self.step = 1
        else:
            # pick step coprime with range
            self.step = random.choice(
                [s for s in range(1, self.range) if np.gcd(s, self.range) == 1]
            )

        self.offset = random.randint(0, self.range - 1)

    def next(self):
        val = self.low + (self.offset + self.k * self.step) % self.range
        self.k += 1
        return int(val)



def generate_random_parameters(config, vehicle_name, path_type, force_symmetric=False):
    params = {}

    def get_sampler(key, lo, hi):
        if key not in SAMPLERS:
            SAMPLERS[key] = CyclicIntegerSampler(lo, hi)
        return SAMPLERS[key].next()

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # -------- SPEED --------
    gmin, gmax = DEFAULT_RANGES['speed'].get(
        vehicle_name.lower(),
        DEFAULT_RANGES['speed']['default']
    )

    if config.get('speed', {}).get('randomize', True):
        umin = int(config['speed'].get('min', gmin))
        umax = int(config['speed'].get('max', gmax))
        lo = clamp(umin, gmin, gmax)
        hi = clamp(umax, gmin, gmax)
        if lo > hi:
            lo, hi = hi, lo
        params['speed'] = get_sampler(f"speed_{vehicle_name}", lo, hi)
    else:
        params['speed'] = clamp(int(config['speed'].get('value', 30)), gmin, gmax)

    # -------- DISTANCE --------
    dmin, dmax = DEFAULT_RANGES['distance']

    if config.get('distance', {}).get('randomize', True):
        umin = int(config['distance'].get('min', dmin))
        umax = int(config['distance'].get('max', dmax))
        lo = clamp(umin, dmin, dmax)
        hi = clamp(umax, dmin, dmax)
        if lo > hi:
            lo, hi = hi, lo
        params['distance'] = get_sampler("distance", lo, hi)
    else:
        params['distance'] = clamp(int(config['distance'].get('value', 30)), dmin, dmax)

    # -------- FIXED DURATION --------
    params['duration'] = 10.0

    # -------- STRAIGHT --------
    if path_type == 'straight':
        amin, amax = DEFAULT_RANGES['angle']
        if config.get('angle', {}).get('randomize', True):
            umin = int(config['angle'].get('min', amin))
            umax = int(config['angle'].get('max', amax))
            lo = clamp(umin, amin, amax)
            hi = clamp(umax, amin, amax)
            if lo > hi:
                lo, hi = hi, lo
            params['angle'] = get_sampler("angle", lo, hi)
        else:
            params['angle'] = clamp(int(config['angle'].get('value', 0)), amin, amax)

    # -------- PARABOLA --------
    elif path_type == 'parabola':
        a_lo, a_hi = DEFAULT_RANGES['parabola_a']
        h_lo, h_hi = DEFAULT_RANGES['parabola_h']

        a_int = get_sampler("parabola_a", a_lo, a_hi)
        params['a'] = a_int / 10000.0
        params['h'] = get_sampler("parabola_h", h_lo, h_hi)

    # -------- BEZIER --------
    elif path_type == 'bezier':
        cmin, cmax = DEFAULT_RANGES['bezier_coords']

        if force_symmetric:
            # Symmetrize around Y-axis based on span
            span = params['speed'] * params['duration']
            x0 = -0.5 * span
            x3 = 0.5 * span
            # Randomize x1, x2 between x0 and x3
            x1 = get_sampler("bx1_sym", int(x0), int(x3))
            x2 = get_sampler("bx2_sym", int(x0), int(x3))
            params['x0'], params['x1'], params['x2'], params['x3'] = sorted([x0, x1, x2, x3])
        else:
            xs = sorted([
                get_sampler("bx0", cmin, cmax),
                get_sampler("bx1", cmin, cmax),
                get_sampler("bx2", cmin, cmax),
                get_sampler("bx3", cmin, cmax),
            ])
            for i in range(1, 4):
                if xs[i] <= xs[i - 1]:
                    xs[i] = xs[i - 1] + 1
            params['x0'], params['x1'], params['x2'], params['x3'] = xs

        params['y0'] = get_sampler("by0", 5, 80)
        params['y1'] = get_sampler("by1", 5, 80)
        params['y2'] = get_sampler("by2", 5, 80)
        params['y3'] = get_sampler("by3", 5, 80)

    return params

def compute_path_points(path_type, params, n_points=200):
    """Compute (x, y) path points for plotting"""
    if path_type == 'straight':
        duration = params['duration']
        v = params['speed']
        h = params['distance']
        angle = params.get('angle', 0.0)

        t = np.linspace(0.0, duration, n_points)
        t0 = duration / 2.0
        dt = t - t0

        theta = np.deg2rad(angle)
        u = np.array([np.cos(theta), np.sin(theta)])
        n = np.array([-np.sin(theta), np.cos(theta)])

        p_c = h * n
        v_vec = u * v
        p = p_c[:, None] + v_vec[:, None] * dt[None, :]

        x = p[0, :]
        y = p[1, :]

        cx, cy = p_c
        return x, y, (cx, cy)

    elif path_type == 'parabola':
        duration = params['duration']
        v = params['speed']
        a = params['a']
        h = params['h']

        t = np.linspace(0.0, duration, n_points)
        t0 = duration / 2.0
        dt = t - t0

        x = v * dt
        y = a * x**2 + h

        return x, y, None

    elif path_type == 'bezier':
        x0 = float(params['x0'])
        x1 = float(params['x1'])
        x2 = float(params['x2'])
        x3 = float(params['x3'])
        y0 = float(params['y0'])
        y1 = float(params['y1'])
        y2 = float(params['y2'])
        y3 = float(params['y3'])

        u = np.linspace(0.0, 1.0, n_points)
        x = ((1 - u) ** 3) * x0 + 3 * ((1 - u) ** 2) * u * x1 + 3 * (1 - u) * (u ** 2) * x2 + (u ** 3) * x3
        y = ((1 - u) ** 3) * y0 + 3 * ((1 - u) ** 2) * u * y1 + 3 * (1 - u) * (u ** 2) * y2 + (u ** 3) * y3

        return x, y, None

    else:
        # fallback: trivial horizontal line
        x = np.linspace(-10, 10, n_points)
        y = np.zeros_like(x)
        return x, y, None


def save_path_plot(path_type, params, output_dir, base_name):
    """
    Save a PNG path graph for this clip.
    Axis scale and grid are kept.
    Axis labels, legend, and title are removed.
    """
    try:
        x, y, closest = compute_path_points(path_type, params, n_points=200)
        plot_path = os.path.join(output_dir, f"{base_name}_path.png")

        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        # Path
        ax.plot(x, y, linewidth=2, antialiased=False)

        # Start / end points
        ax.scatter([x[0]], [y[0]], s=40)
        ax.scatter([x[-1]], [y[-1]], s=40)

        # Observer at origin
        ax.scatter([0], [0], marker='x', s=50)

        # Closest-distance line (straight only)
        if closest is not None:
            cx, cy = closest
            ax.plot([0, cx], [0, cy], linestyle='--', linewidth=1, antialiased=False)

        # Keep axis scale and grid, remove only text labels
        ax.axis('equal')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.grid(True, which="major", linestyle='--', alpha=0.4)

        # Save
        fig.savefig(
            plot_path,
            dpi=100,
            bbox_inches="tight",
            pil_kwargs={"compress_level": 1}
        )
        plt.close(fig)

        return os.path.basename(plot_path)

    except Exception as e:
        print(f"Failed to save path plot for {base_name}: {e}")
        return None


def save_combined_path_plot(scenes_data, output_dir, base_name, **kwargs):
    """
    Save a PNG graph with all vehicle paths in a scene.
    Now draws curved road boundaries that follow the average path trajectory.
    Graph size is dynamically calculated based on the span of the paths.
    Observer is positioned outside (below) the road boundaries.
    """
    try:
        plot_path = os.path.join(output_dir, f"{base_name}_combined_path.png")
        
        # Collect all vehicle paths to compute road centerline and spatial ranges
        all_paths = []
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        
        for i, (path_type, params, vehicle_name) in enumerate(scenes_data):
            x, y, _ = compute_path_points(path_type, params, n_points=200)
            all_paths.append((x, y))
            x_min = min(x_min, np.min(x))
            x_max = max(x_max, np.max(x))
            y_min = min(y_min, np.min(y))
            y_max = max(y_max, np.max(y))

        # Dynamic figsize calculation - Make it SQUARE
        x_span = x_max - x_min
        num_vehicles = len(scenes_data)
        
        # Base dimensions and scaling
        # Choose a square size that accommodates the span and vehicle count
        # Each vehicle needs room, and long spans need width. 
        # We take the larger of the two required dimensions to keep it square.
        base_size = max(10, min(24, x_span / 15.0, num_vehicles * 1.5 + 4))
        fig, ax = plt.subplots(figsize=(base_size, base_size))

        # Road details from kwargs
        lane_width = kwargs.get('lane_width', 4.0)
        include_opposite = kwargs.get('include_opposite', False)
        
        # Compute average centerline by interpolating all paths on a common x-grid
        x_common = np.linspace(x_min, x_max, 300)
        y_interpolated = []
        
        for x, y in all_paths:
            # Sort by x to ensure monotonic for interpolation
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            
            # Interpolate this path onto common x-grid
            y_interp = np.interp(x_common, x_sorted, y_sorted, left=y_sorted[0], right=y_sorted[-1])
            y_interpolated.append(y_interp)
        
        # Calculate envelopes and group paths to prevent median crossings
        avg_y_per_path = [np.mean(y) for y in y_interpolated]
        
        # Split into two EQUAL clusters (balanced lanes) using median
        if len(y_interpolated) > 1:
            # Sort by average height and split exactly in the middle
            sorted_indices = np.argsort(avg_y_per_path)
            num_cars = len(sorted_indices)
            split_idx = num_cars // 2  # This ensures balanced grouping e.g. 3 vs 3
            
            lower_indices = sorted_indices[:split_idx]
            upper_indices = sorted_indices[split_idx:]
            
            # Find the max of the lower group and min of the upper group at each x
            y_lower_max = np.max([y_interpolated[i] for i in lower_indices], axis=0)
            y_upper_min = np.min([y_interpolated[i] for i in upper_indices], axis=0)
            
            # Centerline is exactly in the gap between the two clusters
            y_centerline = (y_lower_max + y_upper_min) / 2
        else:
            y_centerline = np.mean(y_interpolated, axis=0)
        
        # Global envelopes for road width
        y_min_env = np.min(y_interpolated, axis=0)
        y_max_env = np.max(y_interpolated, axis=0)
        
        # Use the provided lane_width, ensure it fits the cars but don't add massive hardcoded margins
        actual_spread = np.max(y_max_env - y_min_env)
        lane_width = max(lane_width, actual_spread + 2.0) # Small 2m buffer instead of 6m
        
        # Smooth the centerline for better road appearance
        y_centerline_smooth = gaussian_filter1d(y_centerline, sigma=10)
        
        # Compute road boundaries (offset perpendicular to centerline)
        y_upper_boundary = y_centerline_smooth + lane_width / 2
        y_lower_boundary = y_centerline_smooth - lane_width / 2
        
        # Apply transformation: Shift road so the lower boundary is at a safe distance from observer
        # Original observer is at 0. If lower boundary is at y_lower, shift it to y=2 or similar
        y_shift = -np.min(y_lower_boundary) + 5.0 # Ensure road is at least 5m away from observer
        y_upper_boundary += y_shift
        y_lower_boundary += y_shift
        y_centerline_smooth += y_shift
        
        # Draw road boundaries (RED DOTTED lines)
        ax.plot(x_common, y_upper_boundary, color='red', linestyle=':', 
               linewidth=2.5, alpha=0.9, label='Road Edge')
        ax.plot(x_common, y_lower_boundary, color='red', linestyle=':', 
               linewidth=2.5, alpha=0.9)
        
        # Draw median if opposite traffic
        if include_opposite:
            ax.plot(x_common, y_centerline_smooth, color='yellow', linestyle='--', 
                   linewidth=2, alpha=0.8, label='Center Line')
        
        # Now draw vehicle paths on top of the road
        for i, (path_type, params, vehicle_name) in enumerate(scenes_data):
            x, y, _ = compute_path_points(path_type, params, n_points=200)
            y_shifted = y + y_shift
            ax.plot(x, y_shifted, linewidth=2.5, label=f"V{i+1}: {vehicle_name}", alpha=0.9)
            ax.scatter([x[0]], [y_shifted[0]], s=50, zorder=5)
            ax.scatter([x[-1]], [y_shifted[-1]], s=50, zorder=5)
 
        # Observer at origin (0, 0)
        ax.scatter([0], [0], marker='x', s=120, color='red', label='Observer', zorder=10, linewidth=3)
 
        # Set axis limits with some padding
        y_min_plot = min(0, np.min(y_lower_boundary)) - 5
        y_max_plot = np.max(y_upper_boundary) + 5
        ax.set_ylim(y_min_plot, y_max_plot)
        ax.set_xlim(x_min - 20, x_max + 20)
        
        ax.set_xlabel("x (meters)", fontsize=16)
        ax.set_ylabel("y (meters)", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        ax.grid(True, which="major", linestyle='--', alpha=0.3)
        ax.set_aspect('auto')
        ax.grid(True, which="major", linestyle='--', alpha=0.3)

        fig.savefig(
            plot_path,
            dpi=150,
            bbox_inches="tight"
        )
        plt.close(fig)
        return os.path.basename(plot_path)
    except Exception as e:
        print(f"Failed to save combined path plot: {e}")
        traceback.print_exc()
        return None


def save_spectrogram_to_file(y, sr, title, out_path):
    """
    Generate and save a high-resolution spectrogram PNG to a specific path.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # High resolution: n_fft=4096, hop_length=256
        stft = librosa.stft(y, n_fft=4096, hop_length=256)
        D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, hop_length=256)
        ax.set_ylim(0, 2500) # Zoom in to 0-2500 Hz
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')

        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Failed to save spectrogram to {out_path}: {e}")
        return False


def get_doppler_audio_array(vehicle_name, path_type, params):
    """
    Core logic to generate Doppler-shifted audio array.
    """
    # Load vehicle audio
    vehicle_file = None
    folders_to_check = [UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER]
    
    for folder in folders_to_check:
        for ext in ['.wav', '.mp3', '.ogg', '.flac']:
            test_path = os.path.join(folder, f'{vehicle_name}{ext}')
            if os.path.exists(test_path):
                vehicle_file = test_path
                break
        if vehicle_file:
            break

    if not vehicle_file:
        for folder in folders_to_check:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    if filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                        vehicle_file = os.path.join(folder, filename)
                        break
            if vehicle_file:
                break
    
    if not vehicle_file:
        raise FileNotFoundError(f"No audio files found for vehicle '{vehicle_name}'")

    audio_full, sr = librosa.load(vehicle_file, sr=SR, mono=True)
    target_samples = int(SR * params['duration'])

    if len(audio_full) < target_samples:
        from audio_utils import extend_audio_with_overlap
        audio = extend_audio_with_overlap(audio_full, params['duration'], SR)
    else:
        audio = audio_full[:target_samples]

    if path_type == 'straight':
        freq_ratios, amplitudes = calculate_straight_line_doppler(
            params['speed'], params['distance'], params.get('angle', 0), params['duration']
        )
    elif path_type == 'parabola':
        freq_ratios, amplitudes = calculate_parabola_doppler(
            params['speed'], params['a'], params['h'], params['duration']
        )
    elif path_type == 'bezier':
        freq_ratios, amplitudes = calculate_bezier_doppler(
            params['speed'], params['x0'], params['x1'], params['x2'], params['x3'],
            params['y0'], params['y1'], params['y2'], params['y3'], params['duration']
        )
    else:
        freq_ratios = np.ones(target_samples)
        amplitudes = np.ones(target_samples)

    doppler_audio = apply_doppler_to_audio_fixed(audio, freq_ratios, amplitudes)

    # Ensure exact length
    if len(doppler_audio) > target_samples:
        doppler_audio = doppler_audio[:target_samples]
    elif len(doppler_audio) < target_samples:
        padded = np.zeros(target_samples)
        padded[:len(doppler_audio)] = doppler_audio
        doppler_audio = padded

    return doppler_audio, freq_ratios, amplitudes


def generate_single_clip(vehicle_name, path_type, params, output_dir, batch_id, index, config):
    """Generate a single clip and save files"""
    doppler_audio, freq_ratios, amplitudes = get_doppler_audio_array(vehicle_name, path_type, params)

    output_format = config.get('output', {}).get('format', 'wav')
    base_name = f"{vehicle_name}_{path_type}_{params['speed']}mps_{params['distance']}m_{index:07d}"
    filename = f"{base_name}.{output_format}"
    filepath = os.path.join(output_dir, filename)

    if output_format == 'mp3':
        wav_path = filepath.replace('.mp3', '_temp.wav')
        save_audio(doppler_audio, wav_path)
        os.rename(wav_path, filepath.replace('.mp3', '.wav'))
        filename = filename.replace('.mp3', '.wav')
    else:
        save_audio(doppler_audio, filepath)

    path_plot_filename = save_path_plot(path_type, params, output_dir, base_name)

    return {
        'filename': filename,
        'index': index,
        'vehicle': vehicle_name,
        'path_type': path_type,
        'parameters': params,
        'freq_ratio_range': {
            'min': float(np.min(freq_ratios)),
            'max': float(np.max(freq_ratios))
        },
        'path_plot': path_plot_filename
    }


def mix_audio_clips(clips_with_delays):
    """Mix multiple audio arrays with staggered start times"""
    if not clips_with_delays:
        return np.array([])
    
    max_end_sample = 0
    for audio, delay_s in clips_with_delays:
        delay_samples = int(delay_s * SR)
        end_sample = delay_samples + len(audio)
        if end_sample > max_end_sample:
            max_end_sample = end_sample
            
    mixed = np.zeros(max_end_sample)
    for audio, delay_s in clips_with_delays:
        delay_samples = int(delay_s * SR)
        mixed[delay_samples : delay_samples + len(audio)] += audio
        
    # peak normalization
    max_val = np.max(np.abs(mixed))
    if max_val > 0.99:
        mixed = mixed / max_val * 0.9
        
    return mixed


def generate_statistics(clips_metadata, config):
    """Generate statistics summary"""
    stats = []
    stats.append("=" * 60)
    stats.append("BATCH GENERATION STATISTICS")
    stats.append("=" * 60)
    stats.append("")

    stats.append(f"Total Clips Generated: {len(clips_metadata)}")
    stats.append("")

    # Vehicle distribution
    stats.append("Vehicle Distribution:")
    vehicles = {}
    for clip in clips_metadata:
        v = clip['vehicle']
        vehicles[v] = vehicles.get(v, 0) + 1
    for v, count in sorted(vehicles.items()):
        stats.append(f"  {v}: {count} clips ({count / len(clips_metadata) * 100:.1f}%)")
    stats.append("")

    # Path distribution
    stats.append("Path Type Distribution:")
    paths = {}
    for clip in clips_metadata:
        p = clip['path_type']
        paths[p] = paths.get(p, 0) + 1
    for p, count in sorted(paths.items()):
        stats.append(f"  {p}: {count} clips ({count / len(clips_metadata) * 100:.1f}%)")
    stats.append("")

    # Speed statistics
    speeds = [clip['parameters']['speed'] for clip in clips_metadata]
    stats.append("Speed Statistics:")
    stats.append(f"  Min: {min(speeds)} m/s")
    stats.append(f"  Max: {max(speeds)} m/s")
    stats.append(f"  Mean: {np.mean(speeds):.1f} m/s")
    stats.append(f"  Median: {np.median(speeds):.1f} m/s")
    stats.append("")

    # Distance statistics
    distances = [clip['parameters']['distance'] for clip in clips_metadata]
    stats.append("Distance Statistics:")
    stats.append(f"  Min: {min(distances)} m")
    stats.append(f"  Max: {max(distances)} m")
    stats.append(f"  Mean: {np.mean(distances):.1f} m")
    stats.append(f"  Median: {np.median(distances):.1f} m")
    stats.append("")

    # Duration statistics
    durations = [clip['parameters']['duration'] for clip in clips_metadata]
    stats.append("Duration Statistics:")
    stats.append(f"  Min: {min(durations)} s")
    stats.append(f"  Max: {max(durations)} s")
    stats.append(f"  Mean: {np.mean(durations):.1f} s")
    stats.append("")
    stats.append("=" * 60)

    return '\n'.join(stats)


if __name__ == '__main__':
    print("=" * 60)
    print("Doppler Effect Batch Simulator (WITH PATH VALIDATION)")
    print("=" * 60)
    print(f"Vehicle sounds folder: {UPLOAD_FOLDER}")
    print(f"Batch output folder (default root): {OUTPUT_FOLDER}")
    print(f"Single-clip output folder: {SINGLE_OUTPUT_FOLDER}")
    print(f"Server starting on http://0.0.0.0:5050")
    print("=" * 60)
    print("\nNEW FEATURES:")
    print("  ✓ Path validation enabled by default")
    print("  ✓ Detects road boundary violations")
    print("  ✓ Detects median/centerline crossings")
    print("  ✓ Generates validation reports (JSON + TXT)")
    print("  ✓ Batch validation statistics")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5050)


# Reset everything (when needed)
# Delete these two files:
# sampler_state.json
# generation_progress.json