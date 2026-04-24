import os
import numpy as np
import librosa
import traceback
from datetime import datetime

from flask import Blueprint, request, jsonify, send_file

from audio.audio_utils import (
    apply_doppler_to_audio_fixed,
    save_audio,
    extend_audio_with_overlap,
    save_stereo_audio,
    SR
)
from physics.straight_line import calculate_straight_line_doppler, calculate_straight_line_doppler_dual
from physics.parabola import calculate_parabola_doppler, calculate_parabola_doppler_dual
from physics.bezier import calculate_bezier_doppler, calculate_bezier_doppler_dual
from core.config import UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER, SINGLE_OUTPUT_FOLDER
from audio.generation import generate_single_clip

simulate_bp = Blueprint('simulate', __name__)


@simulate_bp.route('/simulate', methods=['POST'])
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


@simulate_bp.route('/simulate_dual', methods=['POST'])
def simulate_dual():
    """
    Dual microphone Doppler simulation endpoint.
    Returns JSON with paths to stereo file and two mono files, plus physics data for both mics.
    """
    try:
        # Basic inputs from form
        path_type = request.form.get('path', 'straight')
        vehicle_type = request.form.get('vehicle_type', 'car')
        mic_separation = float(request.form.get('mic_separation', 10.0))

        # Validate microphone separation
        if mic_separation <= 0:
            return jsonify({'error': 'Microphone separation must be greater than 0'}), 400

        # FORCE all single-clip simulations to 10 seconds
        duration = 10.0

        # Use lower-case name to match uploaded vehicle files
        vehicle_name = vehicle_type.lower()

        # Common parameters
        params = {
            'duration': duration,
            'mic_separation': mic_separation
        }

        # Path-specific parameters
        if path_type == 'straight':
            speed = float(request.form.get('speed', 20.0))
            h = float(request.form.get('h', 10.0))
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
            params['distance'] = 10.0

        else:
            return jsonify({'error': f'Unknown path type: {path_type}'}), 400

        # Generate dual microphone audio
        single_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

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
            return jsonify({'error': f'Audio file for {vehicle_name} not found'}), 404

        audio_full, sr = librosa.load(vehicle_file, sr=SR, mono=True)
        audio = extend_audio_with_overlap(audio_full, duration, SR)

        # Calculate dual microphone physics
        if path_type == 'straight':
            dual_data = calculate_straight_line_doppler_dual(
                params['speed'], params['distance'], params.get('angle', 0),
                duration, mic_separation
            )
        elif path_type == 'parabola':
            dual_data = calculate_parabola_doppler_dual(
                params['speed'], params['a'], params['h'], duration, mic_separation
            )
        elif path_type == 'bezier':
            dual_data = calculate_bezier_doppler_dual(
                params['speed'], params['x0'], params['x1'], params['x2'], params['x3'],
                params['y0'], params['y1'], params['y2'], params['y3'],
                duration, mic_separation
            )
        else:
            return jsonify({'error': 'Invalid path type'}), 400

        # Generate audio for both microphones
        target_samples = int(SR * duration)

        audio_m1 = apply_doppler_to_audio_fixed(
            audio, dual_data['m1']['freq_ratios'], dual_data['m1']['amplitudes']
        )
        audio_m2 = apply_doppler_to_audio_fixed(
            audio, dual_data['m2']['freq_ratios'], dual_data['m2']['amplitudes']
        )

        # Ensure exact length for both
        audio_m1 = audio_m1[:target_samples] if len(audio_m1) > target_samples else np.pad(audio_m1, (0, target_samples - len(audio_m1)))
        audio_m2 = audio_m2[:target_samples] if len(audio_m2) > target_samples else np.pad(audio_m2, (0, target_samples - len(audio_m2)))

        # Save files
        base_name = f"{vehicle_name}_{path_type}_dual_{single_id}"

        m1_filename = f"{base_name}_M1.wav"
        m2_filename = f"{base_name}_M2.wav"
        stereo_filename = f"{base_name}_stereo.wav"

        m1_path = os.path.join(SINGLE_OUTPUT_FOLDER, m1_filename)
        m2_path = os.path.join(SINGLE_OUTPUT_FOLDER, m2_filename)
        stereo_path = os.path.join(SINGLE_OUTPUT_FOLDER, stereo_filename)

        save_audio(audio_m1, m1_path)
        save_audio(audio_m2, m2_path)
        save_stereo_audio(audio_m1, audio_m2, stereo_path)

        # Prepare response with physics data
        response = {
            'success': True,
            'files': {
                'm1': f'/static/single_outputs/{m1_filename}',
                'm2': f'/static/single_outputs/{m2_filename}',
                'stereo': f'/static/single_outputs/{stereo_filename}'
            },
            'physics': {
                'm1': {
                    'distances': dual_data['m1']['distances'].tolist(),
                    'velocities': dual_data['m1']['velocities'].tolist(),
                    'freq_ratios': dual_data['m1']['freq_ratios'].tolist()
                },
                'm2': {
                    'distances': dual_data['m2']['distances'].tolist(),
                    'velocities': dual_data['m2']['velocities'].tolist(),
                    'freq_ratios': dual_data['m2']['freq_ratios'].tolist()
                }
            },
            'mic_positions': {
                'm1': [-mic_separation / 2.0, 0.0],
                'm2': [mic_separation / 2.0, 0.0]
            }
        }

        return jsonify(response)

    except FileNotFoundError as e:
        return jsonify({'error': f'Audio file not found: {str(e)}'}), 404
    except ValueError as e:
        return jsonify({'error': f'Invalid parameter value: {str(e)}'}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Simulation error: {str(e)}'}), 500

@simulate_bp.route('/api/simulate_intersection', methods=['POST'])
def simulate_intersection():
    """
    Multi-vehicle intersection Doppler simulation endpoint.
    Receives JSON with intersection layout and vehicle list.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing JSON body'}), 400

        vehicles_config = data.get('vehicles', [])
        intersection_cfg = data.get('intersection', {})
        
        obs_pos = (
            float(intersection_cfg.get('obs_x', 10.0)),
            float(intersection_cfg.get('obs_y', 10.0))
        )
        duration = float(data.get('duration', 10.0))
        c_sound = float(data.get('c_sound', 343.0))

        if not vehicles_config:
            return jsonify({'error': 'No vehicles provided'}), 400

        from physics.intersection import calculate_intersection_doppler
        from audio.generation import mix_audio_clips
        from audio.audio_utils import extend_audio_with_overlap

        # Calculate physics for all vehicles
        physics_results = calculate_intersection_doppler(
            vehicles_config,
            observer_pos=obs_pos,
            duration_s=duration,
            c_sound=c_sound
        )

        mixed_clips = []
        vehicle_meta = {}

        for v_cfg in vehicles_config:
            v_id = v_cfg['id']
            v_type = v_cfg.get('type', 'car').lower()
            
            # Find audio file
            vehicle_file = None
            folders_to_check = [UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER]
            
            # 1. Exact match
            for folder in folders_to_check:
                for ext in ['.wav', '.mp3']:
                    test_path = os.path.join(folder, f'{v_type}{ext}')
                    if os.path.exists(test_path):
                        vehicle_file = test_path
                        break
                if vehicle_file: break
            
            # 2. Starts with (e.g. car_1.wav for type='car')
            if not vehicle_file:
                for folder in folders_to_check:
                    if os.path.exists(folder):
                        files = [f for f in os.listdir(folder) if f.lower().startswith(v_type)]
                        if files:
                            vehicle_file = os.path.join(folder, files[0])
                            break
            
            # 3. Global fallback
            if not vehicle_file:
                for folder in folders_to_check:
                    if os.path.exists(folder):
                        files = [f for f in os.listdir(folder) if f.lower().endswith(('.wav', '.mp3'))]
                        if files:
                            vehicle_file = os.path.join(folder, files[0])
                            break

            if not vehicle_file:
                continue

            # Load and process audio
            audio_full, sr = librosa.load(vehicle_file, sr=SR, mono=True)
            audio = extend_audio_with_overlap(audio_full, duration * 2.0, SR)
            
            v_physics = physics_results[v_id]
            doppler_audio = apply_doppler_to_audio_fixed(
                audio, v_physics['freq_ratios'], v_physics['amplitudes']
            )
            
            # Ensure exact length
            target_samples = int(SR * duration)
            if len(doppler_audio) > target_samples:
                doppler_audio = doppler_audio[:target_samples]
            else:
                doppler_audio = np.pad(doppler_audio, (0, target_samples - len(doppler_audio)))
                
            mixed_clips.append((doppler_audio, 0.0)) # All vehicles share the same timeline
            
            # Prepare metadata for frontend visualization
            vehicle_meta[v_id] = {
                'positions': v_physics['positions'].tolist(), # [ [x...], [y...] ]
                'freq_ratios': v_physics['freq_ratios'].tolist(),
                'type': v_type
            }

        if not mixed_clips:
            return jsonify({'error': 'Failed to generate any vehicle audio'}), 500

        # Mix all clips
        final_audio = mix_audio_clips(mixed_clips)
        
        # Save result
        sim_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"intersection_{sim_id}.wav"
        filepath = os.path.join(SINGLE_OUTPUT_FOLDER, filename)
        save_audio(final_audio, filepath)
        
        return jsonify({
            'success': True,
            'audio_url': f'/static/single_outputs/{filename}',
            'physics': vehicle_meta,
            'settings': {
                'obs_pos': obs_pos,
                'duration': duration
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Simulation error: {str(e)}'}), 500
