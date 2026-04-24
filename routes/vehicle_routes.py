import os
import uuid
import time
import librosa

from flask import Blueprint, request, jsonify

from audio.audio_utils import SR
from core.config import UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER, SPECTROGRAM_FOLDER
from visualization.plot_utils import save_spectrogram_to_file

import soundfile as sf

vehicle_bp = Blueprint('vehicle', __name__)


@vehicle_bp.route('/api/upload_vehicle', methods=['POST'])
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


@vehicle_bp.route('/api/list_vehicles', methods=['GET'])
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


@vehicle_bp.route('/api/delete_vehicle/<filename>', methods=['DELETE'])
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


@vehicle_bp.route('/api/generate_spectrogram', methods=['POST'])
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
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@vehicle_bp.route('/api/upload_generate_spectrogram', methods=['POST'])
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
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
