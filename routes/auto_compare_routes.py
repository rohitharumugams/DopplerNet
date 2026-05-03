import os
import re
import numpy as np
import librosa
from flask import Blueprint, request, jsonify
from audio.audio_utils import SR
from visualization.plot_utils import save_automated_comparison_plot

auto_compare_bp = Blueprint('auto_compare', __name__)

def parse_filename(filename):
    """
    Given KiaSportage_31.wav or KiaSportage_31.0.wav,
    extract carname and speed.
    """
    if not filename.lower().endswith('.wav'):
        return None, None, None
    base = filename[:-4]  # remove .wav
    parts = base.rsplit('_', 1)
    if len(parts) != 2:
        return None, None, None
    carname = parts[0]
    speed_str = parts[1]
    try:
        speed = float(speed_str)
        norm_carname = carname.replace(' ', '').replace('_', '').replace('-', '').lower()
        return carname, norm_carname, speed
    except ValueError:
        return None, None, None

@auto_compare_bp.route('/api/auto_compare/get_pairs', methods=['POST'])
def get_pairs():
    data = request.get_json() or {}
    dataset_a = data.get('dataset_a', r'D:\Antigravity\vs13-model\RealData')
    dataset_b = data.get('dataset_b', r'D:\Antigravity\vs13-model\MatchedData')
    out_dir = data.get('out_dir', 'static/comparision_outputs')

    if not os.path.exists(dataset_a) or not os.path.exists(dataset_b):
        return jsonify({'error': 'One or both dataset paths do not exist.'}), 400

    # Ensure output directory exists to save missing_comparisons.txt
    os.makedirs(out_dir, exist_ok=True)

    # Scan Dataset A
    a_items = {}
    a_unparsed = []
    for root, _, files in os.walk(dataset_a):
        for f in files:
            orig_carname, norm_carname, speed = parse_filename(f)
            if norm_carname is not None and speed is not None:
                a_items[(norm_carname, speed)] = {
                    'orig_carname': orig_carname,
                    'path': os.path.join(root, f)
                }
            elif f.lower().endswith('.wav'):
                a_unparsed.append(os.path.join(root, f))

    # Scan Dataset B
    b_items = {}
    b_unparsed = []
    for root, _, files in os.walk(dataset_b):
        for f in files:
            _, norm_carname, speed = parse_filename(f)
            if norm_carname is not None and speed is not None:
                b_items[(norm_carname, speed)] = os.path.join(root, f)
            elif f.lower().endswith('.wav'):
                b_unparsed.append(os.path.join(root, f))

    # Find matches and missing
    pairs = []
    a_missing_in_b = []
    
    for (norm_carname, speed), data_a in a_items.items():
        if (norm_carname, speed) in b_items:
            path_b = b_items[(norm_carname, speed)]
            pairs.append({
                'carname': data_a['orig_carname'],
                'speed': speed,
                'path_a': data_a['path'],
                'path_b': path_b
            })
        else:
            a_missing_in_b.append(data_a['path'])
            
    b_missing_in_a = []
    for (norm_carname, speed), path_b in b_items.items():
        if (norm_carname, speed) not in a_items:
            b_missing_in_a.append(path_b)

    # Sort pairs to be deterministic
    pairs.sort(key=lambda x: (x['carname'], x['speed']))
    
    # Write missing comparisons to a text file
    missing_txt_path = os.path.join(out_dir, 'missing_comparisons.txt')
    try:
        with open(missing_txt_path, 'w', encoding='utf-8') as f:
            f.write("=== Missing from MatchedData (Dataset B) ===\n")
            for path in sorted(a_missing_in_b):
                f.write(path + "\n")
            
            f.write("\n=== Missing from RealData (Dataset A) ===\n")
            for path in sorted(b_missing_in_a):
                f.write(path + "\n")
                
            f.write("\n=== Unparseable WAV files in Dataset A ===\n")
            for path in sorted(a_unparsed):
                f.write(path + "\n")
                
            f.write("\n=== Unparseable WAV files in Dataset B ===\n")
            for path in sorted(b_unparsed):
                f.write(path + "\n")
    except Exception as e:
        print(f"Failed to write missing comparisons: {e}")

    return jsonify({
        'success': True,
        'total_pairs': len(pairs),
        'pairs': pairs,
        'missing_txt_path': missing_txt_path
    })


@auto_compare_bp.route('/api/auto_compare/process_pair', methods=['POST'])
def process_pair():
    data = request.get_json() or {}
    path_a = data.get('path_a')
    path_b = data.get('path_b')
    carname = data.get('carname')
    speed = data.get('speed')
    out_dir = data.get('out_dir', 'static/comparision_outputs')

    if not all([path_a, path_b, carname, speed is not None]):
        return jsonify({'error': 'Missing required pair information'}), 400

    try:
        y_a, _ = librosa.load(path_a, sr=SR, mono=True)
        y_b, _ = librosa.load(path_b, sr=SR, mono=True)
    except Exception as e:
        return jsonify({'error': f'Failed to load audio: {str(e)}'}), 500

    if len(y_a) == 0 or len(y_b) == 0:
        return jsonify({'error': 'One of the audio files is empty'}), 400

    # Compute metrics
    n_fft = 2048
    hop_length = 256

    rms_a = librosa.feature.rms(y=y_a, frame_length=n_fft, hop_length=hop_length)[0]
    rms_b = librosa.feature.rms(y=y_b, frame_length=n_fft, hop_length=hop_length)[0]
    n_env = min(len(rms_a), len(rms_b))
    rms_a_norm = rms_a[:n_env] / (np.max(rms_a[:n_env]) + 1e-9)
    rms_b_norm = rms_b[:n_env] / (np.max(rms_b[:n_env]) + 1e-9)

    amp_overlap = float(np.sum(np.minimum(rms_a_norm, rms_b_norm)) / (np.sum(np.maximum(rms_a_norm, rms_b_norm)) + 1e-9) * 100.0)

    if n_env > 1 and (np.std(rms_a_norm) > 1e-9) and (np.std(rms_b_norm) > 1e-9):
        env_corr = float(np.corrcoef(rms_a_norm, rms_b_norm)[0, 1])
    else:
        env_corr = 0.0
    env_corr_pct = float(np.clip((env_corr + 1.0) * 50.0, 0.0, 100.0))

    stft_a = np.abs(librosa.stft(y_a, n_fft=n_fft, hop_length=hop_length))
    stft_b = np.abs(librosa.stft(y_b, n_fft=n_fft, hop_length=hop_length))
    spec_a = np.mean(stft_a, axis=1)
    spec_b = np.mean(stft_b, axis=1)
    spec_a_norm = spec_a / (np.sum(spec_a) + 1e-9)
    spec_b_norm = spec_b / (np.sum(spec_b) + 1e-9)
    spectral_overlap = float(np.sum(np.minimum(spec_a_norm, spec_b_norm)) * 100.0)

    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    dom_freq_a = float(freqs[int(np.argmax(spec_a))]) if len(spec_a) else 0.0
    dom_freq_b = float(freqs[int(np.argmax(spec_b))]) if len(spec_b) else 0.0

    overall_similarity = float(np.clip((spectral_overlap * 0.55) + (amp_overlap * 0.30) + (env_corr_pct * 0.15), 0.0, 100.0))

    metrics = {
        'Duration A (s)': float(len(y_a) / SR),
        'Duration B (s)': float(len(y_b) / SR),
        'Dominant Freq A (Hz)': dom_freq_a,
        'Dominant Freq B (Hz)': dom_freq_b,
        'Envelope Correlation (%)': env_corr_pct,
        'Spectral Overlap (%)': spectral_overlap,
        'Overall Match (%)': overall_similarity
    }

    # Format the speed for display/filename. If it's an integer, format as int.
    speed_disp = int(speed) if isinstance(speed, (int, float)) and float(speed).is_integer() else speed
    filename_base = f"{carname}_{speed_disp}"
    
    # Ensure vehicle folder exists in output dir
    vehicle_out_dir = os.path.join(out_dir, carname)
    os.makedirs(vehicle_out_dir, exist_ok=True)
    
    out_path = os.path.join(vehicle_out_dir, f"{filename_base}.png")

    ok = save_automated_comparison_plot(
        y_a, y_b, SR,
        f"VS13 Data: {carname} - {speed_disp} km/h",
        f"Simulated Data: {carname} - {speed_disp} km/h",
        out_path,
        metrics,
        max_y_freq=2500
    )

    if not ok:
        return jsonify({'error': 'Failed to save comparison plot'}), 500

    return jsonify({
        'success': True,
        'carname': carname,
        'speed': speed_disp,
        'image_path': out_path
    })
