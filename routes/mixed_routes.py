import os
import json
import csv
import threading
import time
import random
import traceback
import shutil
import numpy as np
from flask import Blueprint, request, jsonify
from core.config import OUTPUT_FOLDER
from audio.generation import generate_single_clip, generate_statistics
from audio.audio_utils import SR

mixed_bp = Blueprint('mixed', __name__)

# Global progress state for Mixed Mode
mixed_progress = {
    'total_target': 0,
    'generated_so_far': 0,
    'current_car': '',
    'current_sample_index': 0,
    'is_running': False,
    'batch_dir': '',
    'log_line': ''
}

@mixed_bp.route('/api/generate_real_traffic_batch', methods=['POST'])
def generate_real_traffic_batch():
    global mixed_progress
    if mixed_progress['is_running']:
        return jsonify({'error': 'A mixed batch generation is already in progress'}), 400

    config = request.json or {}

    # Initialize progress
    mixed_progress['is_running'] = True
    mixed_progress['generated_so_far'] = 0
    mixed_progress['log_line'] = 'Starting background thread...'
    
    # Start generation in background
    thread = threading.Thread(target=run_mixed_generation, args=(config,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True})

@mixed_bp.route('/api/mixed_progress')
def get_mixed_progress():
    return jsonify(mixed_progress)

def run_mixed_generation(user_config):
    global mixed_progress
    try:
        metadata_path = os.path.join(os.getcwd(), 'ref_docs', 'vs13(6)metadata.json')
        if not os.path.exists(metadata_path):
            mixed_progress['log_line'] = f"Error: Metadata file not found at {metadata_path}"
            mixed_progress['is_running'] = False
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Calculate total target
        total = sum(len(speeds) for speeds in metadata.values())
        mixed_progress['total_target'] = total
        
        # Folder Mapping for spelling differences
        car_mapping = {
            "Peugeot3008": "Peuguot3008",
            "Peugeot307": "Peuguot307",
            "NissanQashqai": "NissanQashQai"
        }
        
        # Determine root folder and batch ID
        base_output_dir = user_config.get('batch', {}).get('save_path', os.path.join(OUTPUT_FOLDER, "MatchedData"))
        custom_batch_name = user_config.get('batch', {}).get('name', '').strip()
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        if custom_batch_name:
            safe_name = "".join(c for c in custom_batch_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
            root_batch_id = f"{safe_name}_{timestamp}"
        else:
            root_batch_id = f"MatchedData_{timestamp}"

        root_dir = os.path.join(base_output_dir, root_batch_id)
        os.makedirs(root_dir, exist_ok=True)
        
        target_distance = float(user_config.get('distance', 0.5))
        target_angle = float(user_config.get('angle', 0.0))
        output_format = user_config.get('output', {}).get('format', 'wav')
        
        all_clips_metadata = []
        
        for car_json_name, speeds in metadata.items():
            car_folder_name = car_mapping.get(car_json_name, car_json_name)
            mixed_progress['current_car'] = car_json_name
            mixed_progress['log_line'] = f"Processing {car_json_name}..."
            
            car_dir = os.path.join(root_dir, car_folder_name)
            os.makedirs(car_dir, exist_ok=True)
            
            additional_dir = os.path.join(car_dir, "Additional_files")
            os.makedirs(additional_dir, exist_ok=True)
            
            # Batch config for this car
            batch_config = {
                'output': {
                    'spectrogram_type': 'cqt',
                    'generate_diagnostics': True,
                    'format': output_format
                },
                'batch': {
                    'include_sample_folders': False # Flat structure requirement
                },
                'atmosphere': {
                    'add_air_noise': False 
                }
            }
            
            car_clips = []
            generated_filenames = []
            
            for i, speed_kmph in enumerate(speeds):
                mixed_progress['current_sample_index'] = i + 1
                
                # Internal high-precision conversion
                speed_mps = float(speed_kmph) / 3.6
                
                # Parameters
                params = {
                    'speed': speed_mps,
                    'distance': target_distance,
                    'angle': target_angle,
                    'duration': 10.0,
                    'acceleration': 0.0,
                    'temperature': 20,
                    'humidity': 50
                }
                
                base_filename = f"{car_folder_name}_{round(float(speed_kmph), 2)}"
                ext = "wav" if output_format == "wav" else "mp3"
                
                try:
                    # Generate sample
                    clip_meta = generate_single_clip(
                        vehicle_name=car_folder_name,
                        path_type='straight',
                        params=params,
                        output_dir=additional_dir,
                        batch_id=car_folder_name,
                        index=i+1,
                        config=batch_config,
                        custom_filename=base_filename
                    )
                    
                    # Extract audio to car_dir
                    sample_folder = clip_meta['sample_dir']
                    source_audio = os.path.join(additional_dir, sample_folder, "Essential", f"{base_filename}.{ext}")
                    dest_audio = os.path.join(car_dir, f"{base_filename}.{ext}")
                    if os.path.exists(source_audio):
                        shutil.copy2(source_audio, dest_audio)
                    
                    # Create .txt annotation
                    cpa_time = clip_meta['labels'].get('cpa_time_sec', 5.0)
                    annotation_path = os.path.join(car_dir, f"{base_filename}.txt")
                    with open(annotation_path, 'w') as af:
                        af.write(f"{float(speed_kmph):.2f} {cpa_time:.2f}")
                    
                    # Cleanup sample folder
                    shutil.rmtree(os.path.join(additional_dir, sample_folder))
                    
                    clip_meta['vehicle'] = car_json_name
                    car_clips.append(clip_meta)
                    all_clips_metadata.append(clip_meta)
                    generated_filenames.append(base_filename)
                    
                    mixed_progress['log_line'] = f"Generated {car_json_name} @ {speed_kmph} km/h"
                    
                except Exception as e:
                    mixed_progress['log_line'] = f"Error @ {speed_kmph} km/h: {str(e)}"
                    print(traceback.format_exc())
                
                mixed_progress['generated_so_far'] += 1

            # --- Post-process Car Folder ---
            if not car_clips: continue
            
            # 1. Train/Valid Split (80/20)
            random.seed(42) # Reproducible
            shuffled = list(generated_filenames)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * 0.8)
            train_set = set(shuffled[:split_idx])
            
            split_path = os.path.join(car_dir, "Train_valid_split.txt")
            with open(split_path, 'w') as sf:
                for sid in sorted(generated_filenames):
                    tag = "train" if sid in train_set else "valid"
                    sf.write(f"{sid} {tag}\n")
            
            # 2. metadata.json
            with open(os.path.join(additional_dir, f"metadata_{car_folder_name}.json"), 'w') as f:
                json.dump({'car': car_json_name, 'clips': car_clips}, f, indent=2)
                
            # 3. dataset.csv
            with open(os.path.join(additional_dir, "dataset.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'filename', 'speed_kmph', 'speed_mps', 'distance_m', 'angle_deg'])
                for clip in car_clips:
                    p = clip['parameters']
                    writer.writerow([
                        clip['sample_dir'], clip['filename'], 
                        round(p['speed']*3.6, 2), p['speed'], 
                        p['distance'], p['angle']
                    ])

            # 4. statistics.txt
            stats_text = generate_statistics(car_clips, batch_config)
            with open(os.path.join(additional_dir, f"statistics_{car_folder_name}.txt"), 'w') as f:
                f.write(stats_text)
                
            # 5. Generation Log
            log_path = os.path.join(additional_dir, f"generation_log_{car_folder_name}.txt")
            with open(log_path, 'w') as log_f:
                log_f.write(f"DopplerNet Mixed Batch Log: {car_folder_name}\n")
                log_f.write("="*60 + "\n")
                log_f.write(f"Total Clips: {len(car_clips)}\n")
                log_f.write(f"Output Format: VS13-Compatible\n")

            mixed_progress['log_line'] = f"✓ Finalized {car_json_name}"

    except Exception as e:
        mixed_progress['log_line'] = f"Critical Error in mixed generation: {str(e)}"
        print(traceback.format_exc())
    finally:
        mixed_progress['is_running'] = False

