import os
import json
import csv
import threading
import time
import random
import traceback
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

    # Initialize progress
    mixed_progress['is_running'] = True
    mixed_progress['generated_so_far'] = 0
    mixed_progress['log_line'] = 'Starting background thread...'
    
    # Start generation in background
    thread = threading.Thread(target=run_mixed_generation)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True})

@mixed_bp.route('/api/mixed_progress')
def get_mixed_progress():
    return jsonify(mixed_progress)

def run_mixed_generation():
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
        # User metadata vs sound filenames mapping
        car_mapping = {
            "Peugeot3008": "Peuguot3008",
            "Peugeot307": "Peuguot307",
            "NissanQashqai": "NissanQashQai"
        }
        
        for car_json_name, speeds in metadata.items():
            car_folder_name = car_mapping.get(car_json_name, car_json_name)
            mixed_progress['current_car'] = car_json_name
            mixed_progress['log_line'] = f"Generating batch for {car_json_name}..."
            
            # Create car-specific batch2 folder
            batch_id = f"{car_folder_name}_batch2"
            batch_dir = os.path.join(OUTPUT_FOLDER, batch_id)
            os.makedirs(batch_dir, exist_ok=True)
            
            audio_dir = os.path.join(batch_dir, "audio_clips")
            os.makedirs(audio_dir, exist_ok=True)
            
            # Configuration for this car's batch
            batch_config = {
                'output': {'spectrogram_type': 'cqt'},
                'benchmarks': {
                    'enabled': True,
                    'selected': ['B1', 'B7'],
                    'params': {'enable_acceleration': True}
                },
                'acceleration': {'randomize': False, 'value': 0.0}
            }
            
            clips_metadata = []
            
            # Prepare CSV and Log files
            csv_path = os.path.join(batch_dir, "dataset.csv")
            log_path = os.path.join(batch_dir, f"generation_log_{batch_id}.txt")
            
            with open(log_path, 'w') as log_f:
                log_f.write(f"DopplerNet Mixed Batch Generation Log: {batch_id}\n")
                log_f.write("="*60 + "\n")
            
            for i, speed in enumerate(speeds):
                mixed_progress['current_sample_index'] = i + 1
                sample_idx = i + 1
                
                # Parameters based on user's previous request (0.5-0.6 distance, -5 to 5 angle)
                # But standardized to 10s duration for consistency
                params = {
                    'speed': float(speed),
                    'distance': random.uniform(0.5, 0.6),
                    'angle': random.uniform(-5, 5),
                    'duration': 10.0,
                    'acceleration': 0.0,
                    'temperature': 20,
                    'humidity': 50
                }
                
                try:
                    # Generate the sample
                    clip_meta = generate_single_clip(
                        vehicle_name=car_folder_name,
                        path_type='straight',
                        params=params,
                        output_dir=audio_dir,
                        batch_id=batch_id,
                        index=sample_idx,
                        config=batch_config
                    )
                    clips_metadata.append(clip_meta)
                    
                    # Update local log
                    with open(log_path, 'a') as log_f:
                        log_f.write(f"Generated clip {sample_idx}/{len(speeds)}: {clip_meta['filename']} (Speed: {speed} m/s)\n")
                    
                    mixed_progress['log_line'] = f"Generated {car_json_name} sample {sample_idx}/{len(speeds)}"
                    
                except Exception as e:
                    mixed_progress['log_line'] = f"Error generating sample {sample_idx} for {car_json_name}: {str(e)}"
                    with open(log_path, 'a') as log_f:
                        log_f.write(f"FAILED clip {sample_idx}: {str(e)}\n")
                
                mixed_progress['generated_so_far'] += 1

            # Save Batch-level metadata.json
            metadata_file = os.path.join(batch_dir, f"metadata_{batch_id}.json")
            full_meta = {
                'timestamp': time.strftime('%Y%m%d_%H%M%S'),
                'car': car_json_name,
                'batch_id': batch_id,
                'config': batch_config,
                'clips': clips_metadata
            }
            with open(metadata_file, 'w') as f:
                json.dump(full_meta, f, indent=2)

            # Save dataset.csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'filename', 'vehicle_class', 'speed_mps', 'cpa_distance_m', 'acceleration_mps2'])
                for clip in clips_metadata:
                    p = clip['parameters']
                    writer.writerow([
                        clip['sample_dir'],
                        clip['filename'],
                        car_json_name,
                        p['speed'],
                        p['distance'],
                        p.get('acceleration', 0.0)
                    ])

            # Save statistics.txt
            stats_path = os.path.join(batch_dir, f"statistics_{batch_id}.txt")
            stats_text = generate_statistics(clips_metadata, batch_config)
            with open(stats_path, 'w') as f:
                f.write(stats_text)
                
            mixed_progress['log_line'] = f"✓ Completed batch for {car_json_name}"

    except Exception as e:
        mixed_progress['log_line'] = f"Critical Error in mixed generation: {str(e)}"
        print(traceback.format_exc())
    finally:
        mixed_progress['is_running'] = False

