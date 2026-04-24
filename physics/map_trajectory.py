import numpy as np
from audio.audio_utils import SR, apply_distance_fade

NEAR_FIELD_RADIUS = 2.0

def calculate_map_trajectory_doppler(points, speed_mps, duration_s, observer_pos=(0, 0), c_sound=343.0):
    """
    Calculate Doppler shift for a custom trajectory defined by point list.
    Points are (x, y) in meters. 
    
    The speed_mps is the target average speed. 
    The trajectory is sampled to fit the duration.
    """
    points = np.array(points) # (N, 2)
    
    # Calculate cumulative distance along path
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
    total_path_len = cumulative_dist[-1]
    
    # Time axis
    num_samples = int(round(SR * duration_s))
    t = np.linspace(0.0, duration_s, num_samples, endpoint=False)
    
    # Interpolate positions based on constant speed
    # Average speed = total_path_len / duration_s
    # We use user-provided speed_mps to scale time or path
    effective_duration = total_path_len / speed_mps
    
    # Map t to the path distance
    # If speed_mps is used, we might finish path early or need to loop. 
    # For simplicity, we stretch/compress path to fit duration at average speed.
    query_dist = np.linspace(0, total_path_len, num_samples)
    
    px = np.interp(query_dist, cumulative_dist, points[:, 0])
    py = np.interp(query_dist, cumulative_dist, points[:, 1])
    
    p = np.vstack([px, py]) # (2, N)
    
    # Observer at custom position
    obs = np.array(observer_pos).reshape(2, 1)
    p_rel = p - obs
    r = np.linalg.norm(p_rel, axis=0) # (N,)
    
    # Velocity vector (finite difference)
    v = np.diff(p, axis=1, append=p[:, -1:]) * SR # (2, N)
    
    # Radial velocity v_r = (v · p_rel) / |p_rel|
    v_r = np.sum(v * p_rel, axis=0) / np.maximum(r, 1e-9)
    
    freq_ratios = c_sound / (c_sound + v_r)
    spatial_amp = 1.0 / np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    convective_amp = (c_sound / (c_sound + v_r))**2
    amplitudes = (10.0 * spatial_amp * convective_amp)**0.7
    
    # Smooth fade-in/out to prevent abrupt spawning
    amplitudes = apply_distance_fade(amplitudes, fade_duration_s=1.0)
    
    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)
