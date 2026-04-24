import os
import traceback
import numpy as np

import matplotlib
matplotlib.use('Agg')  # non-GUI backend for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d

import librosa
import librosa.display

from audio.audio_utils import SR


def compute_path_points(path_type, params, n_points=200, **kwargs):
    """Compute (x, y) path points for plotting"""
    duration = params.get('duration', 10.0)
    obs_pos = kwargs.get('observer_pos', (0, 0))
    is_absolute = kwargs.get('absolute', False)

    if path_type == 'straight':
        v = params['speed']
        
        # Check if this is an intersection-style straight path
        if 'road' in params:
            road = params['road']
            direction = params.get('direction', 1)
            arrival_time = params.get('arrival_time', duration / 2.0)
            offset = params.get('offset', 0.0)
            
            t = np.linspace(0.0, duration, n_points)
            dt = t - arrival_time
            
            if road == 'horizontal':
                x = direction * v * dt
                # If absolute, this is already world-X. But to center on observer_x:
                if is_absolute:
                    x = x + obs_pos[0]
                y = np.full_like(x, offset)
            else:
                y = direction * v * dt
                if is_absolute:
                    y = y + obs_pos[1]
                x = np.full_like(y, offset)
            
            # Adjust for observer position
            if not is_absolute:
                # obs_pos for intersection defaults to (10,10) if not provided
                default_obs = (10, 10) if 'road' in params else (0, 0)
                curr_obs = kwargs.get('observer_pos', default_obs)
                x = x - curr_obs[0]
                y = y - curr_obs[1]
            
            closest = None
        else:
            # Pass-by pass logic
            h = params.get('distance', 30.0) # Fallback to 30m if distance missing
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

            if is_absolute:
                x = x + obs_pos[0]
                y = y + obs_pos[1]

            cx, cy = p_c
            if is_absolute:
                cx += obs_pos[0]
                cy += obs_pos[1]
            closest = (cx, cy)

    elif path_type == 'parabola':
        v = params['speed']
        a = params['a']
        h = params['h']

        t = np.linspace(0.0, duration, n_points)
        t0 = duration / 2.0
        dt = t - t0

        x = v * dt
        y = a * x**2 + h

        if is_absolute:
            x = x + obs_pos[0]
        closest = None

    elif path_type == 'bezier':
        x0 = float(params.get('x0', 0))
        x1 = float(params.get('x1', 0))
        x2 = float(params.get('x2', 0))
        x3 = float(params.get('x3', 0))
        y0 = float(params.get('y0', 0))
        y1 = float(params.get('y1', 0))
        y2 = float(params.get('y2', 0))
        y3 = float(params.get('y3', 0))

        u = np.linspace(0.0, 1.0, n_points)
        x = ((1 - u) ** 3) * x0 + 3 * ((1 - u) ** 2) * u * x1 + 3 * (1 - u) * (u ** 2) * x2 + (u ** 3) * x3
        y = ((1 - u) ** 3) * y0 + 3 * ((1 - u) ** 2) * u * y1 + 3 * (1 - u) * (u ** 2) * y2 + (u ** 3) * y3

        if not is_absolute:
            x = x - obs_pos[0]
            y = y - obs_pos[1]
        closest = None

    elif path_type == 'map_path':
        points = np.array(params['points'])
        # Sample points to match n_points
        dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
        total_path_len = cumulative_dist[-1]
        
        query_dist = np.linspace(0, total_path_len, n_points)
        x = np.interp(query_dist, cumulative_dist, points[:, 0])
        y = np.interp(query_dist, cumulative_dist, points[:, 1])
        
        if not is_absolute:
            x = x - obs_pos[0]
            y = y - obs_pos[1]
        closest = None
    else:
        # fallback: trivial horizontal line
        x = np.linspace(-10, 10, n_points)
        y = np.zeros_like(x)
        closest = None

    # --- APPLY GLOBAL ROAD CURVATURE (Realistic Roads Fix) ---
    road_curve_a = kwargs.get('road_curve_a', 0.0)
    curve_blend = float(params.get('road_curve_blend', 1.0))
    global_curve_scale = float(params.get('global_curve_scale', 1.0))
    local_curve_a = float(params.get('road_curve_a', 0.0))
    path_curve_a = (road_curve_a * global_curve_scale + local_curve_a) * curve_blend
    if path_curve_a != 0:
        # In absolute mode, the curve is centered on the observer's X position
        x_ref = x - obs_pos[0] if is_absolute else x
        y = y + path_curve_a * (x_ref ** 2)

    # --- APPLY GLOBAL ROAD TILT ---
    road_angle = float(kwargs.get('road_angle', 0.0))
    road_angle += float(params.get('road_angle_offset', 0.0))
    if road_angle != 0:
        theta = np.deg2rad(road_angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Rotate around road centerline anchor instead of origin to avoid large visual drift.
        pivot_x = obs_pos[0] if is_absolute else 0.0
        pivot_y = kwargs.get('road_y_center', obs_pos[1] if is_absolute else 0.0)
        x_rel, y_rel = x - pivot_x, y - pivot_y
        x_rot = x_rel * cos_t - y_rel * sin_t
        y_rot = x_rel * sin_t + y_rel * cos_t
        x, y = x_rot + pivot_x, y_rot + pivot_y

    # --- ABSOLUTE PLOT SAFETY CLAMP (keeps trajectories inside road band) ---
    if is_absolute and kwargs.get('clamp_to_road_band', False):
        lane_width = float(kwargs.get('lane_width', 4.0))
        road_y_center = float(kwargs.get('road_y_center', 0.0))
        road_curve_clamp = kwargs.get('road_curve_a', 0.0)
        x_ref_clamp = x - obs_pos[0] if is_absolute else x
        center_at_x = road_y_center + road_curve_clamp * (x_ref_clamp ** 2)
        y_min = center_at_x - lane_width
        y_max = center_at_x + lane_width
        y = np.clip(y, y_min, y_max)

    return x, y, closest


def save_path_plot(path_type, params, output_dir, base_name):
    """
    Save a PNG path graph for this clip with realistic road aesthetics.
    """
    try:
        x, y, closest = compute_path_points(path_type, params, n_points=200)
        plot_path = os.path.join(output_dir, f"{base_name}.png")

        fig, ax = plt.subplots(figsize=(6, 6))

        # Build legend label ...
        label_parts = [f"Path ({path_type.capitalize()})"]
        for k in ['speed', 'distance', 'offset', 'angle', 'temperature', 'humidity']:
            if k in params:
                val = params[k]
                lbl = {'speed': 'v', 'distance': 'd', 'offset': 'off', 'angle': 'θ'}.get(k, k[0])
                unit = {'speed': 'm/s', 'distance': 'm', 'offset': 'm', 'angle': '°'}.get(k, '')
                label_parts.append(f"{lbl}={val}{unit}")
        
        full_label = ", ".join(label_parts)

        # Path
        ax.plot(x, y, linewidth=1.4, color='#1f77b4', label=full_label, zorder=5)

        # Observer at origin
        ax.scatter([0], [0], marker='.', s=30, color='red', label="Observer", zorder=10)

        if closest is not None:
            cx, cy = closest
            ax.plot([0, cx], [0, cy], linestyle='--', linewidth=1, color='gray', alpha=0.6, zorder=4)

        ax.axis('equal')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax.grid(True, which="major", linestyle=':', alpha=0.4, zorder=0)
        ax.set_facecolor('#fafafa')
        
        # Legend at bottom
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)

        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return os.path.basename(plot_path)

    except Exception as e:
        print(f"Failed to save path plot for {base_name}: {e}")
        return None


def save_combined_path_plot(scenes_data, output_dir, base_name, **kwargs):
    """
    Save a PNG graph with all vehicle paths in a scene.
    Focuses on clear trajectory visualization and readable axis scaling.
    """
    try:
        plot_path = os.path.join(output_dir, f"{base_name}_combined_path.png")
        obs_pos = kwargs.get('observer_pos', (0, 0))
        intersection_mode = bool(kwargs.get('intersection_benchmark', False))
        road_shape = kwargs.get('road_shape', 'straight')
        road_curve_a = float(kwargs.get('road_curve_a', 0.0))
        plot_kwargs = dict(kwargs)
        plot_kwargs['road_curve_a'] = road_curve_a
        # Clamp plotted trajectories to the road band (visual safety; avoids
        # global road curvature + tilt pushing paths outside lane edges).
        plot_kwargs['clamp_to_road_band'] = not intersection_mode
        
        fig, ax = plt.subplots(figsize=(12, 8))

        # Precompute paths once (used for both rendering and viewport).
        sampled_paths = []
        for i, (path_type, params, vehicle_name) in enumerate(scenes_data):
            x, y, _ = compute_path_points(path_type, params, n_points=200, **plot_kwargs)
            sampled_paths.append((i, path_type, params, vehicle_name, x, y))

        # --- MINIMAL ROAD GUIDES (no decorative background) ---
        lane_width = kwargs.get('lane_width', 4.0)
        road_y_center = kwargs.get('road_y_center', 0.0)
        road_angle = kwargs.get('road_angle', 0.0)

        if intersection_mode:
            lane_half = float(lane_width) / 2.0
            half_arm = float(kwargs.get('intersection_half_arm', 90.0))
            int_angle = float(kwargs.get('intersection_angle', 90.0))
            # Primary road (E-W): horizontal
            ax.plot([-half_arm, half_arm], [lane_half, lane_half], color='#666666', linewidth=1.0, label='Road Edge (Primary)', zorder=1)
            ax.plot([-half_arm, half_arm], [-lane_half, -lane_half], color='#666666', linewidth=1.0, zorder=1)
            ax.plot([-half_arm, half_arm], [0.0, 0.0], color='#888888', linestyle='--', linewidth=0.9, label='Median (Primary)', zorder=1)
            # Secondary road at intersection_angle from x-axis
            _ia_rad = np.deg2rad(int_angle)
            _ia_cos, _ia_sin = np.cos(_ia_rad), np.sin(_ia_rad)
            # Direction along secondary arm and its perpendicular (for lane offset)
            def _sec_line(d_start, d_end, lateral):
                """Line from d_start to d_end along the secondary axis, offset by lateral."""
                x0 = d_start * _ia_cos - lateral * _ia_sin
                y0 = d_start * _ia_sin + lateral * _ia_cos
                x1 = d_end * _ia_cos - lateral * _ia_sin
                y1 = d_end * _ia_sin + lateral * _ia_cos
                return [x0, x1], [y0, y1]
            ax.plot(*_sec_line(-half_arm, half_arm, lane_half), color='#666666', linewidth=1.0, label='Road Edge (Secondary)', zorder=1)
            ax.plot(*_sec_line(-half_arm, half_arm, -lane_half), color='#666666', linewidth=1.0, zorder=1)
            ax.plot(*_sec_line(-half_arm, half_arm, 0.0), color='#888888', linestyle='--', linewidth=0.9, label='Median (Secondary)', zorder=1)
            # Dummy series for viewport computation
            x_upper = x_lower = x_med = np.array([-half_arm, half_arm])
            y_upper = np.array([lane_half, lane_half])
            y_lower = np.array([-lane_half, -lane_half])
            y_med = np.array([0.0, 0.0])
        else:
            if sampled_paths:
                path_x = np.concatenate([p[4] for p in sampled_paths])
                path_y = np.concatenate([p[5] for p in sampled_paths])
                x_min_path = float(np.min(path_x))
                x_max_path = float(np.max(path_x))
                x_span = max(60.0, x_max_path - x_min_path)
                x_pad = max(15.0, 0.1 * x_span)
                x_road = np.linspace(x_min_path - x_pad, x_max_path + x_pad, 500)
                if 'road_y_center' not in kwargs:
                    road_y_center = float(np.median(path_y))
            else:
                x_road = np.linspace(-120.0, 120.0, 500)

            x_rel = x_road - obs_pos[0]

            # Build a centerline that can be straight/parabolic/bezier/auto-fit.
            if road_shape == 'bezier':
                t = np.linspace(0.0, 1.0, x_road.size)
                y0 = road_y_center
                y3 = road_y_center
                bulge = float(kwargs.get('road_bezier_bulge', 0.6))
                y1 = road_y_center + bulge
                y2 = road_y_center - bulge
                y_median = (
                    ((1 - t) ** 3) * y0
                    + 3 * ((1 - t) ** 2) * t * y1
                    + 3 * (1 - t) * (t ** 2) * y2
                    + (t ** 3) * y3
                )
            elif road_shape == 'parabola' or abs(road_curve_a) > 0.0:
                y_median = road_y_center + road_curve_a * (x_rel ** 2)
            elif sampled_paths and kwargs.get('auto_fit_road', False):
                # Fit a smooth quadratic centerline from vehicle paths.
                path_x = np.concatenate([p[4] for p in sampled_paths])
                path_y = np.concatenate([p[5] for p in sampled_paths])
                if np.std(path_x) > 1e-6:
                    poly = np.polyfit(path_x, path_y, 2)
                    y_median = np.polyval(poly, x_road)
                else:
                    y_median = np.full_like(x_road, road_y_center, dtype=float)
            else:
                y_median = np.full_like(x_road, road_y_center, dtype=float)

            # Offset road edges along local normal so the edge distance == lane_width.
            dy_dx = np.gradient(y_median, x_road)
            denom = np.sqrt(1.0 + dy_dx ** 2)
            n_x = -dy_dx / denom
            n_y = 1.0 / denom

            x_upper = x_road + lane_width * n_x
            y_upper = y_median + lane_width * n_y
            x_lower = x_road - lane_width * n_x
            y_lower = y_median - lane_width * n_y
            x_med = x_road
            y_med = y_median

            if road_angle != 0.0:
                theta = np.deg2rad(road_angle)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                pivot_x = obs_pos[0]
                pivot_y = road_y_center

                def rotate(px, py):
                    px_rel, py_rel = px - pivot_x, py - pivot_y
                    rx = px_rel * cos_t - py_rel * sin_t
                    ry = px_rel * sin_t + py_rel * cos_t
                    return rx + pivot_x, ry + pivot_y

                x_upper, y_upper = rotate(x_upper, y_upper)
                x_lower, y_lower = rotate(x_lower, y_lower)
                x_med, y_med = rotate(x_med, y_med)

            ax.plot(x_upper, y_upper, color='#666666', linewidth=1.0, label='Road Edge', zorder=1)
            ax.plot(x_lower, y_lower, color='#666666', linewidth=1.0, zorder=1)
            ax.plot(x_med, y_med, color='#888888', linestyle='--', linewidth=0.9, label='Median', zorder=1)

        # --- PLOT VEHICLE PATHS ---
        for i, path_type, params, vehicle_name, x, y in sampled_paths:
            # Determine arrow based on directional intent in UNROTATED frame
            # For straight/parabola it's usually speed/direction, for bezier it's x3 > x0
            is_forward = True
            if path_type == 'straight' and params.get('direction', 1) == -1:
                is_forward = False
            elif path_type == 'straight' and params.get('angle', 0) == 180:
                is_forward = False
            elif path_type == 'parabola' and params.get('speed', 1) < 0:
                is_forward = False
            elif path_type == 'bezier' and params.get('x3', 0) < params.get('x0', 0):
                is_forward = False
                
            arrow = " →" if is_forward else " ←"
            ax.plot(x, y, linewidth=1.4, label=f"V{i+1}: {vehicle_name}{arrow}", alpha=0.95, zorder=5)

        # Observer
        ax.scatter([obs_pos[0]], [obs_pos[1]], marker='.', s=30, color='red', label='Observer', zorder=10)

        # --- ADAPTIVE VIEWPORT ---
        # Scale view from trajectory data so y-axis stays informative.
        x_all = [obs_pos[0]]
        y_all = [obs_pos[1]]
        
        # Include all path points
        for _, _, _, _, px, py in sampled_paths:
            x_all.extend(px)
            y_all.extend(py)
        # Keep road guides visible and make lane width visually meaningful.
        x_all.extend(x_upper)
        x_all.extend(x_lower)
        x_all.extend(x_med)
        y_all.extend(y_upper)
        y_all.extend(y_lower)
        y_all.extend(y_med)
        if intersection_mode:
            lh = float(lane_width) / 2.0
            ha = float(kwargs.get('intersection_half_arm', 90.0))
            # Primary road extents
            x_all.extend([-ha, ha])
            y_all.extend([lh, -lh])
            # Secondary road extents (at intersection_angle)
            _vp_rad = np.deg2rad(float(kwargs.get('intersection_angle', 90.0)))
            _vp_cos, _vp_sin = np.cos(_vp_rad), np.sin(_vp_rad)
            for _d in [-ha, ha]:
                for _lat in [-lh, lh, 0.0]:
                    x_all.append(_d * _vp_cos - _lat * _vp_sin)
                    y_all.append(_d * _vp_sin + _lat * _vp_cos)
        
        x_min, x_max = min(x_all), max(x_all)
        y_min, y_max = min(y_all), max(y_all)

        if intersection_mode:
            # Dedicated square viewport for + intersections; do not reuse strip-road
            # y-capping logic, which flattens the scene.
            ha = float(kwargs.get('intersection_half_arm', 90.0))
            pad = max(8.0, 0.12 * ha)
            x_low = min(x_min, -ha, obs_pos[0]) - pad
            x_high = max(x_max, ha, obs_pos[0]) + pad
            y_low = min(y_min, -ha, obs_pos[1]) - pad
            y_high = max(y_max, ha, obs_pos[1]) + pad
            span = max(x_high - x_low, y_high - y_low)
            cx = 0.5 * (x_low + x_high)
            cy = 0.5 * (y_low + y_high)
            ax.set_xlim(cx - span / 2.0, cx + span / 2.0)
            ax.set_ylim(cy - span / 2.0, cy + span / 2.0)
            ax.set_aspect('equal', adjustable='box')
        else:
            x_pad = (x_max - x_min) * 0.15
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

            # Y-axis: show full road width with generous padding so lanes
            # are clearly visible regardless of the x-axis span.
            road_edge_lo = float(np.min(y_lower))
            road_edge_hi = float(np.max(y_upper))
            road_span = road_edge_hi - road_edge_lo
            y_pad_abs = max(road_span * 0.6, 3.0)
            y_low = road_edge_lo - y_pad_abs
            y_high = road_edge_hi + y_pad_abs

            # Keep observer visible if it is near the road.
            y_low = min(y_low, obs_pos[1] - 1.0)
            y_high = max(y_high, obs_pos[1] + 1.0)

            ax.set_ylim(y_low, y_high)

        ax.set_xlabel("x (meters)", fontsize=14)
        ax.set_ylabel("y (meters)", fontsize=14)
        
        ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Set X-axis gap to 30m
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
        
        ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

        # Keep background plain for easier visual inspection.
        ax.set_facecolor('#fafafa')

        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
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
