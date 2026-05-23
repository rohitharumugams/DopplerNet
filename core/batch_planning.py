"""Standard batch slot planning and synthesis (not linear overlap)."""

from __future__ import annotations

import math
import random
from collections import Counter

from audio.generation import (
    generate_random_parameters,
    generate_single_clip,
    generate_multi_object_clip,
)
from physics.intersection_paths import (
    build_intersection_waypoints,
    intersection_observer_position,
)


def _apply_direction_variant(path_kind, clip_params, want_reverse):
    """Force clip motion direction for direction-classification benchmarks."""
    if path_kind == 'straight':
        if 'track_vx' in clip_params:
            vx = abs(float(clip_params['track_vx']))
            clip_params['track_vx'] = -vx if want_reverse else vx
            clip_params['angle'] = 180 if want_reverse else 0
            clip_params['direction'] = -1 if want_reverse else 1
            # Keep miss trajectories on one side of x=0 after direction flip.
            if 'track_x0' in clip_params:
                duration = max(1e-6, float(clip_params.get('duration', 10.0)))
                margin = 12.0
                min_abs_x0 = vx * duration + margin
                x0 = float(clip_params['track_x0'])
                if want_reverse:
                    clip_params['track_x0'] = max(abs(x0), min_abs_x0)
                else:
                    clip_params['track_x0'] = -max(abs(x0), min_abs_x0)
        else:
            clip_params['angle'] = 180 if want_reverse else 0
            clip_params['direction'] = -1 if want_reverse else 1
    elif path_kind == 'parabola':
        speed_abs = abs(float(clip_params.get('speed', 0.0)))
        clip_params['speed'] = -speed_abs if want_reverse else speed_abs
        clip_params['direction'] = -1 if want_reverse else 1
    elif path_kind == 'bezier':
        x0 = float(clip_params.get('x0', -1.0))
        x3 = float(clip_params.get('x3', 1.0))
        x1 = float(clip_params.get('x1', (x0 + x3) / 2.0))
        x2 = float(clip_params.get('x2', (x0 + x3) / 2.0))
        currently_reverse = x3 < x0
        if want_reverse != currently_reverse:
            clip_params['x0'], clip_params['x3'] = x3, x0
            clip_params['x1'], clip_params['x2'] = x2, x1
        clip_params['direction'] = -1 if want_reverse else 1


def plan_slot(i: int, ctx: dict) -> dict:
    """
    Plan one standard batch slot on the main process (uses SAMPLERS / random state).
    Returns a serializable job dict for synthesize_planned().
    """
    config = ctx["config"]
    total_clips = ctx["total_clips"]
    vehicle_list = ctx["vehicle_list"]
    path_list = ctx["path_list"]
    motion_pass_by_flags = ctx["motion_pass_by_flags"]
    audio_dir = ctx["audio_dir"]
    batch_id = ctx["batch_id"]
    clip_index = i + 1

    vehicle_name = vehicle_list[i]
    path_type = path_list[i]

    # ── Benchmark B8 (Multi-source), B9 (Interaction), B10 (Recognition) ──
    bench_cfg = config.get('benchmarks', {})
    selected_bencharks = bench_cfg.get('selected', [])
    
    # Check for multi-source benchmarks
    multi_bench_active = any(b in selected_bencharks for b in ['B8', 'B9', 'B10'])
    single_bench_active = any(b in selected_bencharks for b in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    bench_params_early = bench_cfg.get('params', {})
    intersection_active = (
        'B9' in selected_bencharks
        and bool(bench_params_early.get('intersection_benchmark', False))
    )
    # Only interleave single-source on odd indices when both single
    # AND multi benchmarks are selected; otherwise honour the selection.
    if bench_cfg.get('enabled', False) and multi_bench_active:
        if single_bench_active:
            is_multi_source = intersection_active or (i % 2 == 0)
        else:
            is_multi_source = True
    else:
        is_multi_source = False

    if is_multi_source:
        # Multi-source mode (Realistic Busy Road)
        bench_params = bench_cfg.get('params', {})
        selected_benchmarks = bench_cfg.get('selected', [])
        
        # Extract Busy Road parameters from benchmark settings
        lane_width = float(bench_params.get('lane_width', 4.0)) # Width of ONE lane
        include_opposite = bench_params.get('include_opposite', True)
        max_stagger = float(bench_params.get('max_stagger', 5.0))
        v_min = int(bench_params.get('vehicle_min', 2))
        v_max = int(bench_params.get('vehicle_max', 5))
        # Keep multi-source controls explicit to avoid NameError in loop logic.
        num_sources = random.randint(v_min, v_max)
        is_force_crossing = bool(
            bench_params.get('force_crossing', False)
            or bench_params.get('is_crossing', False)
        )
        # Distribute selected road shapes equally across scenes.
        available_shapes = bench_params.get('road_shapes', ['straight', 'parabola', 'bezier'])
        if not available_shapes:
            available_shapes = ['straight']
        road_shape = available_shapes[i % len(available_shapes)]
        if road_shape == 'parabola':
            road_curve_a = random.choice([-1, 1]) * random.uniform(3e-5, 2e-4)
        else:
            road_curve_a = 0.0
        road_bezier_bulge = random.uniform(0.4, 1.2) if road_shape == 'bezier' else 0.0

        # Calculate road_y_center to maintain safe distance (10m) from nearest edge
        road_y_center = lane_width + 10.0
        observer_pos = (0.0, 0.0)
        
        v_configs = []

        # ── B9: Plus-shaped intersection benchmark ─────────────────────
        intersection_mode = (
            'B9' in selected_bencharks
            and bool(bench_params.get('intersection_benchmark', False))
        )
        if intersection_mode:
            # Intersection centered at origin; primary road along x-axis,
            # secondary road at the configured intersection angle.
            road_curve_a = 0.0
            road_y_center = 0.0
            half_arm = float(bench_params.get('intersection_half_arm', 90.0))
            # Median-to-edge distance; must match plot_utils intersection
            # guides (lane_half = lane_width there).
            lane_half = float(lane_width)
            ia_min = float(bench_params.get('intersection_angle_min', 30.0))
            ia_max = float(bench_params.get('intersection_angle_max', 150.0))
            # Evenly cover the range across scenes via linear spacing.
            if total_clips > 1:
                intersection_angle = ia_min + (ia_max - ia_min) * (i / (total_clips - 1))
            else:
                intersection_angle = (ia_min + ia_max) / 2.0

            # Observer outside pavement at the same half-width as paths/ plot.
            _ia_rad = math.radians(intersection_angle)
            _ia_sin = math.sin(_ia_rad)
            _ia_cos = math.cos(_ia_rad)
            observer_pos = intersection_observer_position(
                float(lane_width), intersection_angle
            )

            exits_by_approach = {
                'W': ['E', 'N', 'S'],
                'E': ['W', 'S', 'N'],
                'S': ['N', 'W', 'E'],
                'N': ['S', 'E', 'W'],
            }

            # Primary road (E-W) stays along x-axis;
            # secondary road (N-S) is rotated by intersection_angle.
            _sec_cos = _ia_cos
            _sec_sin = _ia_sin

            sel = config.get('vehicles', {}).get('selected', [vehicle_name])
            temp = 20
            hum = 50
            duration = 10.0
            speeds = [random.randint(
                int(config.get('speed', {}).get('min', 15)),
                int(config.get('speed', {}).get('max', 35)),
            ) for _ in range(max(1, num_sources))]

            # Pre-assign unique lane positions per arm so no two
            # vehicles from the same arm share the same lateral slot.
            inner = lane_half * 0.75
            arm_lane_slots = {}
            def _get_lane_slot(arm, count_in_arm):
                if arm not in arm_lane_slots:
                    arm_lane_slots[arm] = 0
                idx = arm_lane_slots[arm]
                arm_lane_slots[arm] += 1
                if count_in_arm <= 1:
                    return random.uniform(-inner * 0.5, inner * 0.5)
                step = (2.0 * inner) / count_in_arm
                return -inner + step * (idx + 0.5) + random.uniform(-step * 0.15, step * 0.15)

            approach_order = ['W', 'E', 'S', 'N']
            # Build arm assignments first to count vehicles per arm.
            arm_assignments = []
            for s_idx in range(max(1, num_sources)):
                start_arm = approach_order[s_idx % 4] if s_idx < 4 else random.choice(approach_order)
                end_arm = random.choices(
                    exits_by_approach[start_arm],
                    weights=[0.30, 0.40, 0.30],
                    k=1,
                )[0]
                arm_assignments.append((start_arm, end_arm))

            start_counts = Counter(a[0] for a in arm_assignments)
            end_counts = Counter(a[1] for a in arm_assignments)

            v_configs = []
            for s_idx, (start_arm, end_arm) in enumerate(arm_assignments):
                entry_lane = _get_lane_slot(start_arm, start_counts[start_arm])
                exit_lane = _get_lane_slot('exit_' + end_arm, end_counts[end_arm])
                waypoints = build_intersection_waypoints(
                    start_arm,
                    end_arm,
                    half_arm,
                    lane_half,
                    entry_lane,
                    exit_lane,
                    _sec_cos,
                    _sec_sin,
                )
                v_configs.append({
                    'vehicle_name': random.choice(sel),
                    'path_type': 'map_path',
                    'params': {
                        'points': waypoints,
                        'speed': speeds[s_idx],
                        'duration': duration,
                        'temperature': temp,
                        'humidity': hum,
                    },
                    'delay': random.uniform(0, max_stagger),
                    'is_crossing': is_force_crossing,
                    'direction': 1,
                })

            return {
                'kind': 'multi',
                'clip_index': clip_index,
                'audio_dir': audio_dir,
                'batch_id': batch_id,
                'config': config,
                'v_configs': v_configs,
                'observer_pos': observer_pos,
                'road_curve_a': road_curve_a,
                'road_y_center': road_y_center,
                'intersection_angle': intersection_angle,
            }
        else:
            # Force even vehicle count for equal distribution across lanes.
            if num_sources % 2 != 0:
                num_sources += 1
            half = num_sources // 2

            # Lane geometry: road centered at road_y_center
            # Forward lane (dir=+1): [road_y_center - lane_width, road_y_center]
            # Opposite lane (dir=-1): [road_y_center, road_y_center + lane_width]
            fwd_y_min = road_y_center - lane_width
            fwd_y_max = road_y_center
            opp_y_min = road_y_center
            opp_y_max = road_y_center + lane_width

            edge_buffer = 0.4
            clamp_lo_fwd = fwd_y_min + edge_buffer
            clamp_hi_fwd = fwd_y_max - edge_buffer
            clamp_lo_opp = opp_y_min + edge_buffer
            clamp_hi_opp = opp_y_max - edge_buffer

            fwd_center = (fwd_y_min + fwd_y_max) / 2.0
            opp_center = (opp_y_min + opp_y_max) / 2.0
            usable_half = (lane_width / 2.0) - edge_buffer

            # Build ordered list: alternate fwd / opp.
            assignments = []
            for k in range(half):
                assignments.append((fwd_center, fwd_y_min, fwd_y_max, clamp_lo_fwd, clamp_hi_fwd, 1))
                assignments.append((opp_center, opp_y_min, opp_y_max, clamp_lo_opp, clamp_hi_opp, -1))

            # Scene-level crossing policy:
            # Force one paired crossing in a randomly chosen lane
            # whenever that lane has at least two vehicles.
            forced_cross_lane = random.choice([1, -1]) if half >= 2 else None

            # Spread vehicles across discrete lateral slots per lane so
            # trajectories are visually and physically separated.
            per_lane_total = {1: half, -1: half}
            per_lane_seen = {1: 0, -1: 0}
            # Target cadence: ~1 in 3 paths intersect per lane, with
            # randomized phase so it is not always the same slot index.
            lane_cross_phase = {1: random.randint(0, 2), -1: random.randint(0, 2)}
            # Keep one pending "cross partner" per lane so two nearby
            # vehicles can form an actual intersection/overtake pair.
            lane_cross_anchor = {1: None, -1: None}

            def _lane_slot(clamp_lo, clamp_hi, direction):
                total = max(1, per_lane_total[direction])
                idx = per_lane_seen[direction]
                per_lane_seen[direction] += 1
                width = max(0.2, clamp_hi - clamp_lo)
                # Keep at least ~0.9 m center-to-center when possible.
                nominal_gap = 0.9
                max_slots_fit = max(1, int(width / nominal_gap))
                n_slots = min(total, max_slots_fit)
                if n_slots <= 1:
                    y_slot = 0.5 * (clamp_lo + clamp_hi) + random.uniform(-0.06, 0.06)
                    return y_slot, idx, total
                # If too many vehicles for available width, wrap on slots
                # and add tiny noise so paths are still distinguishable.
                slot_idx = idx % n_slots
                frac = (slot_idx + 0.5) / n_slots
                y_slot = clamp_lo + frac * width
                return y_slot + random.uniform(-0.05, 0.05), idx, total

            for s_idx, (lane_center, lane_y_min, lane_y_max, clamp_lo, clamp_hi, direction) in enumerate(assignments):
                v_name = random.choice(config.get('vehicles', {}).get('selected', [vehicle_name]))
                s_min = int(config.get('speed', {}).get('min', 15))
                s_max = int(config.get('speed', {}).get('max', 35))
                speed = random.randint(s_min, s_max)

                # Assign a lane slot (with small jitter) to avoid overlap.
                lane_offset_raw, lane_idx, lane_total = _lane_slot(clamp_lo, clamp_hi, direction)
                lane_offset = max(clamp_lo, min(clamp_hi, lane_offset_raw))

                # Prefer road-following trajectories; keep parabola very rare.
                p_type = random.choices(['parabola', 'bezier'], weights=[0.05, 0.95])[0]
                is_forced_cross_pair = (
                    forced_cross_lane is not None
                    and direction == forced_cross_lane
                    and lane_idx in (0, 1)
                    and lane_total >= 2
                )
                if is_forced_cross_pair:
                    # Crossing pair must be Bezier so paths can intersect.
                    p_type = 'bezier'

                road_limit = 100.0
                max_dur = (2.0 * road_limit) / speed
                duration = min(10.0, max_dur * 0.98)

                v_params = {
                    'speed': speed,
                    'duration': duration,
                    'temperature': 20,
                    'humidity': 50,
                }
                v_params['road_curve_blend'] = 1.0
                v_params['road_angle_offset'] = 0.0
                v_params['global_curve_scale'] = 1.0

                if p_type == 'parabola':
                    v_params['h'] = lane_offset
                    span = abs(speed) * duration
                    half_span = max(1.0, span / 2.0)
                    available_down = lane_offset - clamp_lo
                    available_up = clamp_hi - lane_offset
                    # Keep vertical sag very small (realistic lane-following),
                    # unlike aggressive U-shapes that drift toward median.
                    target_dev = min(0.7, max(0.15, usable_half * 0.28))
                    desired_mag = target_dev / max(1.0, half_span ** 2)
                    max_a_up = max(0.0, available_up) / max(1.0, half_span ** 2)
                    max_a_down = max(0.0, available_down) / max(1.0, half_span ** 2)
                    up_cap = min(desired_mag * random.uniform(0.4, 1.0), max_a_up)
                    down_cap = min(desired_mag * random.uniform(0.4, 1.0), max_a_down)
                    if up_cap > 0 and down_cap > 0:
                        v_params['a'] = up_cap if random.random() < 0.5 else -down_cap
                    elif up_cap > 0:
                        v_params['a'] = up_cap
                    elif down_cap > 0:
                        v_params['a'] = -down_cap
                    else:
                        v_params['a'] = 0.0

                elif p_type == 'bezier':
                    # Mostly lane-following trajectories with minor deviations.
                    forced_cross = is_forced_cross_pair
                    crossing_event = forced_cross or (
                        lane_total >= 2
                        and ((lane_idx + lane_cross_phase[direction]) % 3 == 0)
                    )
                    if forced_cross:
                        maneuver = 'cross'
                    else:
                        maneuver = random.choices(
                            ['lane_follow', 'overtake', 'weave', 'cross'],
                            weights=[0.66, 0.17, 0.09, 0.08] if crossing_event else [0.88, 0.05, 0.07, 0.0],
                        )[0]
                    # Longitudinal staggering based on lane occupancy and lane width:
                    # - more vehicles in lane => tighter but still separated spacing
                    # - wider lane => allow slightly larger staggering envelope
                    half_span = min(100.0, (speed * duration) / 2.0)
                    lane_center_idx = 0.5 * (lane_total - 1)
                    spacing_gain = max(0.6, (lane_width / 4.0) ** 0.5)
                    travel_span = max(20.0, 2.0 * half_span * 0.80)
                    x_spacing = (travel_span / max(1, lane_total)) * spacing_gain
                    x_spacing = max(6.0, min(30.0, x_spacing))
                    x_jitter = random.uniform(-0.12 * x_spacing, 0.12 * x_spacing)
                    x_shift = (lane_idx - lane_center_idx) * x_spacing + x_jitter
                    # Max lateral drift from lane center in meters.
                    drift = min(0.65, max(0.12, usable_half * 0.22))
                    if crossing_event:
                        drift = min(usable_half * 0.80, drift * 1.35)
                    # Use per-vehicle lane slot baseline so vehicles in the same
                    # lane do not collapse onto the same trajectory.
                    base_y = lane_offset

                    if maneuver == 'cross':
                        # Smooth lane-change style arc that can intersect another path
                        # without introducing sharp turns.
                        amp = min(usable_half * 0.90, max(0.24, drift * 1.25))
                        anchor = lane_cross_anchor[direction]
                        if anchor is None:
                            side = random.choice([-1, 1])
                            lane_cross_anchor[direction] = {
                                'side': side,
                                'x_shift': x_shift,
                            }
                        else:
                            # Pair with previous cross vehicle in this lane:
                            # opposite sweep direction + nearly same x-shift so
                            # trajectories overlap and intersect near mid-clip.
                            side = -anchor['side']
                            x_shift = anchor['x_shift'] + random.uniform(-0.08 * x_spacing, 0.08 * x_spacing)
                            lane_cross_anchor[direction] = None

                        v_params['y0'] = base_y - side * amp * 0.70
                        v_params['y1'] = base_y - side * amp * 0.22
                        v_params['y2'] = base_y + side * amp * 0.22
                        v_params['y3'] = base_y + side * amp * 0.70
                    elif maneuver == 'overtake':
                        # Mild overtake-like arc that stays lane-local.
                        start_y = base_y + random.uniform(-drift * 0.5, drift * 0.5)
                        swing = random.choice([-1, 1]) * random.uniform(drift * 0.45, drift * 0.80)
                        v_params['y0'] = start_y
                        v_params['y1'] = base_y + swing
                        v_params['y2'] = base_y + swing * random.uniform(0.35, 0.70)
                        v_params['y3'] = start_y + random.uniform(-drift * 0.2, drift * 0.2)
                    elif maneuver == 'weave':
                        # Gentle S-curve around lane center.
                        v_params['y0'] = base_y + random.uniform(-drift * 0.5, drift * 0.5)
                        v_params['y3'] = base_y + random.uniform(-drift * 0.5, drift * 0.5)
                        side = random.choice([-1, 1])
                        v_params['y1'] = base_y + side * random.uniform(drift * 0.35, drift * 0.75)
                        v_params['y2'] = base_y - side * random.uniform(drift * 0.30, drift * 0.65)
                    else:
                        # Default: follow lane with very minor steering variation.
                        v_params['y0'] = base_y + random.uniform(-drift * 0.35, drift * 0.35)
                        v_params['y3'] = base_y + random.uniform(-drift * 0.35, drift * 0.35)
                        v_params['y1'] = base_y + random.uniform(-drift * 0.55, drift * 0.55)
                        v_params['y2'] = base_y + random.uniform(-drift * 0.55, drift * 0.55)

                    for key in ['y0', 'y1', 'y2', 'y3']:
                        v_params[key] = max(clamp_lo, min(clamp_hi, v_params[key]))

                    if direction == 1:
                        v_params['x0'], v_params['x3'] = -half_span + x_shift, half_span + x_shift
                    else:
                        v_params['x0'], v_params['x3'] = half_span + x_shift, -half_span + x_shift
                    v_params['x1'] = v_params['x0'] + (v_params['x3'] - v_params['x0']) * 0.33
                    v_params['x2'] = v_params['x0'] + (v_params['x3'] - v_params['x0']) * 0.66

                delay = random.uniform(0, max_stagger)

                v_configs.append({
                    'vehicle_name': v_name,
                    'path_type': p_type,
                    'params': v_params,
                    'delay': delay,
                    'is_crossing': is_force_crossing,
                    'offset': lane_offset,
                    'direction': direction,
                    'speed': speed,
                })
            
            return {
                'kind': 'multi',
                'clip_index': clip_index,
                'audio_dir': audio_dir,
                'batch_id': batch_id,
                'config': config,
                'v_configs': v_configs,
                'observer_pos': observer_pos,
                'road_curve_a': road_curve_a,
                'road_y_center': road_y_center,
                'road_shape': road_shape,
                'road_bezier_bulge': road_bezier_bulge,
            }
    else:
        # Standard single-source mode
        params = generate_random_parameters(
        config,
        vehicle_name,
        path_type,
        clip_index=clip_index,
        total_clips=total_clips,
        motion_pass_by=motion_pass_by_flags[i],
        )
        path_type_use = params.pop('_force_path_type', path_type)
        params.pop('_clip_index', None)
        params.pop('_total_clips', None)
        params.pop('_motion_pass_by', None)
        bench_cfg = config.get('benchmarks', {}) or {}
        bench_selected = bench_cfg.get('selected', []) or []
        bench_params = bench_cfg.get('params', {}) or {}
        direction_benchmark_active = (
            bool(bench_cfg.get('enabled', False))
            and ('B2' in bench_selected)
            and bool(bench_params.get('alternate_direction_clips', False))
        )
        if direction_benchmark_active:
            _apply_direction_variant(path_type_use, params, want_reverse=(i % 2 == 1))
        return {
            'kind': 'single',
            'clip_index': clip_index,
            'vehicle_name': vehicle_name,
            'path_type': path_type_use,
            'params': params,
            'audio_dir': audio_dir,
            'batch_id': batch_id,
            'config': config,
        }


def synthesize_planned(job: dict) -> dict:
    """Run audio synthesis for a planned slot (safe in worker processes)."""
    config = job['config']
    clip_index = job['clip_index']
    audio_dir = job['audio_dir']
    batch_id = job['batch_id']
    if job['kind'] == 'multi':
        kwargs = {
            'v_configs': job['v_configs'],
            'output_dir': audio_dir,
            'batch_name': batch_id,
            'index': clip_index,
            'config': config,
            'observer_pos': job.get('observer_pos', (0.0, 0.0)),
            'road_curve_a': job.get('road_curve_a', 0.0),
            'road_y_center': job.get('road_y_center', 0.0),
        }
        if 'intersection_angle' in job:
            kwargs['intersection_angle'] = job['intersection_angle']
        if 'road_shape' in job:
            kwargs['road_shape'] = job['road_shape']
        if 'road_bezier_bulge' in job:
            kwargs['road_bezier_bulge'] = job['road_bezier_bulge']
        return generate_multi_object_clip(**kwargs)
    return generate_single_clip(
        job['vehicle_name'],
        job['path_type'],
        job['params'],
        audio_dir,
        batch_id,
        clip_index,
        config,
    )


def plan_slot_with_retries(i: int, ctx: dict, max_attempts: int = 8) -> dict:
    """Plan a slot with retries for single-source parameter/geometry failures."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            return plan_slot(i, ctx)
        except Exception as exc:
            last_err = exc
            if attempt == max_attempts - 1:
                raise
    raise last_err or RuntimeError('clip planning failed')


def execute_slot(i: int, ctx: dict) -> dict:
    """Plan and synthesize one slot sequentially (legacy / fallback)."""
    last_err = None
    for attempt in range(8):
        try:
            job = plan_slot(i, ctx)
            return synthesize_planned(job)
        except Exception as exc:
            last_err = exc
            if attempt == 7:
                raise
    raise last_err or RuntimeError('clip generation failed')


# Alias used by batch route docs / legacy references.
_plan_slot = plan_slot
