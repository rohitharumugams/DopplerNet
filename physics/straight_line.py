import numpy as np
from audio.audio_utils import SR
from physics.road_frame import straight_passby_kinematics
from physics.straight_trajectory import is_miss_trajectory, straight_track_positions

C_SOUND_STANDARD = 343.0  # m/s
NEAR_FIELD_RADIUS = 6.0  # m – broader near-field to avoid sharp CPA peaks


def calculate_straight_line_doppler(
    speed_mps, min_distance_m, angle_deg, duration_s, c_sound=343.0, cpa_time_s=None
):
    """
    Straight-line pass-by with angle and near-field-safe amplitude.
    (Constant velocity variant)
    """
    return calculate_straight_line_accelerated_doppler(
        speed_mps, 0.0, min_distance_m, angle_deg, duration_s, c_sound, cpa_time_s=cpa_time_s
    )


def calculate_straight_line_track_doppler(
    duration_s,
    x0,
    y0,
    vx,
    vy,
    c_sound=343.0,
    accel_mps2=0.0,
):
    """
    Straight kinematic track in world coordinates (observer at origin).

    No CPA alignment — for fly-by / miss trajectories that never pass near the observer.
    """
    num_samples = int(round(SR * duration_s))
    track_params = {
        "duration": duration_s,
        "track_x0": x0,
        "track_y0": y0,
        "track_vx": vx,
        "track_vy": vy,
        "acceleration": accel_mps2,
        "pass_by_in_clip": False,
        "motion_scenario": "miss_track",
    }
    x, y, v_vec_x, v_vec_y, _closest = straight_track_positions(
        track_params, num_samples
    )

    r = np.sqrt(x**2 + y**2)
    r_safe = np.maximum(r, 1e-9)
    v_r = (v_vec_x * x + v_vec_y * y) / r_safe

    freq_ratios = c_sound / (c_sound + v_r)
    r_ref = 10.0
    spatial_amp = r_ref / np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    convective_amp = (c_sound / (c_sound + v_r)) ** 1.0
    amplitudes = (spatial_amp * convective_amp) ** 0.7
    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)


def calculate_straight_line_accelerated_doppler(
    speed_v0_mps,
    accel_mps2,
    min_distance_m,
    angle_deg,
    duration_s,
    c_sound=343.0,
    cpa_time_s=None,
    *,
    track_params=None,
):
    """
    Straight-line motion with constant acceleration (B7).

    Prefer ``track_params`` (parallel pass-by / miss track); legacy centered CPA
  geometry uses ``straight_passby_kinematics`` when track is absent.
    """
    if track_params is not None and all(
        k in track_params for k in ("track_x0", "track_y0", "track_vx", "track_vy")
    ):
        return calculate_straight_line_track_doppler(
            duration_s,
            track_params["track_x0"],
            track_params["track_y0"],
            track_params["track_vx"],
            track_params["track_vy"],
            c_sound=c_sound,
            accel_mps2=accel_mps2,
        )

    num_samples = int(round(SR * duration_s))
    x, y, v_x, v_y, _lateral, _cpa_xy = straight_passby_kinematics(
        speed_v0_mps,
        accel_mps2,
        min_distance_m,
        angle_deg,
        duration_s,
        cpa_time_s,
        num_samples,
    )
    r = np.sqrt(x**2 + y**2)
    r_safe = np.maximum(r, 1e-9)
    v_r = (v_x * x + v_y * y) / r_safe
    
    freq_ratios = c_sound / (c_sound + v_r)
    r_ref = 10.0
    spatial_amp = r_ref / np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    convective_amp = (c_sound / (c_sound + v_r))**1.0
    amplitudes = (spatial_amp * convective_amp)**0.7
    
    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)
