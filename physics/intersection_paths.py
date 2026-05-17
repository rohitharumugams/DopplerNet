"""
Waypoint polylines for B9 plus-shaped intersection scenes.

Geometry matches visualization ``intersection_mode``: primary (E–W) has edges at
y = ±lane_half and median y = 0; secondary uses ``sec_cos`` / ``sec_sin`` with
the same half-width ``lane_half`` in the perpendicular (``lane_pos``) sense.

For B9 batch, pass ``lane_half = bench lane_width`` (median-to-edge), the same
value used to draw road edges in ``plot_utils`` — not ``lane_width / 2``.

All world samples are projected onto the plus-road union, then clamped to the
assigned lane half on each leg so paths do not hug medians, cross into oncoming
traffic, or leave the paved corridor.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple


def _arm_point(
    arm: str,
    dist_from_center: float,
    lane_pos: float,
    sec_cos: float,
    sec_sin: float,
) -> Tuple[float, float]:
    if arm == "W":
        return (-dist_from_center, lane_pos)
    if arm == "E":
        return (dist_from_center, lane_pos)
    if arm == "N":
        return (
            dist_from_center * sec_cos - lane_pos * sec_sin,
            dist_from_center * sec_sin + lane_pos * sec_cos,
        )
    return (
        -dist_from_center * sec_cos - lane_pos * sec_sin,
        -dist_from_center * sec_sin + lane_pos * sec_cos,
    )


def _signed_secondary(x: float, y: float, sec_sin: float, sec_cos: float) -> float:
    return -x * sec_sin + y * sec_cos


def _side_sign(lat: float, lh: float) -> float:
    eps = max(1e-6, 0.08 * lh)
    if lat > eps:
        return 1.0
    if lat < -eps:
        return -1.0
    return random.choice((-1.0, 1.0))


def _lane_band(side: float, lh: float) -> Tuple[float, float]:
    """Strict band on one side of median, away from centerline and outer edge."""
    margin = 0.22 * lh
    cap = 0.86 * lh
    if side > 0:
        return (margin, cap)
    return (-cap, -margin)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _clamp_lat_in_band(lat: float, lo: float, hi: float) -> float:
    return _clamp(lat, lo, hi)


def _project_to_plus_road(
    x: float,
    y: float,
    lh: float,
    sec_sin: float,
    sec_cos: float,
) -> Tuple[float, float]:
    """
    Project (x, y) onto { |y| <= lh } ∪ { |signed_secondary| <= lh }.

    If already in the union, return unchanged — do not clip ``y`` when the point is
    valid on the secondary strip with large |y|.
    """
    lim = lh * 0.998
    for _ in range(10):
        s = _signed_secondary(x, y, sec_sin, sec_cos)
        in_h = abs(y) <= lim
        in_s = abs(s) <= lim
        if in_h or in_s:
            return x, y
        du = abs(y) - lim
        dv = abs(s) - lim
        if du >= dv:
            y = math.copysign(lim, y)
        else:
            ds = math.copysign(lim, s) - s
            x -= sec_sin * ds
            y += sec_cos * ds
    s = _signed_secondary(x, y, sec_sin, sec_cos)
    if abs(y) <= lim or abs(s) <= lim:
        return x, y
    y = math.copysign(lim, y)
    return x, y


def _set_secondary_lateral(x: float, y: float, target_s: float, sec_sin: float, sec_cos: float) -> Tuple[float, float]:
    s = _signed_secondary(x, y, sec_sin, sec_cos)
    ds = target_s - s
    return x - sec_sin * ds, y + sec_cos * ds


def _clamp_horizontal_y(x: float, y: float, y_lo: float, y_hi: float) -> Tuple[float, float]:
    return x, _clamp(y, y_lo, y_hi)


def _turn_corner_point(
    p0: Tuple[float, float],
    p2: Tuple[float, float],
    lh: float,
) -> Tuple[float, float]:
    """Interior Bezier control inside the central box, biased toward the chord midpoint."""
    mx = 0.5 * (p0[0] + p2[0])
    my = 0.5 * (p0[1] + p2[1])
    span = 0.42 * lh * random.uniform(0.35, 1.0)
    ang = random.uniform(0.0, 2.0 * math.pi)
    cx = mx + span * math.cos(ang)
    cy = my + span * math.sin(ang)
    b = 0.82 * lh
    return _clamp(cx, -b, b), _clamp(cy, -b, b)


def _quadratic_samples(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    n: int,
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for k in range(1, n + 1):
        t = k / (n + 1)
        omt = 1.0 - t
        x = omt * omt * p0[0] + 2.0 * omt * t * p1[0] + t * t * p2[0]
        y = omt * omt * p0[1] + 2.0 * omt * t * p1[1] + t * t * p2[1]
        out.append((x, y))
    return out


def _segment_clamp_arm(
    x: float,
    y: float,
    arm: str,
    y_lo: float,
    y_hi: float,
    s_lo: float,
    s_hi: float,
    lh: float,
    sec_sin: float,
    sec_cos: float,
) -> Tuple[float, float]:
    x, y = _project_to_plus_road(x, y, lh, sec_sin, sec_cos)
    if arm in ("W", "E"):
        x, y = _clamp_horizontal_y(x, y, y_lo, y_hi)
    else:
        s = _signed_secondary(x, y, sec_sin, sec_cos)
        s_t = _clamp(s, s_lo, s_hi)
        x, y = _set_secondary_lateral(x, y, s_t, sec_sin, sec_cos)
    return _project_to_plus_road(x, y, lh, sec_sin, sec_cos)


def _segment_clamp_interior_straight(
    x: float,
    y: float,
    primary_is_y: bool,
    lo: float,
    hi: float,
    lh: float,
    sec_sin: float,
    sec_cos: float,
) -> Tuple[float, float]:
    x, y = _project_to_plus_road(x, y, lh, sec_sin, sec_cos)
    if primary_is_y:
        x, y = _clamp_horizontal_y(x, y, lo, hi)
    else:
        s = _signed_secondary(x, y, sec_sin, sec_cos)
        s_t = _clamp(s, lo, hi)
        x, y = _set_secondary_lateral(x, y, s_t, sec_sin, sec_cos)
    return _project_to_plus_road(x, y, lh, sec_sin, sec_cos)


def _segment_clamp_interior_turn(
    x: float,
    y: float,
    hy0: float,
    hy1: float,
    s0: float,
    s1: float,
    lh: float,
    sec_sin: float,
    sec_cos: float,
) -> Tuple[float, float]:
    """Keep samples inside the paved plus and near the entry–exit corridor."""
    x, y = _project_to_plus_road(x, y, lh, sec_sin, sec_cos)
    y_pad = 0.08 * lh
    s_pad = 0.08 * lh
    y_lo = _clamp(min(hy0, hy1) - y_pad, -lh * 0.99, lh * 0.99)
    y_hi = _clamp(max(hy0, hy1) + y_pad, -lh * 0.99, lh * 0.99)
    s_lo = _clamp(min(s0, s1) - s_pad, -lh * 0.99, lh * 0.99)
    s_hi = _clamp(max(s0, s1) + s_pad, -lh * 0.99, lh * 0.99)
    for _ in range(5):
        y = _clamp(y, y_lo, y_hi)
        s = _signed_secondary(x, y, sec_sin, sec_cos)
        s = _clamp(s, s_lo, s_hi)
        x, y = _set_secondary_lateral(x, y, s, sec_sin, sec_cos)
    x, y = _project_to_plus_road(x, y, lh, sec_sin, sec_cos)
    return x, y


def build_intersection_waypoints(
    start_arm: str,
    end_arm: str,
    half_span: float,
    lane_half: float,
    entry_lane: float,
    exit_lane: float,
    sec_cos: float,
    sec_sin: float,
) -> List[Tuple[float, float]]:
    lh = float(lane_half)
    half_span = float(half_span)

    side_e = _side_sign(entry_lane, lh)
    side_x = _side_sign(exit_lane, lh)
    e_lo, e_hi = _lane_band(side_e, lh)
    x_lo, x_hi = _lane_band(side_x, lh)

    entry_edge = _arm_point(start_arm, lh, entry_lane, sec_cos, sec_sin)
    exit_edge = _arm_point(end_arm, lh, exit_lane, sec_cos, sec_sin)
    hy0, hy1 = entry_edge[1], exit_edge[1]
    s0 = _signed_secondary(entry_edge[0], entry_edge[1], sec_sin, sec_cos)
    s1 = _signed_secondary(exit_edge[0], exit_edge[1], sec_sin, sec_cos)

    n_arm = random.randint(16, 24)
    wobble_lat = 0.055 * lh
    phase_arm = random.uniform(0.0, 2.0 * math.pi)

    pts: List[Tuple[float, float]] = []

    # --- Approach: only arm-native lateral noise, clamped to entry band ---
    lat0 = _clamp_lat_in_band(entry_lane + random.uniform(-wobble_lat, wobble_lat), e_lo, e_hi)
    for k in range(n_arm):
        frac = k / max(1, n_arm - 1)
        d = half_span + frac * (lh - half_span)
        lat = lat0 + frac * (entry_lane - lat0)
        lat += wobble_lat * math.sin(frac * math.pi * 1.15 + phase_arm)
        lat = _clamp_lat_in_band(lat, e_lo, e_hi)
        x, y = _arm_point(start_arm, d, lat, sec_cos, sec_sin)
        x, y = _segment_clamp_arm(x, y, start_arm, e_lo, e_hi, e_lo, e_hi, lh, sec_sin, sec_cos)
        pts.append((x, y))

    is_straight = frozenset((start_arm, end_arm)) in (
        frozenset(("W", "E")),
        frozenset(("N", "S")),
    )

    n_mid = random.randint(14, 22)

    if is_straight:
        dx = exit_edge[0] - entry_edge[0]
        dy = exit_edge[1] - entry_edge[1]
        chord = math.hypot(dx, dy)
        if chord < 1e-9:
            pts.append(exit_edge)
        else:
            ux, uy = dx / chord, dy / chord
            px, py = -uy, ux
            same_y = hy0 * hy1 > 0.0 and min(abs(hy0), abs(hy1)) > 0.1 * lh
            same_s = s0 * s1 > 0.0 and min(abs(s0), abs(s1)) > 0.1 * lh
            if frozenset((start_arm, end_arm)) == frozenset(("W", "E")) and same_y:
                xb_lo, xb_hi = _lane_band(side_x, lh)
                y_lo = min(e_lo, xb_lo)
                y_hi = max(e_hi, xb_hi)
                y_lo = _clamp(y_lo, -lh * 0.99, lh * 0.99)
                y_hi = _clamp(y_hi, -lh * 0.99, lh * 0.99)
                amp = 0.04 * lh * random.uniform(0.5, 1.0)
                for k in range(1, n_mid + 1):
                    t = k / (n_mid + 1)
                    bx = entry_edge[0] + t * dx + px * amp * math.sin(t * math.pi)
                    by = entry_edge[1] + t * dy + py * amp * math.sin(t * math.pi)
                    bx, by = _segment_clamp_interior_straight(bx, by, True, y_lo, y_hi, lh, sec_sin, sec_cos)
                    pts.append((bx, by))
            elif frozenset((start_arm, end_arm)) == frozenset(("N", "S")) and same_s:
                sb_lo, sb_hi = _lane_band(side_x, lh)
                s_lo = min(e_lo, sb_lo, s0, s1) - 0.02 * lh
                s_hi = max(e_hi, sb_hi, s0, s1) + 0.02 * lh
                s_lo = _clamp(s_lo, -lh * 0.99, lh * 0.99)
                s_hi = _clamp(s_hi, -lh * 0.99, lh * 0.99)
                amp = 0.04 * lh * random.uniform(0.5, 1.0)
                for k in range(1, n_mid + 1):
                    t = k / (n_mid + 1)
                    bx = entry_edge[0] + t * dx + px * amp * math.sin(t * math.pi)
                    by = entry_edge[1] + t * dy + py * amp * math.sin(t * math.pi)
                    bx, by = _segment_clamp_interior_straight(bx, by, False, s_lo, s_hi, lh, sec_sin, sec_cos)
                    pts.append((bx, by))
            else:
                for k in range(1, n_mid + 1):
                    t = k / (n_mid + 1)
                    bx = entry_edge[0] + t * dx
                    by = entry_edge[1] + t * dy
                    bx, by = _project_to_plus_road(bx, by, lh, sec_sin, sec_cos)
                    pts.append((bx, by))
    else:
        p1 = _turn_corner_point(entry_edge, exit_edge, lh)
        for bx, by in _quadratic_samples(entry_edge, p1, exit_edge, n_mid):
            bx, by = _segment_clamp_interior_turn(bx, by, hy0, hy1, s0, s1, lh, sec_sin, sec_cos)
            pts.append((bx, by))

    # --- Departure ---
    lat1 = _clamp_lat_in_band(exit_lane + random.uniform(-wobble_lat, wobble_lat), x_lo, x_hi)
    for k in range(n_arm):
        frac = k / max(1, n_arm - 1)
        d = lh + frac * (half_span - lh)
        lat = exit_lane + frac * (lat1 - exit_lane)
        lat += wobble_lat * math.sin(frac * math.pi * 1.15)
        lat = _clamp_lat_in_band(lat, x_lo, x_hi)
        x, y = _arm_point(end_arm, d, lat, sec_cos, sec_sin)
        x, y = _segment_clamp_arm(x, y, end_arm, x_lo, x_hi, x_lo, x_hi, lh, sec_sin, sec_cos)
        pts.append((x, y))

    return pts


def intersection_observer_position(
    road_half_width: float,
    intersection_angle_deg: float,
    clearance_frac: float = 0.35,
) -> Tuple[float, float]:
    """
    Listener position in Q1, outside the plus-road union.

    ``road_half_width`` is the median-to-outer-edge distance — the same
    ``lane_width`` / ``lane_half`` used for waypoints and plot road guides.
    """
    w = max(float(road_half_width), 0.5)
    theta = math.radians(float(intersection_angle_deg))
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    sin_safe = sin_t if abs(sin_t) > 1e-6 else (1e-6 if sin_t >= 0 else -1e-6)

    # Q1 outer corner where primary upper edge (y = w) meets secondary edge (s = w).
    corner_x = abs(w * (1.0 + cos_t) / sin_safe)
    corner_y = w

    def outside_pavement(ox: float, oy: float) -> bool:
        s = -ox * sin_t + oy * cos_t
        return abs(oy) > w and abs(s) > w

    scale = 1.0 + float(clearance_frac)
    ox, oy = scale * corner_x, scale * corner_y
    while not outside_pavement(ox, oy) and scale < 5.0:
        scale += 0.12
        ox, oy = scale * corner_x, scale * corner_y
    return (ox, oy)
