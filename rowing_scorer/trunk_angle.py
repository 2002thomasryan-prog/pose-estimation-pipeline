"""
Trunk angle, measured separately at the catch and at the finish.

Signed convention: 0 deg upright, positive = leaning toward the footplate
(forward lean at the catch), negative = leaning away from it (layback at the
finish).  The sign is normalised by facing direction so it does not depend on
which way the rower faces in frame.

Catch target 25-35 deg forward, finish target -30 to -20 deg (British Rowing
Plus; Soper & Hume 2004).  Replaces the previous whole-cycle p90-p10 range,
which mixed two distinct postural questions into one number.  The range is
retained below, unscored, for the dashboard trace.
"""

import numpy as np
from scorer_utils import _jxy, signed_trunk_angle

from .cycle_detection import rower_facing_sign


EVENT_WINDOW = 2   # +/- frames median-averaged around an event frame


def compute_trunk_angle_at_event(seg, frame_local, near_side,
                                 window=EVENT_WINDOW):
    """
    Signed trunk angle at a single stroke event, in degrees.

    Parameters
    ----------
    seg         : list[dict]  keypoints for one cycle (normalised)
    frame_local : int         local frame index of the event
    near_side   : str         'left' | 'right'
    window      : int         +/- frames median-averaged around the event

    Returns
    -------
    float | None   positive = forward lean, negative = layback.
                   None if the window contains no finite samples.
    """
    n = len(seg)
    if n == 0:
        return None

    prefix = 'l' if near_side == 'left' else 'r'
    hip_xy = _jxy(seg, f'{prefix}_hip')
    sh_xy  = _jxy(seg, f'{prefix}_shoulder')

    lo = int(np.clip(frame_local - window, 0, n - 1))
    hi = int(np.clip(frame_local + window, 0, n - 1))

    angles = np.array([
        signed_trunk_angle(hip_xy[i], sh_xy[i])
        for i in range(lo, hi + 1)
    ], dtype=float)

    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return None

    return float(np.median(angles) * rower_facing_sign(seg, near_side))


def compute_trunk_angle_catch(seg, catch_local, near_side):
    """Signed trunk angle at the catch. Expected positive (forward lean)."""
    return compute_trunk_angle_at_event(seg, catch_local, near_side)


def compute_trunk_angle_finish(seg, finish_local, near_side):
    """Signed trunk angle at the finish. Expected negative (layback)."""
    return compute_trunk_angle_at_event(seg, finish_local, near_side)


def compute_trunk_angle_range(seg, near_side):
    """
    p90-p10 range of signed trunk angle across one stroke cycle.

    No longer scored -- retained because the dashboard and the detail plots
    still display it.  Always positive.
    """
    series = compute_trunk_angle_series(seg, near_side)
    if series.size == 0:
        return float('nan')
    return float(np.percentile(series, 90) - np.percentile(series, 10))


def compute_trunk_angle_series(seg, near_side):
    """
    Full facing-normalised signed trunk angle time series for one cycle.

    Returns
    -------
    np.ndarray  shape (n,)  degrees, positive = forward lean
    """
    prefix = 'l' if near_side == 'left' else 'r'
    hip_xy = _jxy(seg, f'{prefix}_hip')
    sh_xy  = _jxy(seg, f'{prefix}_shoulder')

    n    = len(seg)
    sign = rower_facing_sign(seg, near_side)

    return np.array([
        signed_trunk_angle(hip_xy[i], sh_xy[i]) * sign
        for i in range(n)
    ])
