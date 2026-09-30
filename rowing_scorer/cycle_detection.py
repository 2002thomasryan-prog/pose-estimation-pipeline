"""
Detects rowing stroke cycles from near-side wrist x-position, cross-validated against knee angle.

Catch = wrist at maximum extension away from the body; finish = maximum
displacement toward it, searched in [25%, 80%] of each cycle. Plateaus near
either extreme resolve to their midpoint. Disagreement between the wrist and
knee signals beyond CATCH_CROSS_VAL_TOL raises a warning; wrist stays primary.

Which x-extreme is the catch depends on which way the rower faces, not on
which anatomical side is nearest the camera. Facing is taken from the sign of
(ankle_x - hip_x), the feet being footplate-fixed ahead of the seat for the
whole stroke, so a wrong near-side detection can no longer invert catch and
finish.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from scorer_utils import _jxy, angle_at_joint

from .params import (
    PLATEAU_TOL, PLATEAU_MIN_FRAMES,
    CATCH_CROSS_VAL_TOL, CATCH_PEAK_PROM_FRAC,
    CATCH_MIN_DISTANCE, EDGE_CATCH_MARGIN,
)


def rower_facing_sign(kp_norm, near_side):
    """
    Return +1 if the rower faces the +x direction in frame, -1 if -x.

    Derived from ankle-hip geometry rather than from near-side detection:
    the feet are strapped to the footplate ahead of the seat throughout the
    stroke, so sign(ankle_x - hip_x) gives the facing direction directly.
    Returns +1 if the geometry is degenerate.
    """
    prefix = 'l' if near_side == 'left' else 'r'
    hip_xy = _jxy(kp_norm, f'{prefix}_hip')
    ank_xy = _jxy(kp_norm, f'{prefix}_ankle')

    d = float(np.nanmean(ank_xy[:, 0] - hip_xy[:, 0]))
    if not np.isfinite(d) or abs(d) < 1e-6:
        return 1.0
    return 1.0 if d > 0 else -1.0


def plateau_midpoint(signal, peak_idx,
                     tol=PLATEAU_TOL, min_frames=PLATEAU_MIN_FRAMES):
    """
    Find the midpoint of a plateau around peak_idx.

    A plateau is defined as a contiguous region where the signal stays
    within tol × total_range of the peak value for at least min_frames frames.

    Parameters
    ----------
    signal    : np.ndarray  1-D signal
    peak_idx  : int         candidate peak / trough index
    tol       : float       fraction of total signal range defining plateau
    min_frames: int         minimum width to qualify as a plateau

    Returns
    -------
    midpoint_idx : int
    is_plateau   : bool
    """
    total_range = np.nanmax(signal) - np.nanmin(signal)
    if total_range < 1e-9:
        return peak_idx, False

    threshold = tol * total_range
    peak_val  = signal[peak_idx]

    lo, hi = peak_idx, peak_idx
    while lo > 0 and abs(signal[lo - 1] - peak_val) <= threshold:
        lo -= 1
    while hi < len(signal) - 1 and abs(signal[hi + 1] - peak_val) <= threshold:
        hi += 1

    if (hi - lo) >= (min_frames - 1):
        return (lo + hi) // 2, True
    return peak_idx, False


def _wrist_catches(wrist_x, facing, smooth_size=9):
    """
    Detect catch frames from near-side wrist x-position.

    Facing +x: catch = maximum wrist x (hand extended furthest right).
    Facing -x: catch = minimum wrist x (hand extended furthest left).

    Uses find_peaks on the sign-oriented signal so catch is always a peak,
    then applies plateau midpoint correction.

    Parameters
    ----------
    wrist_x   : np.ndarray  near-side wrist x time series (raw pixels)
    facing    : float       +1 or -1 from rower_facing_sign()

    Returns
    -------
    np.ndarray  catch frame indices (may be empty)
    np.ndarray  smoothed signal used for peak finding
    """
    ws = uniform_filter1d(wrist_x, size=smooth_size)

    signal = ws * facing

    prom = CATCH_PEAK_PROM_FRAC * (
        np.percentile(signal, 90) - np.percentile(signal, 10)
    )
    raw_catches, _ = find_peaks(signal, prominence=prom,
                                distance=CATCH_MIN_DISTANCE)

    corrected = []
    for c in raw_catches:
        mid, _ = plateau_midpoint(wrist_x, c)
        corrected.append(mid)
    catches = np.array(corrected, dtype=int)

    n = len(wrist_x)
    if len(catches) > 0:
        if (catches[0] > EDGE_CATCH_MARGIN
                and signal[0] > signal[catches[0]] - prom * 0.5):
            catches = np.concatenate([[0], catches])
        if (catches[-1] < n - EDGE_CATCH_MARGIN
                and signal[-1] > signal[catches[-1]] - prom * 0.5):
            catches = np.concatenate([catches, [n - 1]])

    return catches, ws


def _knee_catches(kp_norm, near_side, smooth_size=9):
    """
    Detect catch frames from near-side minimum knee angle.
    Used only for cross-validation against the wrist signal.

    Returns
    -------
    np.ndarray  catch frame indices
    np.ndarray  smoothed knee angle time series (full recording)
    """
    prefix  = 'l' if near_side == 'left' else 'r'
    hip_xy  = _jxy(kp_norm, f'{prefix}_hip')
    knee_xy = _jxy(kp_norm, f'{prefix}_knee')
    ank_xy  = _jxy(kp_norm, f'{prefix}_ankle')

    n  = len(kp_norm)
    ka = np.array([
        angle_at_joint(hip_xy[i], knee_xy[i], ank_xy[i])
        for i in range(n)
    ])
    ka_sm = uniform_filter1d(ka, size=smooth_size)

    prom = CATCH_PEAK_PROM_FRAC * (
        np.percentile(ka_sm, 90) - np.percentile(ka_sm, 10)
    )
    raw_catches, _ = find_peaks(-ka_sm, prominence=prom,
                                distance=CATCH_MIN_DISTANCE)

    corrected = []
    for c in raw_catches:
        mid, _ = plateau_midpoint(ka_sm, c)
        corrected.append(mid)
    catches = np.array(corrected, dtype=int)

    n_frames = len(kp_norm)
    if len(catches) > 0:
        if (catches[0] > EDGE_CATCH_MARGIN
                and ka_sm[0] < ka_sm[catches[0]] + prom * 0.5):
            catches = np.concatenate([[0], catches])
        if (catches[-1] < n_frames - EDGE_CATCH_MARGIN
                and ka_sm[-1] < ka_sm[catches[-1]] + prom * 0.5):
            catches = np.concatenate([catches, [n_frames - 1]])

    return catches, ka_sm


def _cross_validate(wrist_catches, knee_catches, tol=CATCH_CROSS_VAL_TOL):
    """
    Compare wrist and knee catch detections frame-by-frame.

    Returns a list of warning strings (empty if agreement is good).
    The list is consumed by RowingScorer and exposed as scorer.warnings
    for the GUI to display as a visual alert.
    """
    warnings = []

    if len(wrist_catches) == 0:
        warnings.append(
            "Catch detection failed: no catch events found in wrist signal. "
            "Check that the camera captures the full stroke and that the rower "
            "is clearly visible throughout."
        )
        return warnings

    if len(knee_catches) == 0:
        warnings.append(
            "Catch cross-validation skipped: knee angle signal did not yield "
            "detectable catch events. Results rely on wrist signal alone."
        )
        return warnings

    n = min(len(wrist_catches), len(knee_catches))
    disagreements = [
        i for i in range(n)
        if abs(int(wrist_catches[i]) - int(knee_catches[i])) > tol
    ]

    if disagreements:
        max_diff = max(
            abs(int(wrist_catches[i]) - int(knee_catches[i]))
            for i in disagreements
        )
        warnings.append(
            f"Catch signal disagreement detected in {len(disagreements)} of {n} "
            f"cycles (wrist vs knee angle, max difference {max_diff} frames, "
            f"threshold {tol} frames). Wrist position is used as primary signal. "
            f"Verify camera angle and keypoint tracking quality."
        )

    return warnings


def _detect_finish_frames(kp_norm, cycles, near_side, facing, smooth_size=9):
    """
    Detect finish frame within each cycle from near-side wrist x-position.

    Finish = wrist at maximum displacement toward the body (opposite extreme
    from catch), searched in the window [25%, 80%] of each cycle to avoid
    the catch position and early recovery confusing the search.

    Facing +x: finish = minimum wrist x in window (pulled furthest left).
    Facing -x: finish = maximum wrist x in window (pulled furthest right).
    """
    prefix = 'l' if near_side == 'left' else 'r'
    wr_x   = uniform_filter1d(
        _jxy(kp_norm, f'{prefix}_wrist')[:, 0],
        size=smooth_size
    )

    finishes = []
    for (start, end) in cycles:
        clen   = end - start
        win_lo = start + max(1, int(0.25 * clen))
        win_hi = start + min(clen - 1, int(0.80 * clen))
        if win_hi <= win_lo:
            win_hi = win_lo + 1

        seg_x = wr_x[win_lo : win_hi + 1]

        local = int(np.argmin(seg_x)) if facing > 0 \
                else int(np.argmax(seg_x))
        global_idx = win_lo + local

        mid, _ = plateau_midpoint(wr_x, global_idx)
        finishes.append(mid)

    return np.array(finishes, dtype=int)


def detect_rowing_cycles(kp_norm, near_side):
    """
    Detect rowing stroke cycles using near-side wrist position (primary)
    and near-side knee angle (cross-validation).

    Parameters
    ----------
    kp_norm   : list[dict]  normalised keypoints from normalise_keypoints()
    near_side : str         'left' | 'right'

    Returns
    -------
    cycles        : list[(start_frame, end_frame)]
    catch_frames  : np.ndarray   wrist-based catch indices (catch-to-catch bounds)
    knee_angles   : np.ndarray   near-side knee angle time series (full recording)
    finish_frames : np.ndarray   per-cycle finish frame indices
    warnings      : list[str]    quality alerts for the GUI
    """
    prefix = 'l' if near_side == 'left' else 'r'
    wr_x   = _jxy(kp_norm, f'{prefix}_wrist')[:, 0]

    facing = rower_facing_sign(kp_norm, near_side)

    wrist_catches, _ = _wrist_catches(wr_x, facing)

    knee_catches, ka_smooth = _knee_catches(kp_norm, near_side)

    cv_warnings = _cross_validate(wrist_catches, knee_catches)

    hip_xy  = _jxy(kp_norm, f'{prefix}_hip')
    knee_xy = _jxy(kp_norm, f'{prefix}_knee')
    ank_xy  = _jxy(kp_norm, f'{prefix}_ankle')
    n = len(kp_norm)
    knee_angles = np.array([
        angle_at_joint(hip_xy[i], knee_xy[i], ank_xy[i])
        for i in range(n)
    ])

    catch_frames = wrist_catches

    if len(catch_frames) < 2:
        cv_warnings.append(
            "Fewer than 2 catch events detected — cannot form complete cycles. "
            "Ensure the recording contains at least one full stroke and that "
            "the rower's hand is visible throughout."
        )
        return [], catch_frames, knee_angles, np.array([], dtype=int), cv_warnings

    cycles = [
        (int(catch_frames[i]), int(catch_frames[i + 1]))
        for i in range(len(catch_frames) - 1)
    ]

    finish_frames = _detect_finish_frames(kp_norm, cycles, near_side, facing)

    return cycles, catch_frames, knee_angles, finish_frames, cv_warnings
