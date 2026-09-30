"""Running-specific keypoint cleaning: per-joint MAD spike detection (reversals only) with linear interpolation."""

import numpy as np


_THRESHOLDS = {
    'hip':   {'k': 1.2, 'abs_thresh': 8},
    'toe':   {'k': 3.5, 'abs_thresh': 40},
    'heel':  {'k': 3.5, 'abs_thresh': 40},
    'ankle': {'k': 3.0, 'abs_thresh': 20},
    'knee':  {'k': 4.0, 'abs_thresh': 20},
}

_JOINT_GROUP = {
    'l_hip':   'hip',   'r_hip':   'hip',
    'l_toe':   'toe',   'r_toe':   'toe',
    'l_heel':  'heel',  'r_heel':  'heel',
    'l_ankle': 'ankle', 'r_ankle': 'ankle',
    'l_knee':  'knee',  'r_knee':  'knee',
}


def _robust_mad(arr):
    """Median absolute deviation of a 1-D signal (finite values only)."""
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 1.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(max(mad, 1e-6))


def _detect_spikes(arr, k, abs_thresh):
    """
    Flag single-frame spikes in a 1-D signal.

    A frame i is flagged only when it is a genuine REVERSAL — the signal
    moves one way into frame i and back the other way out of it — AND that
    reversal is large by either measure:
      (a) relative  — |jump_in| > k*MAD and |jump_out| > k*MAD
      (b) absolute  — |jump_in| > abs_thresh and |jump_out| > abs_thresh

    The mask is then expanded by ±1 frame to catch shoulder artefacts.

    Both jump_in and jump_out are defined as consistent forward differences
    (arr[i]-arr[i-1] and arr[i+1]-arr[i]) so that sustained, fast, genuine
    motion in one direction — e.g. the heel accelerating down into ground
    contact on a pronounced heel-striker — is never mistaken for a spike.
    Fixed June/July 2026: the previous version defined jump_out as
    arr[i]-arr[i+1] (reversed), which meant sign(jump_in) != sign(jump_out)
    was mathematically true for ANY monotonic ramp, not just true spikes,
    and the absolute-guard branch fired on magnitude alone with no reversal
    requirement at all. Together these silently discarded real fast-motion
    frames whenever a joint's frame-to-frame speed exceeded abs_thresh
    (confirmed on LF's heel_y, which moves 30-55px/frame into contact,
    against the old abs_thresh=40) — a real bug worth fixing regardless of
    downstream impact on any one metric. NOTE: confirmed empirically that
    this fix does NOT change LF's TD frames (the affected frames happened to
    fall entirely below the TD trim threshold either way) — LF's TD bias has
    a separate, still-open cause; see cycle_detection.py discussion. This
    fix stands on its own merit as a real correctness bug, independent of
    that investigation. A real single-frame glitch (detector jumps to a
    wrong location for one frame then returns) is still a reversal
    regardless of magnitude, so this catches exactly what the absolute
    guard was meant to catch without also catching sustained real motion.

    Parameters
    ----------
    arr        : 1-D array of float
    k          : float   MAD multiplier
    abs_thresh : float   absolute pixel jump threshold

    Returns
    -------
    bad : np.ndarray bool, same length as arr
    """
    arr = np.asarray(arr, float)
    n   = len(arr)
    bad = np.zeros(n, dtype=bool)

    d   = np.diff(arr)
    mad = _robust_mad(d)
    rel = k * mad

    for i in range(1, n - 1):
        if not (np.isfinite(arr[i - 1]) and np.isfinite(arr[i])
                and np.isfinite(arr[i + 1])):
            continue

        j_in  = arr[i] - arr[i - 1]
        j_out = arr[i + 1] - arr[i]

        if j_in == 0 or j_out == 0 or np.sign(j_in) == np.sign(j_out):
            continue

        if (abs(j_in) > rel and abs(j_out) > rel) or \
           (abs(j_in) > abs_thresh and abs(j_out) > abs_thresh):
            bad[i] = True

    for i in np.where(bad)[0]:
        if i > 0:       bad[i - 1] = True
        if i < n - 1:   bad[i + 1] = True

    return bad


def _interpolate_bad(arr, bad):
    """
    Linear interpolation over flagged (bad) samples.
    Non-finite finite values are also treated as bad.
    """
    arr  = np.asarray(arr, float)
    bad  = np.asarray(bad, dtype=bool)
    n    = len(arr)
    good = (~bad) & np.isfinite(arr)

    if good.sum() < 2:
        return arr.copy()

    out = arr.copy()
    idx = np.arange(n)
    out[~good] = np.interp(idx[~good], idx[good], arr[good])
    return out


def clean_keypoints_running(kp_seq, kp_map):
    """
    Clean running keypoints using per-joint MAD spike detection.

    Only the joints listed in _JOINT_GROUP are processed; all other joints
    pass through unchanged (no risk of corrupting shoulder/elbow etc).
    X and Y coordinates are cleaned independently.

    Parameters
    ----------
    kp_seq : np.ndarray  shape (n_frames, n_kp, 3)
    kp_map : dict        canonical joint name → keypoint index

    Returns
    -------
    np.ndarray  shape (n_frames, n_kp, 3)  cleaned copy (float64)
    """
    cleaned = np.array(kp_seq, dtype=float)

    processed_indices = set()

    for joint_name, group in _JOINT_GROUP.items():
        idx = kp_map.get(joint_name)
        if idx is None or idx in processed_indices:
            continue

        t = _THRESHOLDS[group]
        dims = [1] if group in ('toe', 'heel', 'ankle') else range(2)
        for dim in dims:
            arr = cleaned[:, idx, dim]
            bad = _detect_spikes(arr, t['k'], t['abs_thresh'])
            if bad.any():
                cleaned[:, idx, dim] = _interpolate_bad(arr, bad)

        processed_indices.add(idx)

    return cleaned
