#!/usr/bin/env python3
"""Shared keypoint index maps, geometry helpers and scoring functions used by all three sport scorers."""

import re
import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


KP_MMPOSE = {
    'nose': 0,
    'l_eye': 1,  'r_eye': 2,
    'l_ear': 3,  'r_ear': 4,
    'l_shoulder': 5,  'r_shoulder': 6,
    'l_elbow': 7,     'r_elbow': 8,
    'l_wrist': 9,     'r_wrist': 10,
    'l_hip': 11,      'r_hip': 12,
    'l_knee': 13,     'r_knee': 14,
    'l_ankle': 15,    'r_ankle': 16,
    'l_heel': 92, 'l_foot_mid': 93, 'l_toe': 94,
    'r_heel': 95, 'r_foot_mid': 96, 'r_toe': 97,
}

KP_YOLO = {
    'nose': 0,
    'l_eye': 1,  'r_eye': 2,
    'l_ear': 3,  'r_ear': 4,
    'l_shoulder': 5,  'r_shoulder': 6,
    'l_elbow': 7,     'r_elbow': 8,
    'l_wrist': 9,     'r_wrist': 10,
    'l_hip': 11,      'r_hip': 12,
    'l_knee': 13,     'r_knee': 14,
    'l_ankle': 15,    'r_ankle': 16,
    'l_heel': 15, 'l_foot_mid': 15, 'l_toe': 15,
    'r_heel': 16, 'r_foot_mid': 16, 'r_toe': 16,
}

KP_RTMPOSE = {
    'nose': 0,
    'l_eye': 1,   'r_eye': 2,
    'l_ear': 3,   'r_ear': 4,
    'l_shoulder': 5,  'r_shoulder': 6,
    'l_elbow': 7,     'r_elbow': 8,
    'l_wrist': 9,     'r_wrist': 10,
    'l_hip': 11,      'r_hip': 12,
    'l_knee': 13,     'r_knee': 14,
    'l_ankle': 15,    'r_ankle': 16,
    'l_toe': 20,      'r_toe': 21,
    'l_foot_mid': 22, 'r_foot_mid': 23,
    'l_heel': 24,     'r_heel': 25,
}

KP_SAM3D = {}

KEYPOINT_MAPS = {
    'mmpose':   KP_MMPOSE,
    'yolov8':   KP_YOLO,
    'rtmpose':  KP_RTMPOSE,
    'sam3d':    KP_SAM3D,
}

SPORT_COLORS = {
    'running': '#E63946',
    'rowing':  '#2196F3',
    'cycling': '#4CAF50',
}

KP_MHR70_FALLBACK = {
    'nose':       0,
    'l_eye':      1,   'r_eye':      3,
    'l_ear':      2,   'r_ear':      4,
    'l_shoulder': 7,   'r_shoulder': 7,
    'l_elbow':    23,  'r_elbow':    23,
    'l_wrist':    49,  'r_wrist':    49,
    'l_hip':      9,   'r_hip':      10,
    'l_knee':     11,  'r_knee':     12,
    'l_ankle':    13,  'r_ankle':    14,
    'l_heel':     13,  'r_heel':     14,
    'l_toe':      13,  'r_toe':      14,
    'l_foot_mid': 13,  'r_foot_mid': 14,
}


def _build_kp_sam3d():
    """
    Build the MHR70 keypoint index map from the name→index dict loaded by
    pose_extractor._load_sam3d().  Falls back to KP_MHR70_FALLBACK (empirically
    verified hardcoded indices) if SAM-3D has not been loaded yet — NOT KP_YOLO,
    which uses COCO-17 indices that are wrong for MHR70 (e.g. COCO wrist=9/10
    collides with MHR70 hip=9/10).

    MHR70 has dedicated toe and heel keypoints (left_big_toe, right_big_toe,
    left_heel, right_heel, left_small_toe, right_small_toe).  When the model
    is loaded these are resolved by name from the metadata, giving correct
    distinct indices for all foot joints.
    """
    try:
        from pose_extractor import get_sam3d_name_to_idx
        n2i = get_sam3d_name_to_idx()
        if not n2i:
            return KP_MHR70_FALLBACK.copy()
    except ImportError:
        return KP_MHR70_FALLBACK.copy()

    required = {
        'nose':         'nose',
        'l_eye':        'left_eye',        'r_eye':       'right_eye',
        'l_ear':        'left_ear',        'r_ear':       'right_ear',
        'l_shoulder':   'left_shoulder',   'r_shoulder':  'right_shoulder',
        'l_elbow':      'left_elbow',      'r_elbow':     'right_elbow',
        'l_wrist':      'left_wrist',      'r_wrist':     'right_wrist',
        'l_hip':        'left_hip',        'r_hip':       'right_hip',
        'l_knee':       'left_knee',       'r_knee':      'right_knee',
        'l_ankle':      'left_ankle',      'r_ankle':     'right_ankle',
        'l_heel':       'left_heel',       'r_heel':      'right_heel',
        'l_toe':        'left_big_toe',    'r_toe':       'right_big_toe',
        'l_foot_mid':   'left_small_toe',  'r_foot_mid':  'right_small_toe',
    }

    kp = {}
    ankle_l = n2i.get('left_ankle',  n2i.get('left_foot', 0))
    ankle_r = n2i.get('right_ankle', n2i.get('right_foot', 0))

    for scorer_key, mhr70_name in required.items():
        if mhr70_name in n2i:
            kp[scorer_key] = n2i[mhr70_name]
        else:
            kp[scorer_key] = ankle_l if scorer_key.startswith('l_') else ankle_r

    return kp


CANONICAL_JOINTS = [
    'nose',
    'l_eye', 'r_eye', 'l_ear', 'r_ear',
    'l_shoulder', 'r_shoulder',
    'l_elbow',    'r_elbow',
    'l_wrist',    'r_wrist',
    'l_hip',      'r_hip',
    'l_knee',     'r_knee',
    'l_ankle',    'r_ankle',
    'l_heel',     'r_heel',
    'l_toe',      'r_toe',
    'l_foot_mid', 'r_foot_mid',
]


def normalise_keypoints(kp_array, backend):
    """
    Convert a raw keypoint array from any backend into a list of per-frame dicts.

    Parameters
    ----------
    kp_array : np.ndarray  shape (n_frames, n_kp, 2+)
    backend  : str         'mmpose' | 'yolov8' | 'sam3d'

    Returns
    -------
    List[Dict[str, np.ndarray]]
        One dict per frame.  Each dict maps every canonical joint name
        (see CANONICAL_JOINTS) to a float64 array of shape (2,) containing
        [x, y] in the original pixel coordinate space.

    Notes
    -----
    - For SAM-3D-Body the empirically-verified MHR70 fallback map is used
      when the model has not been loaded (scoring-only mode).
    - Where a backend does not have a dedicated keypoint for a canonical
      name (e.g. YOLO has no heel) the fallback index defined in that
      backend's map is used (typically the nearest available joint).
    - Both l_wrist and r_wrist point to the same physical joint index for
      SAM-3D side-view recordings; the scorer uses whichever is most
      appropriate for the view.
    """
    backend_key = backend.lower()
    if backend_key == 'sam3d':
        kp_map = _build_kp_sam3d()
    else:
        kp_map = KEYPOINT_MAPS.get(backend_key, KP_MMPOSE)

    kp_array = np.asarray(kp_array, dtype=float)
    frames = []
    for frame in kp_array:
        d = {}
        for name in CANONICAL_JOINTS:
            idx = kp_map.get(name)
            if idx is not None and idx < frame.shape[0]:
                d[name] = frame[idx, :2].copy()
            else:
                d[name] = np.zeros(2, dtype=float)
        frames.append(d)
    return frames


def _jxy(kp_norm, name):
    """
    Extract an (n_frames, 2) array for a named joint from a normalised dict list.

    Parameters
    ----------
    kp_norm : List[Dict[str, np.ndarray]]  output of normalise_keypoints()
    name    : str  canonical joint name

    Returns
    -------
    np.ndarray  shape (n_frames, 2)
    """
    return np.array([f[name] for f in kp_norm], dtype=float)


def angle_at_joint(A, B, C):
    """Interior angle (degrees) at joint B, formed by segments BA and BC."""
    A, B, C = np.asarray(A[:2], float), np.asarray(B[:2], float), np.asarray(C[:2], float)
    BA = A - B
    BC = C - B
    denom = np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(BA, BC) / denom, -1.0, 1.0))))


def angle_from_vertical(top, bottom):
    """
    Angle (degrees) of segment top→bottom from vertical.
    In image coords Y increases downward.  Returns unsigned angle.
    """
    v = np.asarray(bottom[:2], float) - np.asarray(top[:2], float)
    vertical = np.array([0.0, 1.0])
    denom = np.linalg.norm(v) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(v / denom, vertical), -1.0, 1.0))))


def angle_from_horizontal(A, B):
    """Angle (degrees) of segment A→B from horizontal.  Returns unsigned angle."""
    v = np.asarray(B[:2], float) - np.asarray(A[:2], float)
    horizontal = np.array([1.0, 0.0])
    denom = np.linalg.norm(v) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(v / denom, horizontal), -1.0, 1.0))))


def signed_trunk_angle(hip_pt, sh_pt):
    """
    Signed trunk angle from vertical (degrees), using atan2.

    Positive = forward lean (shoulder ahead of hip in +x direction).
    Negative = backward lean.
    """
    dx = float(sh_pt[0] - hip_pt[0])
    dy = float(sh_pt[1] - hip_pt[1])
    return float(np.degrees(np.arctan2(dx, -dy)))


def segment_length(A, B):
    return float(np.linalg.norm(np.asarray(A[:2], float) - np.asarray(B[:2], float)))


def clean_keypoints(kp_seq, jump_threshold=0.15):
    """
    Detect large frame-to-frame keypoint jumps (tracking failures) and
    linearly interpolate over them.

    jump_threshold: max allowed displacement as fraction of frame height.
    kp_seq: np.ndarray (n_frames, n_kp, 2+)
    Returns cleaned array (float64).
    """
    kp_seq = np.asarray(kp_seq, dtype=float)
    n_frames, n_kp = kp_seq.shape[:2]

    y_vals = kp_seq[:, :, 1].ravel()
    y_vals = y_vals[np.isfinite(y_vals)]
    frame_height = float(np.ptp(y_vals)) if len(y_vals) > 0 else 1.0
    max_jump = jump_threshold * frame_height

    cleaned = kp_seq.copy()

    for kp_idx in range(n_kp):
        xy = cleaned[:, kp_idx, :2]
        bad = np.zeros(n_frames, dtype=bool)

        diffs = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        bad[1:] = diffs > max_jump

        if not bad.any():
            continue

        good = ~bad
        if good.sum() < 2:
            continue

        frames = np.arange(n_frames)
        for dim in range(2):
            f_interp = interp1d(
                frames[good], xy[good, dim],
                kind='linear', bounds_error=False,
                fill_value=(xy[good, dim][0], xy[good, dim][-1])
            )
            cleaned[bad, kp_idx, dim] = f_interp(frames[bad])

    return cleaned


def score_value(value, param_def):
    """
    Map a measured value to a 0-1 score.
      - Within optimal range  → 1.0
      - Outside optimal range → linear decay to 0 at (range width) outside the bound
      - NaN                   → 0.0

    Supports direction: 'range', 'minimize', 'maximize'
    """
    if value is None or np.isnan(value):
        return 0.0

    lo, hi = param_def['optimal']
    direction = param_def['direction']
    rng = max(hi - lo, 1e-9)

    if lo <= value <= hi:
        return 1.0

    if direction == 'minimize':
        if value <= hi:
            return 1.0
        return max(0.0, 1.0 - (value - hi) / rng)

    elif direction == 'maximize':
        if value >= lo:
            return 1.0
        return max(0.0, 1.0 - (lo - value) / rng)

    else:
        if value < lo:
            return max(0.0, 1.0 - (lo - value) / rng)
        else:
            return max(0.0, 1.0 - (value - hi) / rng)


def aggregate_scores(param_results, param_defs):
    """
    Average per-cycle values, score each parameter, compute total (0-100).

    Returns:
        mean_values:  {key: mean measured value}
        param_scores: {key: 0-1 score}
        total_score:  float 0-100
    """
    mean_values  = {}
    param_scores = {}

    for param in param_defs:
        key  = param['key']
        vals = param_results.get(key, [])
        mv   = float(np.mean(vals)) if vals else float('nan')
        mean_values[key]  = mv
        param_scores[key] = score_value(mv, param)

    valid = [v for v in param_scores.values() if not np.isnan(v)]
    total = float(np.mean(valid)) * 100.0 if valid else 0.0

    return mean_values, param_scores, total


def _parse_target(t, default_tol=10):
    """
    Parse a target string into a structured spec dict.

    Recognised formats (degree symbol optional, spaces ignored):
      "90°±10°"    → {'type': 'pm',     'base': 90,   'tol': 10}
      "80-100°"    → {'type': 'range',  'lo': 80,     'hi': 100}
      "-20-20°"    → {'type': 'range',  'lo': -20,    'hi': 20}
      ">110°"      → {'type': 'gt',     'thr': 110}
      "<15°"       → {'type': 'le',     'thr': 15}
      qualitative  → None   (no digits in string)
    """
    if t is None:
        return None
    t = str(t).strip()
    if not any(ch.isdigit() for ch in t):
        return None

    tc = (t.replace(' ', '')
           .replace('−', '-').replace('–', '-').replace('—', '-')
           .replace('°', ''))

    if '±' in tc:
        parts = tc.split('±')
        if len(parts) == 2:
            try:
                return {'type': 'pm', 'base': float(parts[0]),
                        'tol': float(parts[1])}
            except ValueError:
                pass

    m = re.match(r'^(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)$', tc)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return {'type': 'range', 'lo': lo, 'hi': hi}

    if tc.startswith('>'):
        try:
            return {'type': 'gt', 'thr': float(tc[1:])}
        except ValueError:
            pass

    if tc.startswith('<') or tc.startswith('≤'):
        try:
            return {'type': 'le', 'thr': float(tc[1:])}
        except ValueError:
            pass

    m = re.search(r'-?\d+(?:\.\d+)?', tc)
    if m:
        return {'type': 'single', 'val': float(m.group(0)), 'tol': default_tol}

    return None


def _acceptance_interval(spec):
    """Return (lo, hi) acceptance interval for a parsed spec dict."""
    if spec is None:
        return None
    t = spec['type']
    if t == 'range':
        return spec['lo'], spec['hi']
    if t == 'pm':
        return spec['base'] - spec['tol'], spec['base'] + spec['tol']
    if t == 'single':
        return spec['val'] - spec['tol'], spec['val'] + spec['tol']
    if t == 'gt':
        return spec['thr'], np.inf
    if t == 'le':
        return -np.inf, spec['thr']
    return None


def _deviation(actual, spec):
    """
    Signed deviation of actual from nearest edge of the acceptance interval.
    Returns 0.0 if inside, negative if below lo, positive if above hi.
    Returns None if spec is None or actual is non-finite.
    """
    if spec is None or actual is None or not np.isfinite(actual):
        return None
    lo, hi = _acceptance_interval(spec)
    if lo <= actual <= hi:
        return 0.0
    return (actual - lo) if actual < lo else (actual - hi)


def _traffic_light(actual, target_str, uncertainty=None):
    """
    Classify a measurement against a target string.

    The uncertainty band is symmetric about each threshold: a measurement is
    AMBER whenever its band straddles a boundary of the acceptance interval,
    from either side.  A value sitting just inside a threshold whose band
    crosses outward is no better determined than the mirror-image value just
    outside, and both are reported as uncertain.

    An earlier form of this function returned GREEN as soon as the measured
    value fell inside the interval, before the band was consulted, which made
    the rule asymmetric: every borderline pass was reported more confidently
    than a borderline failure of identical margin.

    GREEN therefore now means "inside the interval by more than the
    measurement uncertainty", not merely "inside the interval".

    Infinite bounds (from '>x' and '<x' targets) are skipped: a one-sided
    target has only one real threshold to straddle.

    Returns
    -------
    score : float  1.0 = GREEN, 0.5 = AMBER, 0.0 = RED
    label : str    'GREEN' | 'AMBER' | 'RED' | '—'
    dev   : float | None
    """
    spec = _parse_target(target_str)
    if spec is None:
        return None, '—', None

    if actual is None or not np.isfinite(float(actual)):
        return 0.0, 'RED', None

    dev = _deviation(actual, spec)

    if uncertainty is not None and float(uncertainty) > 0:
        acc = _acceptance_interval(spec)
        if acc is not None:
            u = float(uncertainty)
            a_lo, a_hi = actual - u, actual + u
            t_lo, t_hi = acc
            straddles = ((np.isfinite(t_lo) and a_lo < t_lo < a_hi) or
                         (np.isfinite(t_hi) and a_lo < t_hi < a_hi))
            if straddles:
                return 0.5, 'AMBER', dev

    if dev == 0.0:
        return 1.0, 'GREEN', 0.0

    return 0.0, 'RED', dev


def _score_metric(actual, rule, qualitative_passed=None):
    """
    Score a single numeric or qualitative metric reading.

    Parameters
    ----------
    actual             : float | None   measured value
    rule               : dict           entry from METRIC_RULES
                                        must have 'target' and 'uncertainty' keys
    qualitative_passed : bool | None    pass/fail for qualitative targets only

    Returns
    -------
    score : float   1.0 / 0.5 / 0.0
    label : str     'GREEN' | 'AMBER' | 'RED'
    dev   : float | None
    """
    target = rule.get('target', '')
    unc    = rule.get('uncertainty')
    spec   = _parse_target(target)

    if spec is None:
        if qualitative_passed is None:
            return 0.0, 'RED', None
        s   = 1.0 if qualitative_passed else 0.0
        lbl = 'GREEN' if qualitative_passed else 'RED'
        return s, lbl, None

    s, lbl, dev = _traffic_light(actual, target, unc)
    return s, lbl, dev


def _average_metric_list(records):
    """
    Aggregate per-cycle metric records into a single summary dict.

    Numeric targets  : average raw actuals, then re-score the average.
    Qualitative      : majority vote — mean score ≥ 0.75 → GREEN,
                       ≥ 0.25 → AMBER, else RED.

    Parameters
    ----------
    records : list[dict]   each produced by a per-cycle _score_metric call

    Returns
    -------
    dict with keys: score, label, actual, target, uncertainty, deviation, n_cycles
    """
    if not records:
        return {'score': 0.0, 'label': 'RED', 'actual': None,
                'target': '—', 'uncertainty': None, 'deviation': None,
                'n_cycles': 0}

    first  = next((r for r in records if isinstance(r, dict)), {})
    target = first.get('target', '—')
    unc    = first.get('uncertainty')

    actual_vals = [
        r['actual'] for r in records
        if isinstance(r.get('actual'), (int, float))
        and np.isfinite(float(r['actual']))
    ]
    avg_actual = float(np.mean(actual_vals)) if actual_vals else None

    spec = _parse_target(target)

    if spec is None:
        score_vals = [r['score'] for r in records
                      if isinstance(r.get('score'), (int, float))]
        mean_sc = float(np.mean(score_vals)) if score_vals else 0.0
        if mean_sc >= 0.75:
            agg_s, agg_lbl = 1.0, 'GREEN'
        elif mean_sc >= 0.25:
            agg_s, agg_lbl = 0.5, 'AMBER'
        else:
            agg_s, agg_lbl = 0.0, 'RED'
        return {'score': agg_s, 'label': agg_lbl,
                'actual': round(avg_actual, 4) if avg_actual is not None else None,
                'target': target, 'uncertainty': unc,
                'deviation': None, 'n_cycles': len(records)}

    if avg_actual is None:
        return {'score': 0.0, 'label': 'RED', 'actual': None,
                'target': target, 'uncertainty': unc,
                'deviation': None, 'n_cycles': len(records)}

    s, lbl, dev = _traffic_light(avg_actual, target, unc)
    return {'score': s, 'label': lbl,
            'actual': round(avg_actual, 4),
            'target': target, 'uncertainty': unc,
            'deviation': dev, 'n_cycles': len(records)}
