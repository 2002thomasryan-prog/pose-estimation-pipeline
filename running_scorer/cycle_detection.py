"""Detects gait cycles and their TD, MS, TO and MVP frames in a single forward pass through the keypoint series."""

import numpy as np

from .params import (
    CYCLE_MIN_FRAMES,
    PLATEAU_DELTA_TD,
    PLATEAU_DELTA_TO,
    PLATEAU_GAP_TOL,
    PLATEAU_MIN_DUR,
    DELTA_TO_ANGLE_REF,
    DELTA_TO_MAX_BOOST,
    DELTA_TO_POWER,
    STANCE_SWING_OUTLIER_STD_MULT,
    STANCE_SWING_OUTLIER_MIN_FRAMES,
    CROSSING_JUMP_LOW_Z,
    TD_APPROACH_OUTLIER_Z,
)


def _find_next_plateau(y, search_from, delta, gap_tol=None, min_dur=None):
    """
    Find the next stance plateau in signal y starting at or after search_from.

    A frame is in-plateau when y[i] >= y_peak - delta * (y_peak - y_min),
    where y_peak and y_min are computed over a local 40-frame window ahead of
    search_from (falls back to full signal if the window is too short) so the
    threshold adapts to each stride without being dominated by a single
    outlier peak elsewhere in the recording.

    Parameters
    ----------
    y           : 1-D float array  full-sequence signal
    search_from : int              start searching from this index
    delta       : float            fraction of signal range for threshold
    gap_tol     : int              max gap to bridge (default PLATEAU_GAP_TOL)
    min_dur     : int              min plateau length (default PLATEAU_MIN_DUR)

    Returns
    -------
    (plateau_start, plateau_end)  global indices, both inclusive
    or None if no valid plateau found after search_from
    """
    if gap_tol is None:
        gap_tol = PLATEAU_GAP_TOL
    if min_dur is None:
        min_dur = PLATEAU_MIN_DUR

    y = np.asarray(y, float)
    n = len(y)

    finite_all = y[np.isfinite(y)]
    if len(finite_all) == 0:
        return None

    _WIN = 40
    hi_win = min(n, search_from + _WIN)
    finite_local = y[search_from:hi_win]
    finite_local = finite_local[np.isfinite(finite_local)]
    if len(finite_local) < 3:
        finite_local = finite_all

    y_peak  = float(np.max(finite_local))
    y_min   = float(np.min(finite_local))
    y_range = y_peak - y_min
    if y_range < 1e-3:
        return None

    threshold = y_peak - delta * y_range

    i = search_from
    while i < n:
        if np.isfinite(y[i]) and y[i] >= threshold:
            j = i
            while j < n:
                if np.isfinite(y[j]) and y[j] >= threshold:
                    j += 1
                elif j + gap_tol < n and any(
                    np.isfinite(y[j + k]) and y[j + k] >= threshold
                    for k in range(1, gap_tol + 1)
                ):
                    j += 1
                else:
                    break
            plateau_end = j - 1
            length = plateau_end - i + 1
            if length >= min_dur:
                return (i, plateau_end)
            else:
                i = j
        else:
            i += 1

    return None


def _trim_to_right_edge(y, plateau_end, plateau_start):
    """
    Trim the right edge of a plateau to the tighter PLATEAU_DELTA_TO threshold.

    The wide delta (PLATEAU_DELTA_TD=0.20) used to find the plateau run
    intentionally includes the gradual toe-liftoff region so that the run is
    not broken by mid-stance foot roll.  After the run is identified, this
    function walks back from the right edge with the tighter delta
    (PLATEAU_DELTA_TO=0.09) to find the last frame where the signal is still
    clearly in the plateau — this is the validated TO location.

    The y_peak and y_range are computed over [plateau_start, plateau_end+5]
    (a small look-ahead to include the steep post-liftoff rise) so the
    threshold is consistent with the signal morphology at that stride.

    Parameters
    ----------
    y             : 1-D float array  full-sequence signal (heel or toe Y)
    plateau_end   : int              right edge from _find_next_plateau
    plateau_start : int              left edge from _find_next_plateau

    Returns
    -------
    int  refined TO frame index (≤ plateau_end)
    """
    y = np.asarray(y, float)
    local = y[plateau_start: min(len(y), plateau_end + 5)]
    finite = local[np.isfinite(local)]
    if len(finite) < 2:
        return plateau_end

    y_peak  = float(np.max(finite))
    y_min   = float(np.min(finite))
    y_range = y_peak - y_min
    if y_range < 1e-3:
        return plateau_end

    threshold = y_peak - PLATEAU_DELTA_TO * y_range
    to = plateau_end
    while to > plateau_start and (not np.isfinite(y[to]) or y[to] < threshold):
        to -= 1
    return to


def _trim_to_left_edge(y, plateau_start, plateau_end, delta_to=None):
    """
    Trim the left edge of a plateau to the tighter PLATEAU_DELTA_TO threshold.

    The wide delta (PLATEAU_DELTA_TD=0.20) used to find the plateau run
    intentionally starts a few frames before true ground contact, while the
    foot is still descending through the lower part of its range. This
    function walks FORWARD from plateau_start with the tighter delta
    (PLATEAU_DELTA_TO=0.09 by default) to find the first frame where the
    signal has actually reached near-peak (foot planted) — this is the
    validated TD location. Mirrors _trim_to_right_edge but for the left edge.

    y_peak and y_range are computed over [plateau_start, plateau_end]
    (the plateau run itself), matching the morphology used to find it.

    Parameters
    ----------
    delta_to : float or None
        Trim threshold to use in place of the module default PLATEAU_DELTA_TO.
        Callers doing subject-level TD detection pass the angle-scaled value
        from _scaled_delta_to() here; classify_strike_pattern()'s internal
        approx-TD estimate and TO trimming both leave this as None (fixed
        0.09) since scaling only applies to the main TD trim.

    Returns
    -------
    int  refined TD frame index (>= plateau_start, <= plateau_end)
    """
    delta = PLATEAU_DELTA_TO if delta_to is None else delta_to

    y = np.asarray(y, float)
    local = y[plateau_start: plateau_end + 1]
    finite = local[np.isfinite(local)]
    if len(finite) < 2:
        return plateau_start

    y_peak  = float(np.max(finite))
    y_min   = float(np.min(finite))
    y_range = y_peak - y_min
    if y_range < 1e-3:
        return plateau_start

    threshold = y_peak - delta * y_range
    td = plateau_start
    while td < plateau_end and (not np.isfinite(y[td]) or y[td] < threshold):
        td += 1
    return td


def _refine_td_onset(y, plateau_start, lookback=3, frac=0.35):
    """
    Refine TD from the plateau left edge to the true onset of foot descent.

    Steps backwards from plateau_start to find the last frame where the
    signal has not yet risen by frac * local_range over lookback frames.
    That frame is the onset of the rapid foot descent = true initial contact.
    """
    lo = max(0, plateau_start - 20)
    hi = min(len(y), plateau_start + 5)
    local = y[lo:hi]
    finite = local[np.isfinite(local)]
    if len(finite) < 3:
        return plateau_start

    lrange = float(np.max(finite) - np.min(finite))
    if lrange < 1e-3:
        return plateau_start

    threshold = frac * lrange

    for i in range(plateau_start, max(lookback, plateau_start - 20), -1):
        if i - lookback < 0:
            break
        rise = float(y[i]) - float(y[i - lookback])
        if rise < threshold:
            return i

    return plateau_start


_PAIR_WINDOW          = 15
_HEEL_LEAD_THR_FRAMES = 2
_ANGLE_THR_DEG        = 3.0


def _scaled_delta_to(mean_angle):
    """
    Scale PLATEAU_DELTA_TO for TD left-edge trimming based on strike profile
    flatness. Flat/near-midfoot contacts (small |mean_angle|) produce a
    shallower, more gradual plateau rise than pronounced heel-strikers, so
    the fixed 0.09 trim threshold lands slightly too early into that rise
    (observed as a systematic +1fr TD bias on flat-profile subjects).

    scale = 1 + (DELTA_TO_MAX_BOOST - 1) * ((ANGLE_REF - |angle|)/ANGLE_REF) ** POWER

    Sub-linear in angle (POWER=0.6 < 1): boost rises quickly just inside the
    ANGLE_REF=3.0° boundary and flattens out approaching 0°, rather than
    ramping linearly. This shape was fit directly against a real GT run
    (RTMPose, 4 videos / 33 cycles) after a first linear attempt (POWER=1.0)
    undershot — it left SB-1-Sag-Good (angle 1.22°) at 0.69 MAE instead of
    the achievable 0.00, and SB-Sag-BAD-2 (angle 2.91°) unimproved from
    0.4 MAE. POWER=0.6 / MAX_BOOST=4.4 reproduces both videos' true optimal
    trim threshold (found by direct grid search on the GT-annotated frames)
    to within 0.005.

    Pronounced heel-strikers (|mean_angle| >= DELTA_TO_ANGLE_REF, e.g. COG at
    9.8° and LF at 18.2°) get scale=1.0 regardless of POWER/MAX_BOOST — zero
    behaviour change, confirmed against GT.

    This is used ONLY for the TD trim in the main detection loop — never for
    TO trimming (the heel is reliably steep at lift-off regardless of strike
    profile) and never inside classify_strike_pattern()'s own approx-TD
    estimate (which must stay on the fixed threshold to avoid a circular
    dependency on the angle it's trying to measure).

    Parameters
    ----------
    mean_angle : float or None   from classify_strike_pattern()

    Returns
    -------
    float  effective PLATEAU_DELTA_TO to use for TD trimming this recording
    """
    if mean_angle is None:
        return PLATEAU_DELTA_TO
    angle = abs(mean_angle)
    if angle >= DELTA_TO_ANGLE_REF:
        return PLATEAU_DELTA_TO
    frac  = ((DELTA_TO_ANGLE_REF - angle) / DELTA_TO_ANGLE_REF) ** DELTA_TO_POWER
    scale = 1.0 + (DELTA_TO_MAX_BOOST - 1.0) * frac
    return PLATEAU_DELTA_TO * scale


def _collect_plateau_pairs(y_heel, y_toe, n_frames):
    """
    Sweep the whole recording once, pairing each heel plateau with the
    nearest toe plateau from the same stance (plateau starts within
    _PAIR_WINDOW frames of each other).

    Used ONLY for subject-level strike-pattern classification — this is
    a coarse whole-recording scan, not the validated per-cycle TD/TO
    detection used in detect_running_cycles().

    Returns
    -------
    list[((heel_start, heel_end), (toe_start, toe_end))]
    """
    def _sweep(y):
        plateaus = []
        pos = 0
        while pos < n_frames:
            p = _find_next_plateau(y, pos, delta=PLATEAU_DELTA_TD)
            if p is None:
                break
            plateaus.append(p)
            pos = p[1] + 1
        return plateaus

    heel_plateaus = _sweep(y_heel)
    toe_plateaus  = _sweep(y_toe)

    pairs = []
    used_toe = set()
    for hp in heel_plateaus:
        best_k, best_dist = None, None
        for k, tp in enumerate(toe_plateaus):
            if k in used_toe:
                continue
            dist = abs(tp[0] - hp[0])
            if dist <= _PAIR_WINDOW and (best_dist is None or dist < best_dist):
                best_k, best_dist = k, dist
        if best_k is not None:
            used_toe.add(best_k)
            pairs.append((hp, toe_plateaus[best_k]))
    return pairs


def _foot_inclination_deg(kp_seq, heel_idx, toe_idx, frame_g):
    """
    Angle of the heel→toe vector relative to horizontal at frame_g.
    Positive = toe held above heel (heel-strike signature).
    Negative = heel held above toe (forefoot/toe-strike signature).
    Returns None if either keypoint is missing at frame_g.
    """
    heel = kp_seq[frame_g, heel_idx, :2].astype(float)
    toe  = kp_seq[frame_g, toe_idx,  :2].astype(float)
    if not (np.all(np.isfinite(heel)) and np.all(np.isfinite(toe))):
        return None
    dx = float(toe[0] - heel[0])
    dy = float(toe[1] - heel[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(-dy, abs(dx))))


def classify_strike_pattern(kp_seq, kp_map, side='right'):
    """
    Classify the subject's strike pattern ONCE, using every paired stride
    in the recording. Combines plateau-lead timing with foot inclination
    at contact; defaults to heel-strike when the signals disagree or are
    both near-neutral (flat/midfoot contact).

    Returns
    -------
    dict
        'strike_type' : 'heel' | 'toe'
        'confidence'  : 'high' | 'medium' | 'low'
        'median_lead' : float | None   (frames; +ve = heel plateau leads)
        'mean_angle'  : float | None   (degrees; +ve = toe above heel)
        'n_strides'   : int            (paired strides used for classification)
    """
    prefix   = 'r' if side == 'right' else 'l'
    heel_idx = kp_map[f'{prefix}_heel']
    toe_idx  = kp_map[f'{prefix}_toe']

    y_heel   = kp_seq[:, heel_idx, 1].astype(float)
    y_toe    = kp_seq[:, toe_idx,  1].astype(float)
    n_frames = len(kp_seq)

    pairs = _collect_plateau_pairs(y_heel, y_toe, n_frames)
    if not pairs:
        return {'strike_type': 'heel', 'confidence': 'low',
                'median_lead': None, 'mean_angle': None, 'n_strides': 0}

    leads, angles = [], []
    for hp, tp in pairs:
        leads.append(float(tp[0] - hp[0]))
        if hp[0] <= tp[0]:
            approx_td = _trim_to_left_edge(y_heel, hp[0], hp[1])
        else:
            approx_td = _trim_to_left_edge(y_toe, tp[0], tp[1])
        angle = _foot_inclination_deg(kp_seq, heel_idx, toe_idx, approx_td)
        if angle is not None:
            angles.append(angle)

    median_lead = float(np.median(leads))
    mean_angle  = float(np.mean(angles)) if angles else None

    lead_says_heel  = median_lead > _HEEL_LEAD_THR_FRAMES
    lead_says_toe   = median_lead < -_HEEL_LEAD_THR_FRAMES
    angle_says_heel = mean_angle is not None and mean_angle > _ANGLE_THR_DEG
    angle_says_toe  = mean_angle is not None and mean_angle < -_ANGLE_THR_DEG

    if lead_says_heel and angle_says_heel:
        strike_type, confidence = 'heel', 'high'
    elif lead_says_toe and angle_says_toe:
        strike_type, confidence = 'toe', 'high'
    elif median_lead > 0 and (mean_angle is None or mean_angle >= 0):
        strike_type, confidence = 'heel', 'medium'
    elif median_lead < 0 and (mean_angle is None or mean_angle <= 0):
        strike_type, confidence = 'toe', 'medium'
    else:
        strike_type, confidence = 'heel', 'low'

    return {'strike_type': strike_type, 'confidence': confidence,
            'median_lead': median_lead, 'mean_angle': mean_angle,
            'n_strides': len(pairs)}


_MAX_SLAP_OFFSET   = 6
_MIN_SLAP_FRAC     = 0.35
_MIN_TOTAL_RISE_PX = 5.0


def _confirm_foot_slap(y_other, td_g, n_frames):
    """
    Confirm a TD candidate via the "foot-slap": the OTHER foot segment
    (toe for a heel-strike TD, heel for a toe-strike TD) should show a
    rapid Y increase within a few frames after contact, as the rest of
    the foot drops onto the ground.

    Normalised against the confirmation window's OWN total rise (not an
    external range) — a real slap is a single frame-to-frame jump that
    accounts for most of the window's total displacement, as opposed to
    a gradual, evenly-spread descent. A window with negligible total
    rise (near-flat/midfoot contact, where there's little foot rotation
    left to complete) is treated as unconfirmed rather than computing an
    unstable ratio on noise.

    This does NOT change td_g — it only flags whether the physiological
    cross-signal confirms the plateau-based TD, for diagnostics.

    Returns
    -------
    bool
    """
    hi = min(n_frames, td_g + _MAX_SLAP_OFFSET)
    window = y_other[td_g:hi]
    window = window[np.isfinite(window)]
    if len(window) < 2:
        return False

    total_rise = float(window[-1] - window[0])
    if total_rise < _MIN_TOTAL_RISE_PX:
        return False

    slap_mag = float(np.max(np.diff(window)))
    return slap_mag > _MIN_SLAP_FRAC * total_rise


def _find_td(y_primary, y_other, search_pos, n_frames, delta_to=None):
    """
    Find the next TD on the classified strike-pattern signal (y_primary =
    heel for heel-strikers, toe for toe-strikers), cross-checked against
    the foot-slap on y_other.

    Parameters
    ----------
    delta_to : float or None
        Angle-scaled TD trim threshold from _scaled_delta_to(), computed
        once per subject/recording by the caller and passed through
        unchanged for every TD in the recording. None falls back to the
        fixed PLATEAU_DELTA_TO.

    Returns
    -------
    (td_g, plateau, confirmed)  or  None if no plateau found
    """
    plateau = _find_next_plateau(y_primary, search_pos, delta=PLATEAU_DELTA_TD)
    if plateau is None:
        return None
    td_g = _trim_to_left_edge(y_primary, plateau[0], plateau[1], delta_to=delta_to)
    confirmed = _confirm_foot_slap(y_other, td_g, n_frames)
    return td_g, plateau, confirmed


def _find_ms(kp_seq, kp_map, side, td_g, to_g):
    """
    MS = argmin |ankle_x - hip_x| strictly between TD and TO.

    Parameters
    ----------
    kp_seq       : full keypoint array
    kp_map       : dict
    side         : str
    td_g, to_g   : global frame indices of TD and TO

    Returns
    -------
    int  global frame index of MS
    """
    prefix  = 'r' if side == 'right' else 'l'
    ank_idx = kp_map[f'{prefix}_ankle']
    hip_idx = kp_map[f'{prefix}_hip']

    lo = td_g + 1
    hi = to_g - 1

    if hi < lo:
        lo = td_g
        hi = to_g

    best_g    = lo
    best_dist = np.inf

    for g in range(lo, hi + 1):
        ank_x = float(kp_seq[g, ank_idx, 0])
        hip_x = float(kp_seq[g, hip_idx, 0])
        if not (np.isfinite(ank_x) and np.isfinite(hip_x)):
            continue
        d = abs(ank_x - hip_x)
        if d < best_dist:
            best_dist = d
            best_g    = g

    return best_g


def _find_mvp(kp_seq, kp_map, side, to_g, end_g):
    """
    MVP = first local minimum of |near_toe_Y - far_toe_Y| after TO.

    Searches from TO+1 up to end_g. Stops at the first local minimum
    (valley) in the bilateral toe Y difference signal — this is the
    scissors position. Does not search the entire remaining sequence,
    which would risk grabbing a minimum from a later stride.

    Falls back to the global minimum in (TO, end_g) if no local minimum
    is found (e.g. monotonically decreasing signal).

    Parameters
    ----------
    kp_seq       : full keypoint array
    kp_map       : dict
    side         : str
    to_g         : global frame index of TO
    end_g        : global frame index to stop searching (exclusive)

    Returns
    -------
    int  global frame index of MVP, or None if window is invalid
    """
    prefix = 'r' if side == 'right' else 'l'
    opp    = 'l' if prefix == 'r' else 'r'
    toe_i  = kp_map[f'{prefix}_toe']
    toe_oi = kp_map[f'{opp}_toe']

    lo = to_g + 1
    hi = min(end_g - 1, len(kp_seq) - 1)

    if hi < lo:
        return None

    diffs = []
    frames = []
    for g in range(lo, hi + 1):
        ny = float(kp_seq[g, toe_i,  1])
        fy = float(kp_seq[g, toe_oi, 1])
        if np.isfinite(ny) and np.isfinite(fy):
            diffs.append(abs(ny - fy))
            frames.append(g)

    if not diffs:
        return None

    diffs  = np.array(diffs)
    frames = np.array(frames)

    for k in range(1, len(diffs) - 1):
        if diffs[k] < diffs[k - 1] and diffs[k] < diffs[k + 1]:
            return int(frames[k])

    return int(frames[np.argmin(diffs)])


def detect_running_cycles(kp_seq, kp_map, side='right'):
    """
    Detect gait cycles via a single forward pass through the data.

    Pass structure:
      find TD → find MS (TD..TO) → find TO → find MVP (TO..next_TD)
      → find next TD → repeat

    Only complete cycles are returned. A partial cycle at the end of the
    recording (no closing TD found after MVP) is discarded.

    Strike pattern (heel vs. toe/forefoot) is classified ONCE for the
    whole recording via classify_strike_pattern() before this loop runs.
    Every TD in the recording is then found on that single classified
    signal (heel plateau for heel-strikers, toe plateau for toe-strikers),
    cross-checked against a foot-slap confirmation on the other segment.
    There is no per-cycle re-classification and no dual-signal arbitration
    fallback — a low-confidence classification simply defaults to the
    heel branch upstream, in classify_strike_pattern().

    Parameters
    ----------
    kp_seq : np.ndarray  shape (n_frames, n_kp, 3)
    kp_map : dict        canonical joint name -> keypoint index
    side   : str         'right' | 'left'

    Returns
    -------
    cycles       : list[(start_frame, end_frame)]  TD-to-TD global indices
    cycle_events : list[dict]
        Keys: 'td', 'to', 'ms', 'mvp' — LOCAL indices (0 = TD).
        'td' is always 0.
    """
    prefix   = 'r' if side == 'right' else 'l'
    heel_idx = kp_map[f'{prefix}_heel']
    toe_idx  = kp_map[f'{prefix}_toe']

    n_frames    = len(kp_seq)
    y_heel_full = kp_seq[:, heel_idx, 1].astype(float)
    y_toe_full  = kp_seq[:, toe_idx,  1].astype(float)

    strike_info  = classify_strike_pattern(kp_seq, kp_map, side=side)
    strike_type  = strike_info['strike_type']
    if strike_type == 'heel':
        y_primary, y_other, td_signal = y_heel_full, y_toe_full, 'heel'
    else:
        y_primary, y_other, td_signal = y_toe_full, y_heel_full, 'toe'

    td_delta_to = _scaled_delta_to(strike_info['mean_angle'])

    cycles       = []
    cycle_events = []
    cycle_lens_so_far = []

    search_pos = 0

    while search_pos < n_frames:

        found = _find_td(y_primary, y_other, search_pos, n_frames, delta_to=td_delta_to)
        if found is None:
            break
        td_g, td_plateau, td_confirmed = found

        crossing_jump = (float(y_primary[td_g] - y_primary[td_g - 1])
                          if td_g > 0 else float('nan'))

        to_plateau = _find_next_plateau(
            y_toe_full, td_g, delta=PLATEAU_DELTA_TD)
        if to_plateau is None:
            to_signal = 'heel'
            to_plateau = _find_next_plateau(y_heel_full, td_g, delta=PLATEAU_DELTA_TD)
            if to_plateau is None:
                search_pos = td_g + 1
                continue
            to_g = _trim_to_right_edge(y_heel_full, to_plateau[1], to_plateau[0])
        else:
            to_signal = 'toe'
            to_g = _trim_to_right_edge(y_toe_full, to_plateau[1], to_plateau[0])

        if td_g < search_pos:
            search_pos = td_g + 1
            continue

        ms_g = _find_ms(kp_seq, kp_map, side, td_g, to_g)


        stance_dur = max(to_g - td_g, 5)
        window_floor = (int(np.median(cycle_lens_so_far)) + 10
                        if cycle_lens_so_far else 0)
        mvp_search_end = min(n_frames, to_g + max(3 * stance_dur, window_floor))

        mvp_g = _find_mvp(kp_seq, kp_map, side, to_g, mvp_search_end)
        if mvp_g is None:
            search_pos = to_g + 1
            continue

        next_found = _find_td(y_primary, y_other, mvp_g + 1, n_frames, delta_to=td_delta_to)
        if next_found is None:
            break
        next_td_g, next_td_plateau, next_td_confirmed = next_found

        cycle_len = next_td_g - td_g
        if cycle_len < CYCLE_MIN_FRAMES:
            search_pos = next_td_g
            continue

        ank_idx = kp_map[f'{prefix}_ankle']
        ank_x   = kp_seq[:, ank_idx, 0].astype(float)

        def _speed(i, j):
            if 0 <= i < n_frames and 0 <= j < n_frames:
                v = ank_x[j] - ank_x[i]
                if np.isfinite(v):
                    return abs(float(v))
            return np.nan

        td_decel_ok = None
        spd_before_td = _speed(max(td_g - 4, 0), max(td_g - 1, 0))
        spd_at_td     = _speed(max(td_g - 1, 0), min(td_g + 1, n_frames - 1))
        if np.isfinite(spd_before_td) and np.isfinite(spd_at_td):
            td_decel_ok = bool(spd_at_td <= spd_before_td)

        to_accel_ok = None
        spd_at_to    = _speed(max(to_g - 1, 0), min(to_g + 1, n_frames - 1))
        spd_after_to = _speed(min(to_g + 1, n_frames - 1), min(to_g + 4, n_frames - 1))
        if np.isfinite(spd_at_to) and np.isfinite(spd_after_to):
            to_accel_ok = bool(spd_after_to >= spd_at_to)
        cycles.append((int(td_g), int(next_td_g)))
        cycle_events.append({
            'td':  0,
            'ms':  int(ms_g  - td_g),
            'to':  int(to_g  - td_g),
            'mvp': int(mvp_g - td_g),
            'strike_type':        td_signal,
            'strike_confidence':  strike_info['confidence'],
            'strike_median_lead': strike_info['median_lead'],
            'strike_mean_angle':  strike_info['mean_angle'],
            'td_delta_to_used':   float(td_delta_to),
            'td_plateau_start':   int(td_plateau[0]),
            'td_plateau_end':     int(td_plateau[1]),
            'td_confirmed':       bool(td_confirmed),
            'to_signal':          to_signal,
            'to_plateau_start':   int(to_plateau[0]),
            'to_plateau_end':     int(to_plateau[1]),
            'cycle_len':          int(cycle_len),
            'stance_frames':      int(to_g - td_g),
            'swing_frames':       int(next_td_g - to_g),
            'crossing_jump':      crossing_jump,
            'td_decel_ok':        td_decel_ok,
            'to_accel_ok':        to_accel_ok,
            'mvp_in_flight':      bool(to_g < mvp_g < next_td_g),
        })
        cycle_lens_so_far.append(cycle_len)

        search_pos = next_td_g

    if len(cycle_events) >= 3:
        lens = np.array([e['cycle_len'] for e in cycle_events], dtype=float)
        med  = float(np.median(lens))
        std  = float(np.std(lens))
        for e in cycle_events:
            e['cycle_len_outlier'] = bool(std > 1e-6 and abs(e['cycle_len'] - med) > 2 * std)
    else:
        for e in cycle_events:
            e['cycle_len_outlier'] = False

    if len(cycle_events) >= 3:
        stances = np.array([e['stance_frames'] for e in cycle_events], dtype=float)
        swings  = np.array([e['swing_frames']  for e in cycle_events], dtype=float)
        jumps   = np.array([e['crossing_jump']  for e in cycle_events], dtype=float)

        approach_rates = []
        for (td_g, _next_td_g), e in zip(cycles, cycle_events):
            y_primary = y_heel_full if e.get('td_signal') == 'heel' else y_toe_full
            lo = max(0, td_g - 4)
            approach_rates.append(float(y_primary[td_g] - y_primary[lo]))
        approach_rates = np.array(approach_rates, dtype=float)

        stance_med, stance_std = float(np.median(stances)), float(np.std(stances))
        swing_med,  swing_std  = float(np.median(swings)),  float(np.std(swings))
        jump_finite = jumps[np.isfinite(jumps)]
        jump_mean = float(np.mean(jump_finite)) if len(jump_finite) else 0.0
        jump_std  = float(np.std(jump_finite))  if len(jump_finite) else 0.0
        appr_finite = approach_rates[np.isfinite(approach_rates)]
        appr_med = float(np.median(appr_finite)) if len(appr_finite) else 0.0
        appr_std = float(np.std(appr_finite))    if len(appr_finite) else 0.0

        stance_thresh = max(STANCE_SWING_OUTLIER_STD_MULT * stance_std, STANCE_SWING_OUTLIER_MIN_FRAMES)
        swing_thresh  = max(STANCE_SWING_OUTLIER_STD_MULT * swing_std,  STANCE_SWING_OUTLIER_MIN_FRAMES)

        for e, jump, appr in zip(cycle_events, jumps, approach_rates):
            stance_outlier = bool(abs(e['stance_frames'] - stance_med) > stance_thresh)
            swing_outlier  = bool(abs(e['swing_frames']  - swing_med)  > swing_thresh)
            if np.isfinite(jump) and jump_std > 1e-6:
                jz = float((jump - jump_mean) / jump_std)
            else:
                jz = 0.0
            low_jump = bool(jz < CROSSING_JUMP_LOW_Z)

            if np.isfinite(appr) and appr_std > 1e-6:
                appr_z = float((appr - appr_med) / appr_std)
            else:
                appr_z = 0.0
            approach_outlier = bool(abs(appr_z) > TD_APPROACH_OUTLIER_Z)

            e['stance_frames_outlier'] = stance_outlier
            e['swing_frames_outlier']  = swing_outlier
            e['crossing_jump_z']       = jz
            e['low_crossing_jump']     = low_jump
            e['td_approach_rate']      = float(appr)
            e['td_approach_z']         = appr_z
            e['td_approach_outlier']   = approach_outlier
            e['td_confidence_flag']    = bool(
                (stance_outlier or swing_outlier) and (low_jump or approach_outlier))
    else:
        for e in cycle_events:
            e['stance_frames_outlier'] = False
            e['swing_frames_outlier']  = False
            e['crossing_jump_z']       = 0.0
            e['low_crossing_jump']     = False
            e['td_approach_rate']      = 0.0
            e['td_approach_z']         = 0.0
            e['td_approach_outlier']   = False
            e['td_confidence_flag']    = False

    return cycles, cycle_events
