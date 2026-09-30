"""Event-based running scorer: computes all twelve metrics at TD, MS, TO and MVP across every detected gait cycle and applies traffic-light scoring."""

import re
import numpy as np

from scorer_utils import (
    KEYPOINT_MAPS, KP_MMPOSE, _build_kp_sam3d,
    normalise_keypoints,
    _parse_target, _acceptance_interval, _deviation,
    _traffic_light, _score_metric,
)
from side_detection import detect_lateral_side

from .params import (METRIC_RULES, RUNNING_PARAMS, CYCLE_MIN_FRAMES,
                     EXCLUDE_LOW_CONFIDENCE_CYCLES)
from .keypoint_cleaner import clean_keypoints_running
from .cycle_detection import detect_running_cycles


def _loc_norm(a, b, c):
    """
    Translate to local coordinates centred at b, scale by mean segment length.
    Mirrors Emma's localize_and_normalize() for numerical stability on large
    pixel coordinates.
    Returns (a_local, zero_origin, c_local) — all as float64 arrays.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    a_loc = a - b
    c_loc = c - b
    scale = np.mean([np.linalg.norm(a_loc), np.linalg.norm(c_loc)])
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return a_loc / scale, np.zeros(2, float), c_loc / scale


def _angle(a, b, c):
    """
    Included angle at vertex B (degrees), computed via localize_and_normalize.
    Returns np.nan if any point is non-finite or segments are degenerate.
    """
    if (np.any(~np.isfinite(a)) or np.any(~np.isfinite(b))
            or np.any(~np.isfinite(c))):
        return np.nan
    al, _bl, cl = _loc_norm(a, b, c)
    na = np.linalg.norm(al)
    nc = np.linalg.norm(cl)
    if na < 1e-6 or nc < 1e-6:
        return np.nan
    dot = np.clip(np.dot(al / na, cl / nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def _angle_between(v1, v2):
    """Angle between two free vectors v1 and v2 (degrees). NaN-safe."""
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return np.nan
    dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def _signed_acute_from_vertical(vx, vy):
    """
    Signed acute angle between vector (vx, vy) and upward vertical (0, -1),
    in image coordinates where Y increases downward.

    Sign convention: positive if the vector leans right (vx > 0).

    Returns degrees, or np.nan if the vector is degenerate.
    """
    mag = np.hypot(vx, vy)
    if mag < 1e-6 or not np.isfinite(mag):
        return np.nan
    cos_theta = np.clip((-vy) / mag, -1.0, 1.0)
    theta = float(np.degrees(np.arccos(cos_theta)))
    theta_acute = min(theta, 180.0 - theta)
    return theta_acute if vx > 0 else -theta_acute


def _get(frame_dict, name):
    """
    Extract [x, y] for a canonical joint name from a normalised frame dict.
    Returns [nan, nan] if the joint is missing.
    """
    v = frame_dict.get(name)
    if v is None:
        return np.array([np.nan, np.nan])
    arr = np.asarray(v, float)
    return arr[:2].copy() if arr.shape[0] >= 2 else np.array([np.nan, np.nan])


def _average_metric_list(records):
    """
    Aggregate per-cycle metric records into a single summary dict.

    Numeric targets  : average raw actuals, then re-score the average.
    Qualitative      : majority vote — mean score ≥ 0.75 → GREEN,
                       ≥ 0.25 → AMBER, else RED.

    Parameters
    ----------
    records : list[dict]   each dict from a single cycle's score_* function

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
                'actual': round(avg_actual, 2) if avg_actual is not None else None,
                'target': target, 'uncertainty': unc,
                'deviation': None, 'n_cycles': len(records)}

    if avg_actual is None:
        return {'score': 0.0, 'label': 'RED', 'actual': None,
                'target': target, 'uncertainty': unc,
                'deviation': None, 'n_cycles': len(records)}

    s, lbl, dev = _traffic_light(avg_actual, target, unc)
    return {'score': s, 'label': lbl,
            'actual': round(avg_actual, 2),
            'target': target, 'uncertainty': unc,
            'deviation': dev, 'n_cycles': len(records)}


def _ns_fs(near_side):
    """Return (near_prefix, far_prefix) → ('l','r') or ('r','l')."""
    return ('l', 'r') if near_side == 'left' else ('r', 'l')


def _score_mvp(frame, near_side):
    """
    Score three metrics at the Max Velocity Phase frame.

    near-side = trailing (in float/swing after push-off)
    far-side  = leading  (in stance)
    """
    ns, fs = _ns_fs(near_side)
    out = {}

    key  = 'thigh_sep_mvp'
    rule = METRIC_RULES[key]
    l_hip   = _get(frame, 'l_hip')
    l_knee  = _get(frame, 'l_knee')
    r_knee  = _get(frame, 'r_knee')
    l_thigh = l_knee - l_hip
    r_thigh = r_knee - l_hip
    if np.all(np.isfinite(l_thigh)) and np.all(np.isfinite(r_thigh)):
        actual = round(_angle_between(l_thigh, r_thigh), 1)
    else:
        actual = None
    s, lbl, dev = _score_metric(actual, rule)
    out[key] = {'score': s, 'label': lbl, 'actual': actual,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': dev}

    key  = 'trailing_shin_mvp'
    rule = METRIC_RULES[key]
    tr_knee  = _get(frame, f'{ns}_knee')
    tr_ankle = _get(frame, f'{ns}_ankle')
    tr_toe   = _get(frame, f'{ns}_toe')
    shin_vec = tr_ankle - tr_knee
    shin_vy  = -shin_vec[1]
    shin_angle = (float(np.degrees(np.arctan2(shin_vy, shin_vec[0])))
                  if np.all(np.isfinite(shin_vec)) else np.nan)
    toe_below_knee = (np.isfinite(tr_toe[1]) and np.isfinite(tr_knee[1])
                      and float(tr_toe[1]) > float(tr_knee[1]))
    shin_downward  = np.isfinite(shin_vy) and shin_vy < 0
    passed = bool(toe_below_knee or shin_downward)
    s, lbl, _ = _score_metric(None, rule, qualitative_passed=passed)
    chart_val = 1.0 if passed else 0.0
    out[key] = {'score': s, 'label': lbl, 'actual': chart_val,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': None,
                'note': (f'shin_angle={shin_angle:.1f}°'
                         if np.isfinite(shin_angle) else '')}

    key  = 'leading_knee_mvp'
    rule = METRIC_RULES[key]
    ld_ankle = _get(frame, f'{fs}_ankle')
    ld_knee  = _get(frame, f'{fs}_knee')
    ld_hip   = _get(frame, f'{fs}_hip')
    actual = None
    if all(np.all(np.isfinite(v)) for v in [ld_ankle, ld_knee, ld_hip]):
        actual = round(_angle(ld_ankle, ld_knee, ld_hip), 1)
    s, lbl, dev = _score_metric(actual, rule)
    out[key] = {'score': s, 'label': lbl, 'actual': actual,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': dev}

    return out


def _score_toe_off(frame, near_side):
    """
    Score three metrics at Toe-Off.

    near-side = trailing (pushing off)
    far-side  = leading  (ahead, in stance)
    """
    ns, fs = _ns_fs(near_side)
    out = {}

    key  = 'trailing_foot_to'
    rule = METRIC_RULES[key]
    tr_toe  = _get(frame, f'{ns}_toe')
    tr_heel = _get(frame, f'{ns}_heel')
    actual = None
    if np.all(np.isfinite(tr_toe)) and np.all(np.isfinite(tr_heel)):
        fv = tr_toe - tr_heel
        fv_up = np.array([fv[0], -fv[1]])
        raw_ang = float(np.degrees(np.arctan2(fv_up[1], fv_up[0])))
        actual  = round(abs(90.0 - abs(raw_ang)), 1)
    s, lbl, dev = _score_metric(actual, rule)
    out[key] = {'score': s, 'label': lbl, 'actual': actual,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': dev}

    key  = 'thigh_sep_to'
    rule = METRIC_RULES[key]
    l_hip   = _get(frame, 'l_hip')
    l_knee  = _get(frame, 'l_knee')
    r_knee  = _get(frame, 'r_knee')
    l_thigh = l_knee - l_hip
    r_thigh = r_knee - l_hip
    if np.all(np.isfinite(l_thigh)) and np.all(np.isfinite(r_thigh)):
        actual = round(_angle_between(l_thigh, r_thigh), 1)
    else:
        actual = None
    s, lbl, dev = _score_metric(actual, rule)
    out[key] = {'score': s, 'label': lbl, 'actual': actual,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': dev}

    key  = 'leading_ankle_to'
    rule = METRIC_RULES[key]
    ld_toe   = _get(frame, f'{fs}_toe')
    ld_ankle = _get(frame, f'{fs}_ankle')
    ld_knee  = _get(frame, f'{fs}_knee')
    actual = None
    if all(np.all(np.isfinite(v)) for v in [ld_toe, ld_ankle, ld_knee]):
        foot_vec  = ld_toe   - ld_ankle
        shank_vec = ld_knee  - ld_ankle
        actual    = round(_angle_between(foot_vec, shank_vec), 1)
    s, lbl, dev = _score_metric(actual, rule)
    out[key] = {'score': s, 'label': lbl, 'actual': actual,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': dev}

    return out


def _score_touchdown(frame, near_side):
    """
    Score four metrics at Touchdown (initial contact).

    near-side = landing = leading
    far-side  = trailing (still behind)
    """
    ns, fs = _ns_fs(near_side)
    out = {}

    key  = 'trailing_knee_td'
    rule = METRIC_RULES[key]
    tr_hip  = _get(frame, f'{fs}_hip')
    tr_knee = _get(frame, f'{fs}_knee')
    actual = None
    if np.all(np.isfinite(tr_hip)) and np.all(np.isfinite(tr_knee)):
        if near_side == 'right':
            posterior = tr_knee[0] < tr_hip[0]
        else:
            posterior = tr_knee[0] > tr_hip[0]
        actual = 1.0 if posterior else 0.0
    s = 2 if actual == 1.0 else (0 if actual == 0.0 else None)
    lbl = 'GREEN' if s == 2 else 'RED'
    dev = None
    out[key] = {'score': s, 'label': lbl, 'actual': actual,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': dev}

    key  = 'landing_foot_td'
    rule = METRIC_RULES[key]
    ld_toe  = _get(frame, f'{ns}_toe')
    ld_heel = _get(frame, f'{ns}_heel')
    actual = None
    if np.all(np.isfinite(ld_toe)) and np.all(np.isfinite(ld_heel)):
        vx = ld_heel[0] - ld_toe[0]
        vy = ld_heel[1] - ld_toe[1]
        fv_up = np.array([vx, -vy])
        raw_ang = float(np.degrees(np.arctan2(fv_up[1], fv_up[0])))
        theta_acute = min(abs(raw_ang), 180.0 - abs(raw_ang))
        actual = round(-theta_acute if vx > 0 else theta_acute, 1)
    s, lbl, dev = _score_metric(actual, rule)
    out[key] = {'score': s, 'label': lbl, 'actual': actual,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': dev}

    key  = 'shin_ext_td'
    rule = METRIC_RULES[key]
    ld_knee  = _get(frame, f'{ns}_knee')
    ld_ankle = _get(frame, f'{ns}_ankle')
    actual = None
    if np.all(np.isfinite(ld_knee)) and np.all(np.isfinite(ld_ankle)):
        vx = ld_ankle[0] - ld_knee[0]
        vy = ld_ankle[1] - ld_knee[1]
        raw = _signed_acute_from_vertical(vx, vy)
        actual = round(-raw if near_side == 'left' else raw, 1)
    s, lbl, dev = _score_metric(actual, rule)
    out[key] = {'score': s, 'label': lbl, 'actual': actual,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': dev}

    key  = 'torso_tilt_td'
    rule = METRIC_RULES[key]
    ns_hip = _get(frame, f'{ns}_hip')
    ns_sho = _get(frame, f'{ns}_shoulder')
    actual = None
    if np.all(np.isfinite(ns_hip)) and np.all(np.isfinite(ns_sho)):
        vx = ns_sho[0] - ns_hip[0]
        vy = ns_sho[1] - ns_hip[1]
        signed = _signed_acute_from_vertical(vx, vy)
        if np.isfinite(signed):
            actual = round(abs(signed), 1)
    s, lbl, dev = _score_metric(actual, rule)
    out[key] = {'score': s, 'label': lbl, 'actual': actual,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': dev}

    return out


def _score_midstance(frame, near_side):
    """
    Score two metrics at Mid-Stance.

    near-side = stance leg (support phase)
    """
    ns, fs = _ns_fs(near_side)
    out = {}

    key  = 'thigh_gap_ms'
    rule = METRIC_RULES[key]
    l_hip      = _get(frame, 'l_hip')
    l_knee     = _get(frame, 'l_knee')
    r_knee     = _get(frame, 'r_knee')
    fs_knee_pt = _get(frame, f'{fs}_knee')
    ns_knee_pt = _get(frame, f'{ns}_knee')
    l_thigh = l_knee - l_hip
    r_thigh = r_knee - l_hip
    actual = None
    if (np.all(np.isfinite(l_thigh)) and np.all(np.isfinite(r_thigh))
            and np.all(np.isfinite(fs_knee_pt)) and np.all(np.isfinite(ns_knee_pt))):
        angle_mag = _angle_between(l_thigh, r_thigh)
        if near_side == 'right':
            knee_diff = fs_knee_pt[0] - ns_knee_pt[0]
        else:
            knee_diff = ns_knee_pt[0] - fs_knee_pt[0]
        sign = 1.0 if knee_diff >= 0 else -1.0
        actual = round(sign * angle_mag, 1)
    s, lbl, dev = _score_metric(actual, rule)
    out[key] = {'score': s, 'label': lbl, 'actual': actual,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': dev}

    key  = 'sinking_ms'
    rule = METRIC_RULES[key]
    ns_toe  = _get(frame, f'{ns}_toe')
    ns_knee = _get(frame, f'{ns}_knee')
    actual = None
    if np.all(np.isfinite(ns_toe)) and np.all(np.isfinite(ns_knee)):
        vx = ns_knee[0] - ns_toe[0]
        vy = ns_knee[1] - ns_toe[1]
        raw = _signed_acute_from_vertical(vx, vy)
        actual = round(-raw if near_side == 'left' else raw, 1)
    s, lbl, dev = _score_metric(actual, rule)
    out[key] = {'score': s, 'label': lbl, 'actual': actual,
                'target': rule['target'], 'uncertainty': rule['uncertainty'],
                'deviation': dev}

    return out


class RunningScorer:
    """
    Biomechanical scorer for running.

    Uses Emma's event-based traffic-light framework:
    score at TD / MS / TO / MVP in every detected cycle,
    then average the raw actuals across cycles and re-score.

    Parameters
    ----------
    keypoints_array : np.ndarray  shape (n_frames, n_keypoints, 2+)
    backend         : str         'mmpose' | 'yolov8' | 'sam3d'
    fps             : int
    dominant_side   : str         'right' | 'left'  (fallback if side detection fails)
    video_path      : str | None
    log_cb          : callable | None
    """

    SPORT = 'running'

    def __init__(self, keypoints_array, backend='mmpose',
                 fps=30, dominant_side='right', video_path=None, log_cb=None):

        self.raw_kp        = np.asarray(keypoints_array, dtype=float)
        self.backend       = backend.lower()
        self.fps           = fps
        self.dominant_side = dominant_side
        self.log           = log_cb or print
        self.video_path    = video_path

        self.kp_map = KEYPOINT_MAPS.get(self.backend, KP_MMPOSE)
        if self.backend == 'sam3d':
            self.kp_map = _build_kp_sam3d()

        self.kp_arr        = None
        self.kp_seq        = None

        self.near_side     = None
        self.cycles        = None
        self.cycle_events  = None

        self.excluded_cycles: list       = []
        self.excluded_cycle_events: list = []

        self.cycle_actuals: dict = {}

        self._cycle_records: dict = {}

        self.emma_report: dict = {}

        self.mean_values  = {}
        self.param_scores = {}
        self.total_score  = 0.0

        self._td_frames  = []
        self._to_frames  = []
        self._ms_frames  = []
        self._mvp_frames = []

        self.annotated_video_path = None
        self.warnings = []


    def preprocess(self):
        self.log(f'[RunningScorer] Cleaning {len(self.raw_kp)} frames '
                 f'({self.backend}) with MAD spike detection...')
        self.kp_arr = clean_keypoints_running(self.raw_kp, self.kp_map)
        self.kp_seq = normalise_keypoints(self.kp_arr, self.backend)
        self.log('[RunningScorer] Preprocessing complete.')
        return self

    def detect_side(self):
        if self.kp_arr is None:
            self.preprocess()
        self.near_side = detect_lateral_side(
            self.kp_arr, self.backend, video_path=self.video_path)
        if self.near_side is None:
            self.near_side = self.dominant_side
            self.log(f'[RunningScorer] Side detection inconclusive — '
                     f'using dominant_side={self.dominant_side}')
        self.log(f'[RunningScorer] Near side: {self.near_side}')
        return self

    def detect_cycles(self):
        if self.near_side is None:
            self.detect_side()
        self.cycles, self.cycle_events = detect_running_cycles(
            self.kp_arr, self.kp_map, side=self.near_side)
        self.log(f'[RunningScorer] {len(self.cycles)} gait cycles detected.')
        if not self.cycles:
            raise RuntimeError(
                '[RunningScorer] No cycles detected — check keypoint quality '
                'and that the full stride is visible.')

        self.all_cycles       = list(self.cycles)
        self.all_cycle_events = list(self.cycle_events)

        self.excluded_cycles, self.excluded_cycle_events = [], []
        if EXCLUDE_LOW_CONFIDENCE_CYCLES:
            kept_c, kept_e, dropped = [], [], []
            for i, (c, ev) in enumerate(zip(self.cycles, self.cycle_events)):
                if ev.get('td_confidence_flag'):
                    self.excluded_cycles.append(c)
                    self.excluded_cycle_events.append(ev)
                    dropped.append(i + 1)
                else:
                    kept_c.append(c)
                    kept_e.append(ev)
            self.cycles, self.cycle_events = kept_c, kept_e
            if dropped:
                self.log(
                    f'[RunningScorer] Excluded {len(dropped)} low-confidence '
                    f'cycle(s) (original #{dropped}); '
                    f'{len(kept_c)} retained and renumbered.')
                self.warnings.append(
                    f'{len(dropped)} cycle(s) excluded (low TD confidence): '
                    f'original #{dropped}.')
            if not self.cycles:
                raise RuntimeError(
                    '[RunningScorer] All detected cycles were flagged as '
                    'low-confidence — none retained. Disable '
                    'EXCLUDE_LOW_CONFIDENCE_CYCLES or check keypoint quality.')
        return self

    def compute(self):
        if self.cycles is None:
            self.detect_cycles()

        all_keys = [p['key'] for p in RUNNING_PARAMS]
        for k in all_keys:
            self.cycle_actuals[k]  = []
            self._cycle_records[k] = []

        td_frames, to_frames, ms_frames, mvp_frames = [], [], [], []

        for i, (start, end) in enumerate(self.cycles):
            if (end - start) < CYCLE_MIN_FRAMES:
                continue

            ev        = self.cycle_events[i]
            td_local  = ev.get('td')
            to_local  = ev.get('to')
            ms_local  = ev.get('ms')
            mvp_local = ev.get('mvp')

            td_g  = start + (td_local  if td_local  is not None else 0)
            to_g  = (start + to_local)  if to_local  is not None else None
            ms_g  = (start + ms_local)  if ms_local  is not None else None
            mvp_g = (start + mvp_local) if mvp_local is not None else None

            td_frames.append(int(td_g))
            to_frames.append(int(to_g)  if to_g  is not None else None)
            ms_frames.append(int(ms_g)  if ms_g  is not None else None)
            mvp_frames.append(int(mvp_g) if mvp_g is not None else None)

            td_frame  = self.kp_seq[td_g]
            to_frame  = self.kp_seq[to_g]  if to_g  is not None else None
            ms_frame  = self.kp_seq[ms_g]  if ms_g  is not None else None
            mvp_frame = self.kp_seq[mvp_g] if mvp_g is not None else None

            def _accumulate(event_scores):
                for k, rec in event_scores.items():
                    self._cycle_records[k].append({**rec,
                        'target':      METRIC_RULES[k]['target'],
                        'uncertainty': METRIC_RULES[k]['uncertainty']})
                    self.cycle_actuals[k].append(
                        rec['actual'] if rec['actual'] is not None else np.nan)

            if mvp_frame is not None:
                _accumulate(_score_mvp(mvp_frame, self.near_side))
            if to_frame is not None:
                _accumulate(_score_toe_off(to_frame, self.near_side))
            _accumulate(_score_touchdown(td_frame, self.near_side))
            if ms_frame is not None:
                _accumulate(_score_midstance(ms_frame, self.near_side))

        self._td_frames  = td_frames
        self._to_frames  = to_frames
        self._ms_frames  = ms_frames
        self._mvp_frames = mvp_frames

        self.emma_report = {
            k: _average_metric_list(self._cycle_records[k])
            for k in all_keys
        }

        for k in all_keys:
            rec = self.emma_report[k]
            self.param_scores[k] = rec['score']
            self.mean_values[k]  = (rec['actual']
                                    if rec['actual'] is not None
                                    else float('nan'))

        valid = [v for v in self.param_scores.values()
                 if isinstance(v, (int, float)) and np.isfinite(float(v))]
        self.total_score = float(np.mean(valid)) * 100.0 if valid else 0.0

        return self

    def run(self, save_path=None, detail_path=None):
        """Full pipeline: preprocess → detect_side → detect_cycles → compute."""
        return self.preprocess().detect_side().detect_cycles().compute()


    def print_summary(self):
        if not self.param_scores:
            self.compute()

        _ICON = {'GREEN': '✓', 'AMBER': '~', 'RED': '✗', '—': '?'}

        print(f'\n{"=" * 76}')
        print(f'  RUNNING BIOMECHANICAL ASSESSMENT  [{self.backend.upper()}]')
        print(f'  Cycles analysed: {len(self.cycles)}   |   '
              f'Overall Score: {self.total_score:.1f} / 100')
        print(f'{"=" * 76}')

        current_event = None
        event_order   = ['mvp', 'to', 'td', 'ms']
        event_labels  = {'mvp': 'MAX VELOCITY PHASE', 'to': 'TOE-OFF',
                         'td': 'TOUCHDOWN',           'ms': 'MID-STANCE'}

        for param in RUNNING_PARAMS:
            key    = param['key']
            suffix = key.rsplit('_', 1)[-1]
            if suffix != current_event:
                current_event = suffix
                print(f'\n  ── {event_labels.get(suffix, suffix.upper())} ──')

            rec  = self.emma_report.get(key, {})
            lbl  = rec.get('label', '—')
            mv   = self.mean_values.get(key, float('nan'))
            icon = _ICON.get(lbl, '?')
            unit = param['unit']

            if isinstance(mv, float) and np.isfinite(mv):
                mv_s = f'{mv:.1f}°' if unit == '°' else f'{mv:.2f}'
            else:
                mv_s = 'N/A'

            tgt   = rec.get('target', '—')
            dev   = rec.get('deviation')
            dev_s = (f'  (dev {dev:+.1f}°)'
                     if dev is not None and np.isfinite(float(dev)) and dev != 0
                     else '')

            print(f'  {icon} {lbl:<6}  {param["label"]:<30}  '
                  f'{mv_s:<12}  target: {tgt}{dev_s}')

        print(f'\n{"=" * 76}\n')
        return self

    def to_dict(self):
        if not self.param_scores:
            self.compute()

        def _safe(v):
            if v is None:
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return None if (np.isnan(f) or not np.isfinite(f)) else f

        def _safe_list(lst):
            return [_safe(v) for v in lst]

        raw_traces = {
            'ic_frames':  self._td_frames,
            'td_frames':  self._td_frames,
            'to_frames':  self._to_frames,
            'ms_frames':  self._ms_frames,
            'mvp_frames': self._mvp_frames,
        }
        for p in RUNNING_PARAMS:
            k = p['key']
            raw_traces[k] = _safe_list(self.cycle_actuals.get(k, []))

        return {
            'sport':                self.SPORT,
            'backend':              self.backend,
            'dominant_side':        self.near_side or self.dominant_side,
            'fps':                  int(self.fps),
            'n_frames':             (int(len(self.kp_seq))
                                     if self.kp_seq is not None else 0),
            'n_cycles':             len(self.cycles),
            'n_excluded':           len(self.excluded_cycles),
            'total_score':          round(self.total_score, 2),
            'param_scores':         {k: round(float(v) * 100, 2)
                                     for k, v in self.param_scores.items()},
            'mean_values':          {k: _safe(v)
                                     for k, v in self.mean_values.items()},
            'params_meta':          RUNNING_PARAMS,
            'raw_traces':           raw_traces,
            'video_path':           self.video_path,
            'annotated_video_path': self.annotated_video_path,
            'warnings':             self.warnings,
        }
