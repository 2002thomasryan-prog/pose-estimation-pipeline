"""Cycling pipeline orchestrator: detects crank cycles, computes all seven metrics per cycle and applies traffic-light scoring."""

import numpy as np
from scipy.ndimage import uniform_filter1d

from scorer_utils import (
    KEYPOINT_MAPS, KP_MMPOSE, _build_kp_sam3d,
    clean_keypoints, normalise_keypoints,
    angle_at_joint, angle_from_vertical,
    _score_metric, _average_metric_list,
)

from side_detection import detect_lateral_side

from .params               import CYCLING_PARAMS, METRIC_RULES, CYCLE_MIN_FRAMES
from .cycle_detection      import detect_cycling_cycles
from .knee_angle_bdc       import compute_knee_angle_bdc
from .knee_angle_tdc       import compute_knee_angle_tdc
from .pelvic_stability     import compute_pelvic_stability
from .ankle_rom            import compute_ankle_rom
from .spinal_flexion       import compute_spinal_flexion
from .trunk_angle          import compute_trunk_angle
from .bilateral_symmetry   import compute_bsi


class CyclingScorer:
    """
    Biomechanical scorer for cycling.

    Parameters
    ----------
    keypoints_array : np.ndarray  shape (n_frames, n_keypoints, 2+)
    backend         : str         'mmpose' | 'yolov8' | 'rtmpose' | 'sam3d'
    fps             : int
    dominant_side   : str         'right' | 'left'  (used for BDC detection)
    log_cb          : callable    optional logging function
    """

    SPORT = 'cycling'

    def __init__(self, keypoints_array, backend='mmpose',
                 fps=30, dominant_side='right', video_path=None,
                 annotated_video_path=None, log_cb=None):

        self.raw_kp        = np.asarray(keypoints_array, dtype=float)
        self.backend       = backend.lower()
        self.fps           = fps
        self.dominant_side = dominant_side
        self.log           = log_cb or print

        self.kp_map = KEYPOINT_MAPS.get(self.backend, KP_MMPOSE)
        if self.backend == 'sam3d':
            self.kp_map = _build_kp_sam3d()

        self.kp_seq        = None
        self.kp_arr        = None
        self.kp_norm       = None
        self.near_side     = None
        self.cycles        = None
        self.event_frames  = None
        self.tdc_frames    = None

        self.param_results = None
        self.mean_values   = None
        self.param_scores  = None
        self.total_score   = None

        self.cycle_actuals  = {}
        self._cycle_records = {}
        self.emma_report    = {}

        self.video_path           = video_path
        self.annotated_video_path = annotated_video_path or video_path

        self.warnings = []


    def preprocess(self):
        self.log(f'[CyclingScorer] Cleaning {len(self.raw_kp)} frames ({self.backend})...')
        self.kp_seq  = clean_keypoints(self.raw_kp)
        self.kp_arr  = self.kp_seq
        self.kp_norm = normalise_keypoints(self.kp_seq, self.backend)
        self.log('[CyclingScorer] Preprocessing complete.')
        return self

    def detect_side(self):
        if self.kp_seq is None:
            self.preprocess()
        self.near_side = detect_lateral_side(
            self.kp_seq, self.backend, video_path=self.video_path)
        self.log(f'[CyclingScorer] Near side detected: {self.near_side}')
        return self

    def detect_cycles(self):
        if self.near_side is None:
            self.detect_side()

        self.cycles, self.event_frames = detect_cycling_cycles(
            self.kp_seq, self.kp_map, side=self.near_side)
        self.log(f'[CyclingScorer] {len(self.cycles)} crank cycles detected.')

        prefix = 'r' if self.near_side == 'right' else 'l'
        ank_idx = self.kp_map[f'{prefix}_ankle']
        tdc_list = []
        for (s, e) in self.cycles:
            seg_y     = self.kp_seq[s:e, ank_idx, 1]
            tdc_local = int(np.argmin(seg_y))
            tdc_list.append(s + tdc_local)
        self.tdc_frames = tdc_list

        if len(self.cycles) == 0:
            raise RuntimeError(
                '[CyclingScorer] No cycles detected — check keypoint data quality.')
        return self

    def compute(self):
        if self.cycles is None:
            self.detect_cycles()

        all_keys = [p['key'] for p in CYCLING_PARAMS]

        for k in all_keys:
            self.cycle_actuals[k]  = []
            self._cycle_records[k] = []

        res = {k: [] for k in all_keys}

        f0    = self.kp_seq[0]
        l_sh  = self.kp_map['l_shoulder']; r_sh  = self.kp_map['r_shoulder']
        l_ank = self.kp_map['l_ankle'];    r_ank = self.kp_map['r_ankle']
        sh_y0  = (f0[l_sh, 1] + f0[r_sh, 1]) / 2
        ank_y0 = (f0[l_ank, 1] + f0[r_ank, 1]) / 2
        body_h = abs(ank_y0 - sh_y0) + 1e-9

        bsi, bsi_n_valid, bsi_n_total = compute_bsi(
            self.kp_seq, self.kp_map, self.cycles,
            self.event_frames, self.near_side)

        if bsi is None:
            self.warnings.append(
                f'Bilateral symmetry not reported: only {bsi_n_valid}/'
                f'{bsi_n_total} cycles had usable far-side landmarks.')
            self.log('[CyclingScorer] BSI skipped — insufficient far-side '
                     'keypoint confidence.')
        elif bsi_n_valid < bsi_n_total:
            self.warnings.append(
                f'Bilateral symmetry from {bsi_n_valid}/{bsi_n_total} cycles '
                f'(far-side landmarks unusable in the remainder).')

        for (start, end) in self.cycles:
            seg = self.kp_seq[start:end]
            if len(seg) < CYCLE_MIN_FRAMES:
                continue

            knee_val    = compute_knee_angle_bdc(seg, self.kp_map, self.near_side)
            tdc_val     = compute_knee_angle_tdc(seg, self.kp_map, self.near_side)
            pelvis_val  = compute_pelvic_stability(seg, self.kp_map, body_h)
            ankle_val   = compute_ankle_rom(seg, self.kp_map)
            spinal_val  = compute_spinal_flexion(seg, self.kp_map)
            trunk_val   = compute_trunk_angle(seg, self.kp_map)
            bsi_val     = bsi

            if tdc_val is not None:
                res['knee_angle_tdc'].append(tdc_val)
            res['knee_angle_bdc'].append(knee_val)
            res['pelvic_stability'].append(pelvis_val)
            res['ankle_rom'].append(ankle_val)
            res['spinal_flexion'].append(spinal_val)
            res['trunk_angle'].append(trunk_val)
            res['bilateral_symmetry'].append(bsi_val)

            def _acc(key, actual):
                if actual is None:
                    return
                try:
                    if not np.isfinite(float(actual)):
                        return
                except (TypeError, ValueError):
                    return
                rule = METRIC_RULES[key]
                s, lbl, dev = _score_metric(float(actual), rule)
                record = {
                    'score':       s,
                    'label':       lbl,
                    'actual':      round(float(actual), 4),
                    'target':      rule['target'],
                    'uncertainty': rule['uncertainty'],
                    'deviation':   dev,
                }
                self._cycle_records[key].append(record)
                self.cycle_actuals[key].append(float(actual))

            _acc('knee_angle_tdc',     tdc_val)
            _acc('knee_angle_bdc',     knee_val)
            _acc('pelvic_stability',   pelvis_val)
            _acc('ankle_rom',          ankle_val)
            _acc('spinal_flexion',     spinal_val)
            _acc('trunk_angle',        trunk_val)
            _acc('bilateral_symmetry', bsi_val)

        self.emma_report = {
            k: _average_metric_list(self._cycle_records[k])
            for k in all_keys
        }

        self.param_results = res

        self.param_scores = {}
        self.mean_values  = {}
        for k in all_keys:
            rec = self.emma_report[k]
            self.param_scores[k] = rec['score']
            mv = rec['actual']
            self.mean_values[k]  = (float(mv) if mv is not None
                                    else float('nan'))

        valid = [v for v in self.param_scores.values()
                 if isinstance(v, (int, float)) and np.isfinite(float(v))]
        self.total_score = float(np.mean(valid)) * 100.0 if valid else 0.0

        self.log(f'[CyclingScorer] Scoring complete. '
                 f'Total: {self.total_score:.1f}/100')
        return self


    def _build_raw_traces(self):
        kp  = self.kp_seq
        km  = self.kp_map
        n   = len(kp)

        if kp is None or n == 0:
            return {}

        r_hip = km['r_hip']; l_hip = km['l_hip']
        r_kn  = km['r_knee']; l_kn = km['l_knee']
        r_ank = km['r_ankle']; l_ank = km['l_ankle']
        l_sh  = km['l_shoulder']; r_sh = km['r_shoulder']

        r_knee = np.array([
            180.0 - angle_at_joint(kp[i, r_hip, :2], kp[i, r_kn, :2], kp[i, r_ank, :2])
            for i in range(n)
        ])
        l_knee = np.array([
            180.0 - angle_at_joint(kp[i, l_hip, :2], kp[i, l_kn, :2], kp[i, l_ank, :2])
            for i in range(n)
        ])
        trunk = np.array([
            angle_from_vertical(
                (kp[i, l_sh, :2] + kp[i, r_sh, :2]) / 2,
                (kp[i, l_hip, :2] + kp[i, r_hip, :2]) / 2,
            )
            for i in range(n)
        ])

        hip_y_raw = (kp[:, l_hip, 1] + kp[:, r_hip, 1]) / 2
        hip_y_min = hip_y_raw.min()
        hip_y_rng = np.ptp(hip_y_raw) + 1e-9
        hip_y = (hip_y_raw - hip_y_min) / hip_y_rng

        sh_hip = np.array([
            np.linalg.norm(
                (kp[i, l_sh, :2] + kp[i, r_sh, :2]) / 2
                - (kp[i, l_hip, :2] + kp[i, r_hip, :2]) / 2
            )
            for i in range(n)
        ])
        sh_hip = sh_hip / (sh_hip.max() + 1e-9)

        sm = lambda x: uniform_filter1d(x, size=5).tolist()

        return {
            'frames': list(range(n)),
            'r_knee': sm(r_knee),
            'l_knee': sm(l_knee),
            'trunk':  sm(trunk),
            'hip_y':  sm(hip_y),
            'sh_hip': sm(sh_hip),
            'bdc':    [int(f) for f in (self.event_frames
                       if self.event_frames is not None else [])],
            'tdc':    [int(f) for f in (self.tdc_frames
                       if self.tdc_frames is not None else [])],
        }


    def run(self, save_path=None, detail_path=None):
        """Full pipeline: preprocess → detect_side → detect_cycles → compute."""
        self.preprocess().detect_side().detect_cycles().compute()
        return self


    def print_summary(self):
        if self.param_scores is None:
            self.compute()

        _ICON = {'GREEN': '✓', 'AMBER': '~', 'RED': '✗'}

        print(f'\n{"═" * 72}')
        print(f'  CYCLING BIOMECHANICAL ASSESSMENT  [{self.backend.upper()}]')
        print(f'  Cycles: {len(self.cycles)}   |   '
              f'Overall Score: {self.total_score:.1f} / 100')
        print(f'{"═" * 72}')

        current_event = None
        event_labels  = {
            'bdc':      'BOTTOM DEAD CENTRE',
            'drive':    'DRIVE',
            'posture':  'POSTURE',
            'symmetry': 'SYMMETRY',
        }

        for param in CYCLING_PARAMS:
            key   = param['key']
            event = param.get('event', '')

            if event != current_event:
                current_event = event
                print(f'\n  ── {event_labels.get(event, event.upper())} ──')

            rec   = self.emma_report.get(key, {})
            lbl   = rec.get('label', '—')
            mv    = self.mean_values.get(key, float('nan'))
            icon  = _ICON.get(lbl, '?')
            unit  = param['unit']
            tgt   = rec.get('target', '—')
            dev   = rec.get('deviation')
            dev_s = (f'  (dev {dev:+.2f})'
                     if dev is not None and np.isfinite(float(dev)) and dev != 0
                     else '')

            if isinstance(mv, float) and np.isfinite(mv):
                if unit == '%':
                    mv_s = f'{mv:.1f}%'
                elif unit in ('score', 'ratio', 'norm.'):
                    mv_s = f'{mv:.3f}'
                else:
                    mv_s = f'{mv:.1f}°'
            else:
                mv_s = 'N/A'

            print(f'  {icon} {lbl:<6}  {param["label"]:<28}  '
                  f'{mv_s:<12}  target: {tgt}{dev_s}')

        print(f'\n{"═" * 72}\n')
        return self

    def to_dict(self):
        """
        Return all scores, time-series data, and metadata as a plain dict.

        Backward-compatible with the existing cycling_dash.py contract.
        New Phase 2 keys: emma_report, cycle_records, per_param_cycle_actuals.
        """
        if self.param_scores is None:
            self.compute()

        def _safe(v):
            if v is None:
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return None if (np.isnan(f) or not np.isfinite(f)) else round(f, 4)

        def _clean_records(records):
            out = []
            for rec in records:
                out.append({k: (_safe(v) if isinstance(v, float) else v)
                            for k, v in rec.items()})
            return out

        per_cycle = {
            key: [round(v, 4) for v in vals]
            for key, vals in (self.param_results or {}).items()
        }

        return {
            'sport':         self.SPORT,
            'backend':       self.backend,
            'dominant_side': self.near_side or self.dominant_side,
            'n_cycles':      len(self.cycles),
            'fps':           self.fps,
            'n_frames':      len(self.kp_seq) if self.kp_seq is not None else 0,
            'total_score':   self.total_score,

            'param_scores': {k: round(float(v) * 100, 2)
                             for k, v in self.param_scores.items()},
            'mean_values':  {k: _safe(v)
                             for k, v in self.mean_values.items()},
            'warnings':     self.warnings,

            'video_path':           self.video_path,
            'annotated_video_path': self.annotated_video_path,

            'params_meta': [
                {
                    'key':              p['key'],
                    'label':            p['label'],
                    'unit':             p['unit'],
                    'optimal':          list(p['optimal']),
                    'description':      p.get('description', ''),
                    'source':           p.get('source', ''),
                    'event':            p.get('event', ''),
                    'event_frames_key': p.get('event_frames_key', ''),
                }
                for p in CYCLING_PARAMS
            ],

            'raw_traces':  self._build_raw_traces(),

            'per_cycle':   per_cycle,

            'bdc_frames':  [int(f) for f in (self.event_frames
                            if self.event_frames is not None else [])],

            'key_frames': {
                'bdc': [int(f) for f in (self.event_frames
                        if self.event_frames is not None else [])],
                'tdc': [int(f) for f in (self.tdc_frames
                        if self.tdc_frames is not None else [])],
            },

            'emma_report': {
                k: {ek: ev for ek, ev in rec.items() if ek != 'note'}
                for k, rec in self.emma_report.items()
            },
            'cycle_records': {
                k: _clean_records(recs)
                for k, recs in self._cycle_records.items()
            },
            'per_param_cycle_actuals': {
                k: [_safe(v) for v in vals]
                for k, vals in self.cycle_actuals.items()
            },
        }
