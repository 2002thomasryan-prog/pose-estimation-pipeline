"""Rowing pipeline orchestrator: detects stroke cycles, computes all six metrics per cycle and applies traffic-light scoring."""

import numpy as np
from scipy.ndimage import uniform_filter1d

from scorer_utils import (
    KEYPOINT_MAPS, KP_MMPOSE, _build_kp_sam3d,
    clean_keypoints, normalise_keypoints, _jxy,
    angle_at_joint, signed_trunk_angle, segment_length,
    _score_metric, _average_metric_list,
)

from .params          import ROWING_PARAMS, METRIC_RULES, CYCLE_MIN_FRAMES
from side_detection   import detect_camera_side
from .cycle_detection import detect_rowing_cycles, rower_facing_sign
from .knee_angle      import compute_knee_angle_catch
from .trunk_angle     import (compute_trunk_angle_catch, compute_trunk_angle_finish,
                              compute_trunk_angle_series)
from .spinal_flexion  import compute_spinal_flexion, compute_sh_distance_series
from .coordination    import (compute_drive_sequencing, compute_recovery_sequencing,
                              aggregate_sequencing)
from .rsf             import compute_rsf, compute_rsf_series


def _seq_target(rec, long=False):
    """Target string for a sequencing record, for whichever method produced it."""
    if rec.get('method') == 'completion':
        # Plain ASCII/WinAnsi only: the PDF uses the standard Helvetica font,
        # which has no glyph for a greater-or-equal sign.
        if long:
            return f"at least {rec['pass_lo']:.0f}% complete before next segment starts"
        return f"{rec['pass_lo']:.0f}%+ done"
    return f">{rec['floor']:.1f}% of phase" if long else f">{rec['floor']:.1f}%"


class RowingScorer:
    """
    Biomechanical scorer for rowing.

    Parameters
    ----------
    keypoints_array : np.ndarray  shape (n_frames, n_keypoints, 2+)
    backend         : str         'mmpose' | 'yolov8' | 'rtmpose' | 'sam3d'
    fps             : int
    dominant_side   : str         unused for rowing (side is auto-detected)
    log_cb          : callable    optional log callback
    frame_imgs      : np.ndarray  optional (n_frames, H, W, 3) BGR for insets
    """

    SPORT = 'rowing'

    def __init__(self, keypoints_array, backend='mmpose',
                 fps=30, dominant_side='right',
                 log_cb=None, frame_imgs=None,
                 video_path=None, annotated_video_path=None):

        self.raw_kp                = np.asarray(keypoints_array, dtype=float)
        self.backend               = backend.lower()
        self.fps                   = fps
        self.log                   = log_cb or print
        self.frame_imgs            = frame_imgs
        self.video_path            = video_path
        self.annotated_video_path  = annotated_video_path or video_path

        self.kp_map = KEYPOINT_MAPS.get(self.backend, KP_MMPOSE)
        if self.backend == 'sam3d':
            self.kp_map = _build_kp_sam3d()

        self.kp_seq         = None
        self.kp_arr         = None
        self.kp_norm        = None
        self.near_side      = None
        self.cycles         = None
        self.catch_frames   = None
        self._finish_frames = None
        self._knee_angles   = None

        self.param_results  = None
        self.mean_values    = None
        self.param_scores   = None
        self.total_score    = None

        self.cycle_actuals  = {}
        self._cycle_records = {}
        self.emma_report    = {}
        # Per-cycle sequencing records, and the clip-level aggregate built from
        # them.  Kept apart from _cycle_records because sequencing is scored
        # against the clip's own timing resolution rather than METRIC_RULES.
        self._seq_records   = {'drive': [], 'recovery': []}
        self.sequencing     = {'drive': None, 'recovery': None}

        self._rowing_ts     = None

        self.warnings       = []


    def preprocess(self):
        self.log(f'[RowingScorer] Cleaning {len(self.raw_kp)} frames ({self.backend})...')
        self.kp_seq  = clean_keypoints(self.raw_kp)
        self.kp_arr  = self.kp_seq
        self.kp_norm = normalise_keypoints(self.kp_seq, self.backend)
        self.log('[RowingScorer] Preprocessing complete.')
        return self

    def detect_side(self):
        if self.kp_seq is None:
            self.preprocess()
        self.near_side = detect_camera_side(
            self.kp_seq, self.backend, self.kp_norm)
        self.log(f'[RowingScorer] Near side detected: {self.near_side}')
        return self

    def detect_cycles(self):
        if self.near_side is None:
            self.detect_side()

        (self.cycles,
         self.catch_frames,
         self._knee_angles,
         self._finish_frames,
         cycle_warnings) = detect_rowing_cycles(self.kp_norm, self.near_side)

        self.warnings.extend(cycle_warnings)
        for w in cycle_warnings:
            self.log(f'[RowingScorer] ⚠  {w}')

        self.log(f'[RowingScorer] {len(self.cycles)} stroke cycles detected.')

        if len(self.cycles) == 0:
            raise RuntimeError(
                '[RowingScorer] No complete cycles detected. '
                'Check keypoint data quality and camera coverage.'
            )
        return self

    def compute(self):
        if self.cycles is None:
            self.detect_cycles()

        scored_keys = [p['key'] for p in ROWING_PARAMS]
        # RSF is still computed and reported, but no longer separately scored.
        all_keys    = scored_keys + ['rsf']
        seq_keys    = ('drive_sequencing', 'recovery_sequencing')
        _SEQ_SCORE  = {'GREEN': 1.0, 'AMBER': 0.5, 'RED': 0.0}

        self._seq_records = {'drive': [], 'recovery': []}

        for k in all_keys:
            self.cycle_actuals[k]  = []
            self._cycle_records[k] = []

        res = {k: [] for k in all_keys}

        for ci, (start, end) in enumerate(self.cycles):
            seg = self.kp_norm[start:end]
            n   = len(seg)
            if n < CYCLE_MIN_FRAMES:
                continue

            catch_local = 0

            global_finish = (int(self._finish_frames[ci])
                             if ci < len(self._finish_frames) else -1)
            if global_finish > 0:
                finish_local = int(np.clip(
                    global_finish - start, catch_local + 1, n - 2))
            else:
                finish_local = min(catch_local + max(1, n // 3), n - 1)

            knee_val   = compute_knee_angle_catch(seg, catch_local, self.near_side)
            rsf_val    = compute_rsf(seg, catch_local, finish_local, self.near_side)
            trunk_c    = compute_trunk_angle_catch(seg, catch_local, self.near_side)
            trunk_f    = compute_trunk_angle_finish(seg, finish_local, self.near_side)
            spinal_val = compute_spinal_flexion(seg, self.near_side)

            drv_seq = compute_drive_sequencing(
                seg, catch_local, finish_local, self.near_side)
            rec_seq = compute_recovery_sequencing(
                seg, finish_local, self.near_side)
            self._seq_records['drive'].append(drv_seq)
            self._seq_records['recovery'].append(rec_seq)

            res['knee_angle_catch'].append(knee_val)
            res['spinal_flexion'].append(spinal_val)
            if trunk_c is not None:
                res['trunk_angle_catch'].append(trunk_c)
            if trunk_f is not None:
                res['trunk_angle_finish'].append(trunk_f)
            if rsf_val is not None:
                res['rsf'].append(rsf_val)
            for key, sq in (('drive_sequencing', drv_seq),
                            ('recovery_sequencing', rec_seq)):
                if sq is None:
                    continue
                res[key].append(sq['metric'])
                self._cycle_records[key].append({
                    'score':       _SEQ_SCORE[sq['label']],
                    'label':       sq['label'],
                    'actual':      sq['metric'],
                    'target':      _seq_target(sq, long=True),
                    'uncertainty': sq['floor'],
                    'deviation':   round(sq['metric'] - sq['pass_lo'], 2),
                    'weakest':     sq['weakest'],
                })
                self.cycle_actuals[key].append(sq['metric'])

            def _acc(key, actual):
                """Score actual against METRIC_RULES[key] and store records."""
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

            _acc('knee_angle_catch',  knee_val)
            _acc('trunk_angle_catch', trunk_c)
            _acc('rsf',               rsf_val)
            _acc('trunk_angle_finish', trunk_f)
            _acc('spinal_flexion',    spinal_val)

        # Sequencing is aggregated from its own per-cycle records: the clip
        # mean is the reported figure, and the clip's timing resolution is the
        # pass band, so it cannot go through _average_metric_list's target
        # parsing.
        for phase, key in (('drive', 'drive_sequencing'),
                           ('recovery', 'recovery_sequencing')):
            self.sequencing[phase] = aggregate_sequencing(self._seq_records[phase])

        self.emma_report = {}
        for k in all_keys:
            if k in seq_keys:
                phase = 'drive' if k.startswith('drive') else 'recovery'
                agg   = self.sequencing[phase]
                if agg is None:
                    self.emma_report[k] = {
                        'score': 0.0, 'label': 'RED', 'actual': None,
                        'target': '—', 'uncertainty': None,
                        'deviation': None, 'n_cycles': 0,
                    }
                    self.warnings.append(
                        f'{phase.capitalize()} sequencing not measurable — '
                        f'no cycle had a long enough {phase} phase.')
                    continue
                self.emma_report[k] = {
                    'score':       _SEQ_SCORE[agg['label']],
                    'label':       agg['label'],
                    'actual':      agg['metric'],
                    'target':      _seq_target(agg, long=True),
                    'uncertainty': agg['floor'],
                    'deviation':   round(agg['metric'] - agg['pass_lo'], 2),
                    'n_cycles':    agg['n_strokes'],
                    'weakest':     agg['weakest'],
                    'unstable':    agg['unstable'],
                }
                if agg['unstable'] and agg.get('method') == 'completion':
                    self.warnings.append(
                        f'{phase.capitalize()} sequencing verdict differs between '
                        f'strokes in more than a third of the clip (SD '
                        f'{agg["sd"]:.1f} points) — check catch/finish detection '
                        f'before trusting the verdict.')
                elif agg['unstable']:
                    self.warnings.append(
                        f'{phase.capitalize()} sequencing varies between strokes by '
                        f'more than this clip can resolve (SD {agg["sd"]:.1f}% vs '
                        f'floor {agg["floor"]:.1f}%) — check catch/finish detection '
                        f'before trusting the verdict.')
            else:
                self.emma_report[k] = _average_metric_list(self._cycle_records[k])

        self.param_results = res

        self.param_scores = {}
        self.mean_values  = {}
        for k in scored_keys:
            rec = self.emma_report[k]
            self.param_scores[k] = rec['score']
            mv = rec['actual']
            self.mean_values[k]  = (float(mv) if mv is not None
                                    else float('nan'))

        valid = [v for v in self.param_scores.values()
                 if isinstance(v, (int, float)) and np.isfinite(float(v))]
        self.total_score = float(np.mean(valid)) * 100.0 if valid else 0.0

        self._rowing_ts = self._build_time_series()

        self.log(f'[RowingScorer] Scoring complete. '
                 f'Total: {self.total_score:.1f}/100')
        return self


    def _build_time_series(self):
        """
        Time-normalise all cycles to 101 points (0–100% of stroke).
        Builds mean ± std arrays for every traced parameter and collects
        onset percentages and RSF displacement traces.
        """
        n_points = 101
        pct      = np.linspace(0, 100, n_points)
        prefix   = 'l' if self.near_side == 'left' else 'r'

        knee_stack  = []; trunk_stack  = []; shdist_stack   = []
        drive_on    = []; recovery_on  = []
        rsf_seat    = []; rsf_handle   = []
        finish_pcts = []

        for ci, (start, end) in enumerate(self.cycles):
            seg = self.kp_norm[start:end]
            n   = len(seg)
            if n < CYCLE_MIN_FRAMES:
                continue

            t_orig  = np.linspace(0, 100, n)
            hip_xy  = _jxy(seg, f'{prefix}_hip')
            knee_xy = _jxy(seg, f'{prefix}_knee')
            ank_xy  = _jxy(seg, f'{prefix}_ankle')
            sh_xy   = _jxy(seg, f'{prefix}_shoulder')

            ka = np.array([
                angle_at_joint(hip_xy[i], knee_xy[i], ank_xy[i])
                for i in range(n)
            ])
            knee_stack.append(np.interp(pct, t_orig,
                                        uniform_filter1d(ka, size=9)))

            ta = compute_trunk_angle_series(seg, self.near_side)
            trunk_stack.append(np.interp(pct, t_orig, ta))

            sh_hip = np.array([
                segment_length(sh_xy[i], hip_xy[i]) for i in range(n)
            ])
            max_d = np.nanmax(sh_hip) + 1e-9
            shdist_stack.append(np.interp(pct, t_orig, sh_hip / max_d))

            catch_local = 0
            global_finish = (int(self._finish_frames[ci])
                             if ci < len(self._finish_frames) else -1)
            if global_finish > 0:
                finish_local = int(np.clip(
                    global_finish - start, catch_local + 1, n - 2))
            else:
                finish_local = min(catch_local + max(1, n // 3), n - 1)

            finish_pcts.append(finish_local / max(n - 1, 1) * 100)

            d_seq = compute_drive_sequencing(
                seg, catch_local, finish_local, self.near_side)
            if d_seq:
                pr = d_seq['profiles']
                drive_on.append((pr['leg']['cent'], pr['trunk']['cent'],
                                 pr['arm']['cent']))

            r_seq = compute_recovery_sequencing(seg, finish_local, self.near_side)
            if r_seq:
                pr = r_seq['profiles']
                recovery_on.append((pr['arm']['cent'], pr['trunk']['cent'],
                                    pr['leg']['cent']))

            rsf_data = compute_rsf_series(
                seg, catch_local, finish_local, self.near_side)
            if rsf_data is not None:
                seat_d, handle_d = rsf_data
                pct51     = np.linspace(0, 100, 51)
                pct_drive = np.linspace(0, 100, len(seat_d))
                rsf_seat.append(np.interp(pct51, pct_drive, seat_d))
                rsf_handle.append(np.interp(pct51, pct_drive, handle_d))

        result = {
            'pct':        pct,
            'near_side':  self.near_side,
            'finish_pct': float(np.mean(finish_pcts)) if finish_pcts else 50.0,
        }

        for name, stack in [('knee',   knee_stack),
                             ('trunk',  trunk_stack),
                             ('shdist', shdist_stack)]:
            arr = np.array(stack) if stack else np.zeros((1, n_points))
            result[f'{name}_mean'] = np.mean(arr, axis=0)
            result[f'{name}_std']  = np.std(arr,  axis=0)

        result['drive_onsets']    = drive_on
        result['recovery_onsets'] = recovery_on

        if rsf_seat:
            pct51 = np.linspace(0, 100, 51)
            result['rsf_pct']         = pct51
            result['rsf_seat_mean']   = np.mean(np.array(rsf_seat),   axis=0)
            result['rsf_seat_std']    = np.std(np.array(rsf_seat),    axis=0)
            result['rsf_handle_mean'] = np.mean(np.array(rsf_handle), axis=0)
            result['rsf_handle_std']  = np.std(np.array(rsf_handle),  axis=0)

        if self.cycles:
            mid            = len(self.cycles) // 2
            rep_start, rep_end = self.cycles[mid]
            rep_seg        = self.kp_norm[rep_start:rep_end]
            n_rep          = len(rep_seg)
            if n_rep >= CYCLE_MIN_FRAMES:
                g_catch  = (int(self.catch_frames[mid])
                            if mid < len(self.catch_frames) else rep_start)
                g_finish = (int(self._finish_frames[mid])
                            if mid < len(self._finish_frames) else -1)
                c_loc = int(np.clip(g_catch - rep_start, 0, n_rep - 1))
                f_loc = (int(np.clip(g_finish - rep_start, c_loc + 1, n_rep - 2))
                         if g_finish > 0
                         else min(c_loc + n_rep // 3, n_rep - 1))
                result['rep_catch_frame']      = rep_start + c_loc
                result['rep_finish_frame']     = rep_start + f_loc
                # cycles are catch-to-catch, so the cycle end IS the next
                # catch -- the recovery window is finish -> here, not back
                # to this cycle's own catch.
                result['rep_next_catch_frame'] = rep_end
                result['rep_kp_catch']     = rep_seg[c_loc]
                result['rep_kp_finish']    = rep_seg[f_loc]

        return result


    def _build_raw_traces(self):
        if self.kp_norm is None:
            return {}

        prefix = 'l' if self.near_side == 'left' else 'r'
        n      = len(self.kp_norm)

        hip_xy  = _jxy(self.kp_norm, f'{prefix}_hip')
        knee_xy = _jxy(self.kp_norm, f'{prefix}_knee')
        ank_xy  = _jxy(self.kp_norm, f'{prefix}_ankle')
        sh_xy   = _jxy(self.kp_norm, f'{prefix}_shoulder')
        wr_xy   = _jxy(self.kp_norm, f'{prefix}_wrist')

        knee  = np.array([angle_at_joint(hip_xy[i], knee_xy[i], ank_xy[i])
                          for i in range(n)])
        facing = rower_facing_sign(self.kp_norm, self.near_side)
        trunk  = np.array([signed_trunk_angle(hip_xy[i], sh_xy[i]) * facing
                           for i in range(n)])
        shd   = np.array([segment_length(sh_xy[i], hip_xy[i])
                          for i in range(n)])

        d_leg   = hip_xy[:, 0] - ank_xy[:, 0]
        d_trunk = sh_xy[:, 0]  - hip_xy[:, 0]
        d_arm   = wr_xy[:, 0]  - sh_xy[:, 0]

        return {
            'frames':  list(range(n)),
            'knee':    uniform_filter1d(knee,    size=9).tolist(),
            'trunk':   uniform_filter1d(trunk,   size=9).tolist(),
            'shdist':  uniform_filter1d(shd,     size=9).tolist(),
            'd_leg':   uniform_filter1d(d_leg,   size=9).tolist(),
            'd_trunk': uniform_filter1d(d_trunk, size=9).tolist(),
            'd_arm':   uniform_filter1d(d_arm,   size=9).tolist(),
        }


    def run(self, save_path=None, detail_path=None, frame_imgs=None):
        """Full pipeline: preprocess → detect_side → detect_cycles → compute."""
        self.preprocess().detect_side().detect_cycles().compute()
        return self


    def print_summary(self):
        if self.param_scores is None:
            self.compute()

        _ICON = {'GREEN': '✓', 'AMBER': '~', 'RED': '✗'}

        print(f'\n{"═" * 72}')
        print(f'  ROWING BIOMECHANICAL ASSESSMENT  [{self.backend.upper()}]  '
              f'| Side: {self.near_side}')
        print(f'  Cycles: {len(self.cycles)}   |   '
              f'Overall Score: {self.total_score:.1f} / 100')
        print(f'{"═" * 72}')

        current_event = None
        event_labels  = {
            'catch':    'CATCH',
            'drive':    'DRIVE',
            'finish':   'FINISH',
            'recovery': 'RECOVERY',
        }

        for param in ROWING_PARAMS:
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
                elif unit in ('score', 'ratio'):
                    mv_s = f'{mv:.3f}'
                else:
                    mv_s = f'{mv:.1f}°'
            else:
                mv_s = 'N/A'

            print(f'  {icon} {lbl:<6}  {param["label"]:<28}  '
                  f'{mv_s:<12}  target: {tgt}{dev_s}')

        if self.warnings:
            print(f'\n  ⚠  {len(self.warnings)} warning(s):')
            for w in self.warnings:
                print(f'     • {w}')

        print(f'\n{"═" * 72}\n')
        return self

    def _param_meta(self, p):
        """
        UI metadata for one parameter.

        Sequencing has no fixed target: its pass band is the timing resolution
        of this particular clip, so the target string and the optimal range are
        filled in here from the measured floor rather than carried in params.py.
        """
        meta = {
            'key':              p['key'],
            'label':            p['label'],
            'unit':             p['unit'],
            'optimal':          list(p['optimal']),
            'description':      p.get('description', ''),
            'source':           p.get('source', ''),
            'event':            p.get('event', ''),
            'event_frames_key': p.get('event_frames_key', ''),
        }
        if p['key'] in ('drive_sequencing', 'recovery_sequencing'):
            phase = 'drive' if p['key'].startswith('drive') else 'recovery'
            agg   = self.sequencing.get(phase)
            if agg:
                meta['optimal']    = [agg['pass_lo'], 100.0]
                # Short form: the Target column is narrow.
                meta['target_str'] = _seq_target(agg)
            else:
                meta['target_str'] = '—'
        return meta


    def to_dict(self):
        """
        Return all scores, time-series data, and metadata as a plain dict.

        Backward-compatible with the existing rowing_dash.py contract.
        New Phase 2 keys: emma_report, cycle_records, per_param_cycle_actuals.
        """
        if self.param_scores is None:
            self.compute()

        def _arr(v):
            return v.tolist() if isinstance(v, np.ndarray) else list(v)

        def _ts_serialise(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    out[k] = v.tolist()
                elif isinstance(v, (list, tuple)):
                    out[k] = [
                        x.tolist() if isinstance(x, np.ndarray) else x
                        for x in v
                    ]
                elif isinstance(v, dict):
                    out[k] = _ts_serialise(v)
                else:
                    out[k] = v
            return out

        def _safe(v):
            if v is None:
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return None if (np.isnan(f) or not np.isfinite(f)) else round(f, 4)

        ts    = self._rowing_ts or {}
        d_ons = ts.get('drive_onsets',    [])
        r_ons = ts.get('recovery_onsets', [])

        coord_frames = {'drive': [], 'recovery': []}
        for ci, (start, end) in enumerate(self.cycles):
            clen = max(end - start, 1)
            if ci < len(d_ons):
                d = d_ons[ci]
                coord_frames['drive'].append({
                    'cycle': ci,
                    'catch': start,
                    'leg':   start + int(d[0] / 100.0 * clen),
                    'trunk': start + int(d[1] / 100.0 * clen),
                    'arm':   start + int(d[2] / 100.0 * clen),
                })
            if ci < len(r_ons):
                r = r_ons[ci]
                gf  = (int(self._finish_frames[ci])
                       if ci < len(self._finish_frames) else -1)
                fin = gf if gf > 0 else start + clen // 2
                rlen = max(end - fin, 1)
                coord_frames['recovery'].append({
                    'cycle':  ci,
                    'finish': fin,
                    'arm':    fin + int(r[0] / 100.0 * rlen),
                    'trunk':  fin + int(r[1] / 100.0 * rlen),
                    'leg':    fin + int(r[2] / 100.0 * rlen),
                })

        mid_drive_frames = []
        mid_recovery_frames = []
        for ci, (start, end) in enumerate(self.cycles):
            gf = (int(self._finish_frames[ci])
                  if ci < len(self._finish_frames) else (start + end) // 2)
            mid_drive_frames.append((start + gf) // 2)
            mid_recovery_frames.append((gf + end) // 2)

        key_frames = {
            'catch':        [int(f) for f in self.catch_frames],
            'finish':       [int(f) for f in self._finish_frames],
            'mid_drive':    mid_drive_frames,
            'mid_recovery': mid_recovery_frames,
            'rep_catch':    int(ts.get('rep_catch_frame',  self.catch_frames[0]
                                       if len(self.catch_frames) else 0)),
            'rep_finish':   int(ts.get('rep_finish_frame', self._finish_frames[0]
                                       if len(self._finish_frames) else 0)),
            'rep_next_catch': int(ts.get('rep_next_catch_frame',
                                         self.catch_frames[-1]
                                         if len(self.catch_frames) else 0)),
        }

        def _clean_records(records):
            out = []
            for rec in records:
                out.append({k: (_safe(v) if isinstance(v, float) else v)
                            for k, v in rec.items()})
            return out

        return {
            'sport':       self.SPORT,
            'backend':     self.backend,
            'near_side':   self.near_side,
            'n_cycles':    len(self.cycles),
            'fps':         self.fps,
            'n_frames':    len(self.kp_norm) if self.kp_norm else 0,
            'total_score': self.total_score,

            'param_scores': {k: round(float(v) * 100, 2)
                             for k, v in self.param_scores.items()},
            'mean_values':  {k: _safe(v)
                             for k, v in self.mean_values.items()},
            'warnings':     self.warnings,

            'video_path':           self.video_path,
            'annotated_video_path': self.annotated_video_path,

            'params_meta': [
                self._param_meta(p) for p in ROWING_PARAMS
            ],

            'raw_traces':  self._build_raw_traces(),

            'time_series': _ts_serialise(ts),

            'key_frames':   key_frames,
            'coord_frames': coord_frames,

            # Clip-level sequencing: per-segment contribution profiles
            # (q25 / centroid / q75 / share), both handover gaps, the clip's
            # timing resolution and the stroke-to-stroke spread.  This is what
            # the report timeline strip is drawn from.
            'sequencing': {
                phase: (dict(agg) if agg else None)
                for phase, agg in self.sequencing.items()
            },

            # Kept for reference: computed, no longer separately scored.
            'rsf': _safe(self.emma_report.get('rsf', {}).get('actual')),

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
