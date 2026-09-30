"""Builds the landscape PDF biomechanical assessment report for a single session."""

from __future__ import annotations

import io
import math
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import PIL.Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


PAGE_W, PAGE_H = landscape(A4)

MARGIN    = 1.8 * cm
CONTENT_W = PAGE_W - 2 * MARGIN
CONTENT_H = PAGE_H - 2 * MARGIN

C = {
    'bg':        colors.HexColor('#F7F8FA'),
    'header':    colors.HexColor('#1E2761'),
    'header_lt': colors.HexColor('#CADCFC'),
    'card_bg':   colors.white,
    'border':    colors.HexColor('#D0D5DD'),
    'txt':       colors.HexColor('#1A1A2E'),
    'txt_light': colors.HexColor('#6B7280'),
    'green':     colors.HexColor('#16A34A'),
    'green_lt':  colors.HexColor('#DCFCE7'),
    'amber':     colors.HexColor('#D97706'),
    'amber_lt':  colors.HexColor('#FEF3C7'),
    'red':       colors.HexColor('#DC2626'),
    'red_lt':    colors.HexColor('#FEE2E2'),
    'white':     colors.white,
}


def _tl_bucket(score_0_100: Optional[float]) -> str:
    if score_0_100 is None:
        return 'AMBER'
    if score_0_100 >= 75:
        return 'GREEN'
    if score_0_100 >= 25:
        return 'AMBER'
    return 'RED'


def _tl_colors(bucket: str):
    """Returns (text_color, bg_color) for a traffic-light bucket."""
    return {
        'GREEN': (C['green'], C['green_lt']),
        'AMBER': (C['amber'], C['amber_lt']),
        'RED':   (C['red'],   C['red_lt']),
    }.get(bucket, (C['txt_light'], C['bg']))


import re
_BRACKET_SUFFIX_RE = re.compile(r'\s*\([^)]*\)\s*$')


def _strip_bracket_suffix(label: str) -> str:
    """Remove a trailing parenthetical suffix, e.g. 'Torso Tilt (TD)' -> 'Torso Tilt'."""
    if not label:
        return label
    return _BRACKET_SUFFIX_RE.sub('', label).strip()


def _fmt_frac(value: float) -> str:
    """Format a fractional score (multiples of 0.5) without a trailing '.0'."""
    rounded = round(value * 2) / 2.0
    if rounded == int(rounded):
        return str(int(rounded))
    return f'{rounded:.1f}'


_DIMENSIONLESS_UNITS = {'score', 'ratio', 'norm.', 'index', ''}


def _fmt_bound(v: float) -> str:
    """Format a single target bound at the precision the number needs."""
    v = float(v)
    if v == int(v):
        return str(int(v))
    return f'{v:.3f}'.rstrip('0').rstrip('.')


# Decimal places for angles and percentages, per sport.  Rowing reports round
# to the nearest degree (agreed 18 Sept 2026): the angles carry ~5 deg of
# error, so a decimal place is false precision.
VALUE_DP_BY_SPORT = {'rowing': 0}


def _fmt_value(value, unit: str, dp: int = 1) -> str:
    """Render a measured value with the unit declared in params_meta."""
    if value is None:
        return '—'
    try:
        v = float(value)
    except (TypeError, ValueError):
        return '—'
    if not np.isfinite(v):
        return '—'
    if unit == '°':
        return f'{v:.{dp}f}°'
    if unit == '%':
        return f'{v:.{dp}f}%'
    if unit in _DIMENSIONLESS_UNITS:
        return f'{v:.2f}'
    return f'{v:.2f} {unit}'.strip()


def _fmt_target(pm: dict) -> str:
    """
    Build the Target-column string from a params_meta entry.

    Unit-aware and precision-preserving: decimal bounds survive, and the
    unit symbol (if any) is appended once.
    """
    if pm.get('target_str'):
        # Metrics whose pass band is measured per clip rather than fixed
        # (sequencing) carry their target as a ready-made string.
        return pm['target_str']

    opt  = pm.get('optimal')
    unit = pm.get('unit', '°')
    if not opt or len(opt) != 2:
        return '—'
    try:
        lo, hi = float(opt[0]), float(opt[1])
    except (TypeError, ValueError):
        return '—'

    suffix = unit if unit in ('°', '%') else ''

    if unit == '°':
        if lo <= -90 and hi == 0:
            return '<0°'
        if lo == 0 and hi >= 100:
            return '>0°'
        if hi < 0 and lo <= -90:
            return f'<{_fmt_bound(hi)}°'
        if lo < 0 and hi < 0:
            # genuine two-sided negative range (e.g. rowing layback at the
            # finish) -- an en-dash between two negatives is unreadable
            return f'{_fmt_bound(lo)}° to {_fmt_bound(hi)}°'
        if lo == 0 and hi <= 20:
            return f'<{_fmt_bound(hi)}°'
        return f'{_fmt_bound(lo)}–{_fmt_bound(hi)}°'

    if lo == 0:
        return f'<{_fmt_bound(hi)}{suffix}'
    if hi == 1.0 and unit in _DIMENSIONLESS_UNITS:
        return f'>{_fmt_bound(lo)}'
    return f'{_fmt_bound(lo)}–{_fmt_bound(hi)}{suffix}'


BADGE_LABELS = {'GREEN': 'PASS', 'AMBER': 'AMBER', 'RED': 'FAIL'}


def _metric_points(param_scores: dict, keys: list) -> tuple:
    """
    Score a set of metrics as points out of the number of metrics.

    GREEN = 1 point, AMBER = 0.5, RED = 0.  Returns (points, n_metrics);
    points is None when no metric in the set produced a score.  This is the
    single scheme used for the overall figure and every event sub-score in
    all three sports.
    """
    scores = [param_scores.get(k) for k in keys if param_scores.get(k) is not None]
    n = len(keys)
    if not scores:
        return None, n
    return sum(s / 100.0 for s in scores), n


def _wrap_text_to_width(c: canvas.Canvas, text: str, font: str,
                        size: float, max_width: float) -> list:
    """Greedy word-wrap `text` to fit within max_width at the given font/size."""
    words = text.split()
    lines = []
    cur = ''
    for w in words:
        candidate = f'{cur} {w}'.strip()
        if c.stringWidth(candidate, font, size) <= max_width or not cur:
            cur = candidate
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


METRIC_LAYMAN_TIPS = {
    'trailing_knee_td':  'Back knee should be behind your hip',
    'landing_foot_td':   'Foot should land reasonably flat, not badly heel- or toe-first.',
    'shin_ext_td':        'Shin should be over your ankle at touchdown to avoid braking.',
    'torso_tilt_td':      'Upper body should stay upright, not hunched forward or leaning backwards.',
    'thigh_gap_ms':       'Far leg should be driving forward, and ahead of the stance leg.',
    'sinking_ms':         'Knee should stay over your foot, not collapsing downwards.',
    'trailing_foot_to':   'Foot should be close to vertical at toe-off.',
    'thigh_sep_to':       'Legs should be well split apart as you push off the ground and drive the front leg forwards.',
    'leading_ankle_to':   'Front ankle should be flexed and driving forward.',
    'thigh_sep_mvp':      'Legs should be split wide apart at the peak of your stride.',
    'trailing_shin_mvp':  'Back shin should be angled down, not kicking up behind you.',
    'leading_knee_mvp':   'Front knee should be driving up and forward.',
}


METRIC_LAYMAN_TIPS_ROWING = {
    'knee_angle_catch':      'Shins should come up to vertical at the catch — compressed, but not jammed into the stretcher.',
    'trunk_angle_catch':     'Body should be leaning forward from the hips at the catch, with a flat back rather than a rounded one.',
    'drive_sequencing':      'Power should arrive in order: legs, then back, then arms, with clear daylight between each.',
    'trunk_angle_finish':    'Body should lay back slightly at the finish, not stay upright and not collapse backwards.',
    'spinal_flexion':        'Back should hold its length through the stroke rather than rounding under load.',
    'recovery_sequencing':   'Coming forward the order reverses: hands away, then body over, then the slide.',
}

LAYMAN_TIPS_BY_SPORT = {
    'running': METRIC_LAYMAN_TIPS,
    'rowing':  METRIC_LAYMAN_TIPS_ROWING,
}

# Which metric's overlay is drawn on each frame of a rowing progression strip.
# The phase pages show one metric developing across the phase rather than one
# frame per metric, so the metric is chosen per phase, not per image.
ROWING_PROGRESSION_DRAW_METRIC = {
    'drive':    'drive_sequencing',
    'recovery': 'recovery_sequencing',
}

# Segment colours for the sequencing timeline: one hue, stepped light to dark
# in the order the segments are supposed to fire.  Every bar is labelled, so
# identity never rests on colour alone, and the ramp survives greyscale print.
SEQ_SEGMENT_COLORS = {
    'leg':   colors.HexColor('#1E2761'),
    'trunk': colors.HexColor('#4C5BA6'),
    'arm':   colors.HexColor('#8C98D4'),
}
SEQ_SEGMENT_LABELS = {'leg': 'Knee', 'trunk': 'Back', 'arm': 'Arms'}
# Joint angle each segment is judged on in the angle-based sequencing method.
SEQ_ANGLE_LABELS   = {'leg': 'Knee', 'trunk': 'Trunk', 'arm': 'Elbow'}

# Measurement error quoted on the rowing report (approximate, all angles).
ROWING_ANGLE_ERROR_DEG = 5


def _line_intersection(p1, p2, p3, p4):
    """
    Return the intersection point of the *infinite* lines through (p1,p2)
    and (p3,p4), or None if the lines are parallel/degenerate.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py])


def _draw_dashed_line(img: np.ndarray, pt_a, pt_b,
                      color=(255, 200, 0), dash_len=8, gap_len=5,
                      thickness=None):
    """Draw a dashed line from pt_a to pt_b (used for line extensions)."""
    if pt_a is None or pt_b is None:
        return
    if not (np.all(np.isfinite(pt_a)) and np.all(np.isfinite(pt_b))):
        return
    h = img.shape[0]
    scale = h / 400.0
    t = thickness or max(1, int(1.1 * scale))
    dash_len *= scale
    gap_len  *= scale

    length = float(np.linalg.norm(pt_b - pt_a))
    if length < 1.0:
        return
    direction = (pt_b - pt_a) / length

    pos, drawing = 0.0, True
    while pos < length:
        seg_len = dash_len if drawing else gap_len
        end = min(pos + seg_len, length)
        if drawing:
            p1 = pt_a + direction * pos
            p2 = pt_a + direction * end
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                     color, t, cv2.LINE_AA)
        pos += seg_len
        drawing = not drawing


def _draw_converging_angle(img: np.ndarray, a_hip, a_knee, b_hip, b_knee,
                           color=(255, 200, 0)):
    """
    Draw the true angle between two thighs that each have their own hip.

    The real hip→knee segments are drawn elsewhere (via 'limbs'). This adds:
      1. A dashed extension of whichever segment(s) don't already reach the
         point where the two thigh *lines* actually cross.
      2. The angle arc at that crossing point.

    If the lines are near-parallel or cross far outside the frame, the
    extension/arc is skipped and only the two real segments remain visible.
    """
    pts = (a_hip, a_knee, b_hip, b_knee)
    if any(p is None for p in pts):
        return
    if not all(np.all(np.isfinite(p)) for p in pts):
        return

    inter = _line_intersection(a_hip, a_knee, b_hip, b_knee)
    if inter is None:
        return

    h = img.shape[0]
    max_dist = 3.0 * h
    if (np.linalg.norm(inter - a_hip) > max_dist or
            np.linalg.norm(inter - b_hip) > max_dist):
        return

    for hip, knee in ((a_hip, a_knee), (b_hip, b_knee)):
        d = knee - hip
        denom = float(np.dot(d, d))
        if denom < 1e-6:
            continue
        t = float(np.dot(inter - hip, d)) / denom
        if t < 0:
            _draw_dashed_line(img, hip, inter, color=color)
        elif t > 1:
            _draw_dashed_line(img, knee, inter, color=color)

    _draw_angle_arc(img, a_knee, inter, b_knee, color=color)


SPORT_CONFIGS = {
    'running': {
        'frames_source': 'raw_traces',
        'events':      ['td', 'ms', 'to', 'mvp'],
        'labels':      {
            'td':  'Touchdown',
            'ms':  'Mid-Stance',
            'to':  'Toe-Off',
            'mvp': 'Max Vertical Projection',
        },
        'frames_key':  {
            'td':  'td_frames',
            'ms':  'ms_frames',
            'to':  'to_frames',
            'mvp': 'mvp_frames',
        },
    },
    'cycling': {
        'frames_source': 'key_frames',
        'events':      ['tdc', 'bdc', 'drive', 'recovery', 'symmetry'],
        'labels':      {
            'tdc':      'Top Dead Centre',
            'bdc':      'Bottom Dead Centre',
            'drive':    'Drive Phase',
            'recovery': 'Recovery Phase',
            'symmetry': 'Bilateral Symmetry',
        },
        'frames_key':  {
            'tdc':      'tdc',
            'bdc':      'bdc',
            'drive':    'bdc',
            'recovery': 'tdc',
            'symmetry': 'bdc',
        },
    },
    'rowing': {
        'frames_source': 'key_frames',
        'events':      ['catch', 'drive', 'finish', 'recovery'],
        'labels':      {
            'catch':    'Catch',
            'drive':    'Drive',
            'finish':   'Finish',
            'recovery': 'Recovery',
        },
        'frames_key':  {
            'catch':    'catch',
            'drive':    'mid_drive',
            'finish':   'finish',
            'recovery': 'mid_recovery',
        },
    },
}

EVENTS          = SPORT_CONFIGS['running']['events']
EVENT_LABELS    = SPORT_CONFIGS['running']['labels']
EVENT_FRAMES_KEY = SPORT_CONFIGS['running']['frames_key']

METRIC_EVENT = {
    'thigh_sep_mvp':     'mvp',
    'trailing_shin_mvp': 'mvp',
    'leading_knee_mvp':  'mvp',
    'trailing_foot_to':  'to',
    'thigh_sep_to':      'to',
    'leading_ankle_to':  'to',
    'trailing_knee_td':  'td',
    'landing_foot_td':   'td',
    'shin_ext_td':       'td',
    'torso_tilt_td':     'td',
    'thigh_gap_ms':      'ms',
    'sinking_ms':        'ms',
}


METRIC_DRAW = {
    'thigh_sep_mvp': {
        'limbs':        [('ns_hip', 'ns_knee'), ('fs_hip', 'fs_knee')],
        'converge_arc': ('ns_hip', 'ns_knee', 'fs_hip', 'fs_knee'),
        'arc':          None,
        'vertical':     None,
    },
    'trailing_shin_mvp': {
        'limbs':      [],
        'arc':        None,
        'vertical':   None,
        'horizontal': ['ns_knee', 'ns_ankle'],
        'dots':       [('ns_knee', 'Knee'), ('ns_ankle', 'Ankle')],
        'dot_radius': 5,
    },
    'leading_knee_mvp': {
        'limbs':    [('fs_ankle', 'fs_knee'), ('fs_knee', 'fs_hip')],
        'arc':      ('fs_ankle', 'fs_knee', 'fs_hip'),
        'vertical': None,
    },

    'trailing_foot_to': {
        'limbs':    [('ns_toe', 'ns_heel')],
        'arc':      None,
        'vertical': 'ns_heel',
    },
    'thigh_sep_to': {
        'limbs':        [('ns_hip', 'ns_knee'), ('fs_hip', 'fs_knee')],
        'converge_arc': ('ns_hip', 'ns_knee', 'fs_hip', 'fs_knee'),
        'arc':          None,
        'vertical':     None,
    },
    'leading_ankle_to': {
        'limbs':    [('fs_toe', 'fs_ankle'), ('fs_ankle', 'fs_knee')],
        'arc':      ('fs_toe', 'fs_ankle', 'fs_knee'),
        'vertical': None,
    },

    'trailing_knee_td': {
        'limbs':     [],
        'arc':       None,
        'vertical':  ['fs_hip', 'fs_knee'],
        'dots':      [('fs_hip', 'Hip'), ('fs_knee', 'Knee')],
        'dot_radius': 5,
    },
    'landing_foot_td': {
        'limbs':    [('ns_toe', 'ns_heel')],
        'arc':      None,
        'vertical': None,
    },
    'shin_ext_td': {
        'limbs':     [],
        'arc':       None,
        'vertical':  ['ns_knee', 'ns_ankle'],
        'dots':      [('ns_knee', 'Knee'), ('ns_ankle', 'Ankle')],
        'dot_radius': 5,
    },
    'torso_tilt_td': {
        'limbs':    [('ns_hip', 'ns_shoulder')],
        'arc':      None,
        'vertical': 'ns_hip',
    },

    'thigh_gap_ms': {
        'limbs':        [('ns_hip', 'ns_knee'), ('fs_hip', 'fs_knee')],
        'converge_arc': ('ns_hip', 'ns_knee', 'fs_hip', 'fs_knee'),
        'arc':          None,
        'vertical':     None,
    },
    'sinking_ms': {
        'limbs':     [],
        'arc':       None,
        'vertical':  ['ns_toe', 'ns_knee'],
        'dots':      [('ns_toe', 'Toe'), ('ns_knee', 'Knee')],
        'dot_radius': 5,
    },
}

METRIC_DRAW_CYCLING = {
    'knee_angle_tdc': {
        'limbs':    [('ns_hip', 'ns_knee'), ('ns_knee', 'ns_ankle')],
        'arc':      ('ns_hip', 'ns_knee', 'ns_ankle'),
        'vertical': None,
    },
    'knee_angle_bdc': {
        'limbs':    [('ns_hip', 'ns_knee'), ('ns_knee', 'ns_ankle')],
        'arc':      ('ns_hip', 'ns_knee', 'ns_ankle'),
        'vertical': None,
    },
    'trunk_angle': {
        'limbs':    [('ns_hip', 'ns_shoulder')],
        'arc':      None,
        'vertical': 'ns_hip',
    },
    'spinal_flexion': {
        'limbs':    [('ns_hip', 'ns_shoulder')],
        'arc':      None,
        'vertical': None,
    },
    'pelvic_stability': {
        'limbs':    [('l_hip', 'r_hip')],
        'arc':      None,
        'vertical': None,
    },
    'ankle_rom': {
        'limbs':    [('ns_knee', 'ns_ankle')],
        'arc':      None,
        'vertical': None,
    },
    'bilateral_symmetry': {
        'limbs':    [('l_hip', 'l_knee'), ('l_knee', 'l_ankle'),
                     ('r_hip', 'r_knee'), ('r_knee', 'r_ankle')],
        'arc':      None,
        'vertical': None,
    },
}

METRIC_DRAW_ROWING = {
    'knee_angle_catch': {
        'limbs':    [('ns_hip', 'ns_knee'), ('ns_knee', 'ns_ankle')],
        'arc':      ('ns_hip', 'ns_knee', 'ns_ankle'),
        'vertical': None,
    },
    'trunk_angle_catch': {
        'limbs':    [('ns_hip', 'ns_shoulder')],
        'arc':      None,
        'vertical': 'ns_hip',
    },
    'trunk_angle_finish': {
        'limbs':    [('ns_hip', 'ns_shoulder')],
        'arc':      None,
        'vertical': 'ns_hip',
    },
    'spinal_flexion': {
        'limbs':    [('ns_hip', 'ns_shoulder')],
        'arc':      None,
        'vertical': None,
    },
    'rsf': {
        'limbs':    [('ns_hip', 'ns_wrist')],
        'arc':      None,
        'vertical': None,
        'dots':     [('ns_hip', 'Seat'), ('ns_wrist', 'Handle')],
    },
    'drive_sequencing': {
        'limbs':    [('ns_ankle', 'ns_knee'), ('ns_knee', 'ns_hip'),
                     ('ns_hip', 'ns_shoulder'), ('ns_shoulder', 'ns_wrist')],
        'arc':      None,
        'vertical': None,
    },
    'recovery_sequencing': {
        'limbs':    [('ns_ankle', 'ns_knee'), ('ns_knee', 'ns_hip'), ('ns_hip', 'ns_shoulder')],
        'arc':      None,
        'vertical': None,
    },
}

METRIC_DRAW_BY_SPORT = {
    'running': METRIC_DRAW,
    'cycling': METRIC_DRAW_CYCLING,
    'rowing':  METRIC_DRAW_ROWING,
}

def _kp_dict_at_frame(keypoints: np.ndarray, frame_idx: int,
                      near_side: str, backend: str = 'rtmpose') -> dict:
    """
    Return a dict of canonical joint name → (x, y) for a single frame,
    resolved into near-side (ns_) / far-side (fs_) keys for drawing.

    Uses scorer_utils.normalise_keypoints() so the correct index map is
    applied for every backend (MMPose, YOLOv8, RTMPose, SAM-3D/MHR70).
    Falls back to a minimal hardcoded lookup if scorer_utils is unavailable.
    """
    if keypoints is None or frame_idx < 0 or frame_idx >= len(keypoints):
        return {}

    try:
        from scorer_utils import normalise_keypoints
        single = keypoints[frame_idx:frame_idx + 1]
        frame_dicts = normalise_keypoints(single, backend)
        raw = frame_dicts[0]

        def _clean(v):
            if v is None:
                return None
            arr = np.asarray(v, float)
            return None if not np.all(np.isfinite(arr)) else arr

        raw = {k: _clean(v) for k, v in raw.items()}

    except Exception:
        IDX = {
            'nose': 0,
            'l_shoulder': 5,  'r_shoulder': 6,
            'l_hip': 11,      'r_hip': 12,
            'l_knee': 13,     'r_knee': 14,
            'l_ankle': 15,    'r_ankle': 16,
            'l_toe': 20,      'r_toe': 21,
            'l_heel': 24,     'r_heel': 25,
        }
        raw = {}
        for name, idx in IDX.items():
            if idx < keypoints.shape[1]:
                pt = keypoints[frame_idx, idx, :2]
                raw[name] = pt.copy() if np.all(np.isfinite(pt)) else None
            else:
                raw[name] = None

    ns = 'l' if near_side == 'left' else 'r'
    fs = 'r' if ns == 'l' else 'l'

    resolved = dict(raw)
    for suffix in ('hip', 'knee', 'ankle', 'toe', 'heel', 'shoulder', 'wrist', 'elbow'):
        resolved[f'ns_{suffix}'] = raw.get(f'{ns}_{suffix}')
        resolved[f'fs_{suffix}'] = raw.get(f'{fs}_{suffix}')

    return resolved


def _median_frame(frames: list) -> int:
    """Return the median frame index from a list of frame indices."""
    if not frames:
        return 0
    frames_sorted = sorted(frames)
    return frames_sorted[len(frames_sorted) // 2]


def _extract_frame(video_path: str, frame_idx: int,
                   target_h: int = 400) -> Optional[np.ndarray]:
    """
    Extract a single BGR frame, resized to target_h.

    Strategy:
      1. Try direct seek (fast, works for most formats)
      2. If seek fails, read sequentially up to frame_idx (robust for .mov/H.264)
    """
    if not video_path or not os.path.isfile(video_path):
        return None

    def _resize(f):
        h, w = f.shape[:2]
        scale = target_h / h
        return cv2.resize(f, (int(w * scale), target_h),
                          interpolation=cv2.INTER_AREA)

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return _resize(frame)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    last_frame = None
    limit = min(int(frame_idx) + 1, 2000)
    for i in range(limit):
        ok, f = cap.read()
        if not ok or f is None:
            break
        last_frame = f
    cap.release()
    if last_frame is not None:
        return _resize(last_frame)
    return None


def _extract_frame_raw(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Extract a single BGR frame at NATIVE resolution (no resize).
    Used by the report when smart-cropping must happen in the original
    pixel space so that keypoint coordinates map correctly.
    Falls back to sequential read for .mov / H.264.
    """
    if not video_path or not os.path.isfile(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return frame

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    last_frame = None
    limit = min(int(frame_idx) + 1, 2000)
    for i in range(limit):
        ok, f = cap.read()
        if not ok or f is None:
            break
        last_frame = f
    cap.release()
    return last_frame


def _fit_to_box(bgr: np.ndarray, box_w_pt: float,
                box_h_pt: float) -> tuple:
    """
    Scale bgr so the full image fits within (box_w_pt x box_h_pt) with no
    cropping.  Returns (resized_bgr, draw_w, draw_h, offset_x, offset_y)
    where offset_x/y are centring offsets in PDF points so the caller can
    centre the image inside the box.
    """
    h, w   = bgr.shape[:2]
    scale  = min(box_w_pt / w, box_h_pt / h)
    draw_w = int(w * scale)
    draw_h = int(h * scale)
    resized  = cv2.resize(bgr, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
    offset_x = (box_w_pt  - draw_w) / 2.0
    offset_y = (box_h_pt  - draw_h) / 2.0
    return resized, draw_w, draw_h, offset_x, offset_y


def _smart_crop_to_person(bgr: np.ndarray, kp_dict: dict,
                          pad_frac: float = 0.18) -> np.ndarray:
    """
    Crop bgr to a bounding box around valid keypoints, with padding.
    Filters out zero/near-zero coordinates (uninitialised joints) to avoid
    black-bar artifacts from joints outside the frame.
    Falls back to the full image if no valid keypoints found.
    """
    h, w = bgr.shape[:2]
    xs, ys = [], []
    abs_min = 8

    for v in kp_dict.values():
        if v is None:
            continue
        arr = np.asarray(v, float)
        if arr.shape[0] < 2 or not np.all(np.isfinite(arr)):
            continue
        x, y = float(arr[0]), float(arr[1])
        if x < abs_min or y < abs_min:
            continue
        xs.append(x)
        ys.append(y)

    if not xs:
        return bgr

    pad_x = pad_frac * (max(xs) - min(xs) + 1)
    pad_y = pad_frac * (max(ys) - min(ys) + 1)

    x1 = max(0,     int(min(xs) - pad_x))
    x2 = min(w - 1, int(max(xs) + pad_x))
    y1 = max(0,     int(min(ys) - pad_y))
    y2 = min(h - 1, int(max(ys) + pad_y))

    if x2 <= x1 or y2 <= y1:
        return bgr

    return bgr[y1:y2, x1:x2]


def _frame_to_image_reader(bgr: np.ndarray) -> ImageReader:
    """Convert a BGR numpy frame to a ReportLab ImageReader via PIL."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = PIL.Image.fromarray(rgb)
    return ImageReader(pil_img)


def _compute_video_crop_box(keypoints: np.ndarray,
                             pad_x_frac: float = 0.30,
                             pad_y_frac: float = 0.12,
                             frame_w: int = 0,
                             frame_h: int = 0) -> Optional[tuple]:
    """
    Compute a single fixed crop box (x1, y1, x2, y2) from all keypoints
    across the entire video, so every report frame is cropped consistently.

    Parameters
    ----------
    keypoints   : np.ndarray  shape (n_frames, n_kp, 2+)
    pad_x_frac  : float  padding as fraction of keypoint x-range (each side)
    pad_y_frac  : float  padding as fraction of keypoint y-range (each side)
    frame_w/h   : int    native video dimensions for clamping

    Returns
    -------
    (x1, y1, x2, y2) in pixels, or None if keypoints are invalid.
    """
    if keypoints is None or len(keypoints) == 0:
        return None
    abs_min = 8
    all_x, all_y = [], []
    for frame_kps in keypoints:
        for kp in frame_kps:
            x, y = float(kp[0]), float(kp[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if x < abs_min or y < abs_min:
                continue
            all_x.append(x)
            all_y.append(y)

    if not all_x:
        return None

    kp_w = max(all_x) - min(all_x) + 1
    kp_h = max(all_y) - min(all_y) + 1
    pad_x = pad_x_frac * kp_w
    pad_y = pad_y_frac * kp_h

    x1 = max(0,        int(min(all_x) - pad_x))
    x2 = min(frame_w - 1 if frame_w else 99999, int(max(all_x) + pad_x))
    y1 = max(0,        int(min(all_y) - pad_y))
    y2 = min(frame_h - 1 if frame_h else 99999, int(max(all_y) + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _apply_crop_box(bgr: np.ndarray, crop_box: Optional[tuple]) -> np.ndarray:
    """Apply a pre-computed (x1,y1,x2,y2) crop box to a frame."""
    if crop_box is None or bgr is None:
        return bgr
    x1, y1, x2, y2 = crop_box
    h, w = bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return bgr
    return bgr[y1:y2, x1:x2]


    """Convert a BGR numpy frame to a ReportLab ImageReader via PIL."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = PIL.Image.fromarray(rgb)
    return ImageReader(pil_img)


def _draw_angle_arc(img: np.ndarray, pt_a, vertex, pt_c,
                    color=(255, 200, 0), radius_frac=0.18):
    """
    Draw an angle arc at `vertex` between rays vertex→pt_a and vertex→pt_c.
    radius_frac: arc radius as fraction of image height.

    If the arc's radius extends beyond the real limb line (i.e. pt_a or
    pt_c is closer to the vertex than the arc radius), a dotted line bridges
    the gap from the line's real end out to the arc's edge, so the arc
    always visually connects to the limb rather than floating past it.
    """
    if pt_a is None or vertex is None or pt_c is None:
        return
    if not (np.all(np.isfinite(pt_a)) and np.all(np.isfinite(vertex))
            and np.all(np.isfinite(pt_c))):
        return

    h, w = img.shape[:2]
    scale = h / 400.0
    r = int(radius_frac * h)

    vertex_arr = np.asarray(vertex, dtype=float)
    for far_pt in (pt_a, pt_c):
        far_arr = np.asarray(far_pt, dtype=float)
        seg_vec = far_arr - vertex_arr
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len < 1e-6:
            continue
        if seg_len < r:
            arc_edge_pt = vertex_arr + (seg_vec / seg_len) * r
            _draw_dashed_line(img, far_arr, arc_edge_pt, color=color)

    vx, vy = int(vertex[0]), int(vertex[1])
    angle_a = math.degrees(math.atan2(-(pt_a[1] - vertex[1]),
                                       pt_a[0] - vertex[0]))
    angle_c = math.degrees(math.atan2(-(pt_c[1] - vertex[1]),
                                       pt_c[0] - vertex[0]))

    a1 = angle_a % 360
    a2 = angle_c % 360
    diff = abs(a1 - a2)
    if diff > 180:
        diff = 360 - diff
        a1, a2 = a2, a1

    start_a = min(a1, a2)
    end_a   = max(a1, a2)
    if end_a - start_a > 180:
        start_a, end_a = end_a, start_a + 360

    cv2.ellipse(img, (vx, vy), (r, r), 0,
                -end_a, -start_a, color, max(1, int(1.4 * scale)))


def _draw_line(img: np.ndarray, pt_a, pt_b,
               color=(255, 200, 0), thickness=None):
    if pt_a is None or pt_b is None:
        return
    if not (np.all(np.isfinite(pt_a)) and np.all(np.isfinite(pt_b))):
        return
    h = img.shape[0]
    t = thickness or max(1, int(h / 180))
    cv2.line(img, (int(pt_a[0]), int(pt_a[1])),
             (int(pt_b[0]), int(pt_b[1])), color, t, cv2.LINE_AA)
    r = max(3, int(t * 1.3))
    cv2.circle(img, (int(pt_a[0]), int(pt_a[1])), r, color, -1)
    cv2.circle(img, (int(pt_b[0]), int(pt_b[1])), r, color, -1)


def _scale_kp_to_frame(kp_dict: dict, orig_h: int, target_h: int) -> dict:
    """Scale keypoint coordinates from original frame height to resized height."""
    scale = target_h / orig_h if orig_h > 0 else 1.0
    out = {}
    for k, v in kp_dict.items():
        if v is not None and np.all(np.isfinite(v)):
            out[k] = np.array([v[0] * scale, v[1] * scale])
        else:
            out[k] = v
    return out


def _annotate_frame(bgr: np.ndarray, kp_dict: dict,
                    metric_key: str, tl_bucket: str,
                    sport: str = 'running') -> np.ndarray:
    """
    Draw limb lines, optional vertical reference line, and angle arc
    for a specific metric onto a copy of bgr.
    """
    draw_table = METRIC_DRAW_BY_SPORT.get(sport, METRIC_DRAW)
    spec = draw_table.get(metric_key)
    if spec is None:
        return bgr

    img = bgr.copy()
    h, w = img.shape[:2]
    scale = h / 400.0

    bucket_color_bgr = {
        'GREEN': (50, 200, 80),
        'AMBER': (40, 160, 220),
        'RED':   (60, 60, 220),
    }.get(tl_bucket, (200, 200, 200))

    for (ja, jb) in spec.get('limbs', []):
        _draw_line(img, kp_dict.get(ja), kp_dict.get(jb),
                   color=bucket_color_bgr)

    vert_joint = spec.get('vertical')
    vert_joints = [vert_joint] if isinstance(vert_joint, str) else (vert_joint or [])
    for vj in vert_joints:
        pt = kp_dict.get(vj)
        if pt is not None and np.all(np.isfinite(pt)):
            cx, cy = int(pt[0]), int(pt[1])
            ref_len  = int(70 * scale)
            dash_len = int(8 * scale)
            gap_len  = int(5 * scale)
            y = cy - ref_len
            drawing = True
            while y < cy + ref_len:
                y_end = min(y + dash_len, cy + ref_len)
                if drawing:
                    cv2.line(img, (cx, max(0, y)), (cx, min(h - 1, y_end)),
                            (255, 255, 255), max(1, int(1.0 * scale)),
                            cv2.LINE_AA)
                y += dash_len + gap_len
                drawing = not drawing

    horiz_joint = spec.get('horizontal')
    horiz_joints = [horiz_joint] if isinstance(horiz_joint, str) else (horiz_joint or [])
    for hj in horiz_joints:
        pt = kp_dict.get(hj)
        if pt is not None and np.all(np.isfinite(pt)):
            cx, cy = int(pt[0]), int(pt[1])
            ref_len  = int(70 * scale)
            dash_len = int(8 * scale)
            gap_len  = int(5 * scale)
            x = cx - ref_len
            drawing = True
            while x < cx + ref_len:
                x_end = min(x + dash_len, cx + ref_len)
                if drawing:
                    cv2.line(img, (max(0, x), cy), (min(w - 1, x_end), cy),
                            (255, 255, 255), max(1, int(1.0 * scale)),
                            cv2.LINE_AA)
                x += dash_len + gap_len
                drawing = not drawing

    arc = spec.get('arc')
    if arc:
        _draw_angle_arc(img, kp_dict.get(arc[0]),
                        kp_dict.get(arc[1]),
                        kp_dict.get(arc[2]),
                        color=bucket_color_bgr)

    conv = spec.get('converge_arc')
    if conv:
        a_hip_n, a_knee_n, b_hip_n, b_knee_n = conv
        _draw_converging_angle(img,
                               kp_dict.get(a_hip_n), kp_dict.get(a_knee_n),
                               kp_dict.get(b_hip_n), kp_dict.get(b_knee_n),
                               color=bucket_color_bgr)

    for (joint_name, dot_label) in spec.get('dots', []):
        pt = kp_dict.get(joint_name)
        if pt is None or not np.all(np.isfinite(pt)):
            continue
        px, py = int(pt[0]), int(pt[1])
        base_r = spec.get('dot_radius', 10)
        r = max(5, int(base_r * scale * 0.7))
        cv2.circle(img, (px, py), r, bucket_color_bgr, -1)
        cv2.circle(img, (px, py), r + 1, (255, 255, 255), 1)
        cv2.putText(img, dot_label, (px + r + 4, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale,
                    (255, 255, 255), max(1, int(1.1 * scale)), cv2.LINE_AA)

    return img


def _tip_box_height(c: canvas.Canvas, w, event_metrics: list, tips: dict) -> float:
    """
    Height the tip panel needs for its content.

    Pages with one metric would otherwise get a panel stretched down the whole
    column with a single line floating at the top of it.
    """
    pad    = 0.35 * cm
    max_w  = w - 2 * pad - 0.35 * cm
    needed = pad * 2 + 30
    for pm in event_metrics:
        tip   = tips.get(pm['key'], '')
        label = _strip_bracket_suffix(pm.get('label', pm['key']))
        lw    = c.stringWidth(f'{label}: ', 'Helvetica-Bold', 9.5)
        lines = 1
        if tip:
            first = _wrap_text_to_width(c, tip, 'Helvetica', 9.5, max_w - lw)
            head  = first[0] if first else ''
            tail  = tip[len(head):].strip()
            lines = 1 + (len(_wrap_text_to_width(c, tail, 'Helvetica', 9.5, max_w))
                         if tail else 0)
        needed += lines * 13 + 5.0
    return needed


def _draw_tip_box(c: canvas.Canvas, x, y, w, h,
                  event_metrics: list, param_scores: dict, tips: dict):
    """
    Draw the "What we're looking for" panel: one traffic-light dot and one
    plain-English line per metric on the page.  Shared by every sport so the
    commentary block does not have to be re-implemented per report page.
    """
    tip_pad = 0.35 * cm

    c.setFillColor(C['header_lt'])
    c.setStrokeColor(C['header'])
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=1)

    c.setFillColor(C['header'])
    c.setFont('Helvetica-Bold', 12)
    c.drawString(x + tip_pad, y + h - tip_pad - 10, "What we're looking for")

    name_font, name_size = 'Helvetica-Bold', 9.5
    body_font, body_size = 'Helvetica', 9.5
    leading = 13
    dot_r   = 0.09 * cm
    text_x  = x + tip_pad + 0.35 * cm
    max_w   = w - 2 * tip_pad - 0.35 * cm

    text_y = y + h - tip_pad - 30
    for pm in event_metrics:
        if text_y < y + tip_pad:
            break
        key    = pm['key']
        tlabel = _strip_bracket_suffix(pm.get('label', key))
        tip    = tips.get(key, '')
        bucket = _tl_bucket(param_scores.get(key))
        dot_col, _ = _tl_colors(bucket)

        c.setFillColor(dot_col)
        c.circle(x + tip_pad, text_y + 3, dot_r, fill=1, stroke=0)

        if tlabel:
            c.setFont(name_font, name_size)
            c.setFillColor(C['header'])
            c.drawString(text_x, text_y, f'{tlabel}:')
            label_w = c.stringWidth(f'{tlabel}: ', name_font, name_size)

            c.setFont(body_font, body_size)
            c.setFillColor(C['txt'])
            # First line shares the row with the metric name, so it is wrapped
            # to the narrower width; continuation lines use the full column.
            if tip:
                first = _wrap_text_to_width(c, tip, body_font, body_size,
                                            max_w - label_w)
                head  = first[0] if first else ''
                tail  = tip[len(head):].strip()
                rest_lines = ([head] +
                              (_wrap_text_to_width(c, tail, body_font,
                                                   body_size, max_w)
                               if tail else []))
            else:
                rest_lines = []
            if rest_lines:
                c.drawString(text_x + label_w, text_y, rest_lines[0])
                text_y -= leading
                for extra in rest_lines[1:]:
                    c.drawString(text_x, text_y, extra)
                    text_y -= leading
            else:
                text_y -= leading
        text_y -= 5.0


def _score_badge(c: canvas.Canvas, x, y, w, h, bucket: str, label: str):
    """Draw a filled rounded rectangle badge."""
    txt_col, bg_col = _tl_colors(bucket)
    c.setFillColor(bg_col)
    c.roundRect(x, y, w, h, 4, fill=1, stroke=0)
    c.setFillColor(txt_col)
    c.setFont('Helvetica-Bold', 8)
    c.drawCentredString(x + w / 2, y + h / 2 - 3, label)


def _header_bar(c: canvas.Canvas, title: str, subtitle: str = ''):
    """Draw the page header bar."""
    bar_h = 1.1 * cm
    c.setFillColor(C['header'])
    c.rect(0, PAGE_H - bar_h, PAGE_W, bar_h, fill=1, stroke=0)
    c.setFillColor(C['white'])
    c.setFont('Helvetica-Bold', 13)
    c.drawString(MARGIN, PAGE_H - bar_h + 0.3 * cm, title)
    if subtitle:
        c.setFont('Helvetica', 9)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - bar_h + 0.35 * cm, subtitle)


def _page_background(c: canvas.Canvas):
    c.setFillColor(C['bg'])
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)


def _rounded_card(c: canvas.Canvas, x, y, w, h, fill=None, stroke_col=None):
    fill_col = fill or C['card_bg']
    c.setFillColor(fill_col)
    c.setStrokeColor(stroke_col or C['border'])
    c.setLineWidth(0.5)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=1)


def _build_overview_page(c: canvas.Canvas, data: dict,
                         keypoints: np.ndarray, video_path: str,
                         crop_box=None):
    _page_background(c)

    sport   = data.get('sport', 'running')
    backend = data.get('backend', '').upper()
    side    = (data.get('dominant_side') or data.get('near_side') or 'unknown').capitalize()
    n_cyc   = data.get('n_cycles', 0)
    n_excl  = data.get('n_excluded', 0)
    fps     = data.get('fps', 30)
    total   = data.get('total_score', 0)
    date_s  = datetime.now().strftime('%d %b %Y')

    cfg            = SPORT_CONFIGS.get(sport, SPORT_CONFIGS['running'])
    sport_events   = cfg['events']
    sport_labels   = cfg['labels']
    sport_fk       = cfg['frames_key']
    frames_source  = cfg['frames_source']
    frame_pool     = data.get(frames_source, {})

    _excl_tag = f'   |   {n_excl} excluded' if n_excl else ''
    subtitle = (f'{date_s}   |   {backend}   |   {side} side   |   '
                f'{n_cyc} cycles{_excl_tag}   |   {fps} fps')
    _header_bar(c, f'{sport.capitalize()} Biomechanical Assessment', subtitle)

    HEADER_H = 1.1 * cm
    top_y    = PAGE_H - HEADER_H - MARGIN

    params_meta_for_score = data.get('params_meta', [])
    param_scores_for_score = data.get('param_scores', {})
    _all_keys = ([p['key'] for p in params_meta_for_score]
                 if params_meta_for_score else list(param_scores_for_score.keys()))
    score_fraction, total_metrics = _metric_points(param_scores_for_score, _all_keys)
    if score_fraction is None:
        score_fraction = 0.0

    score_bucket = _tl_bucket(total)
    score_col, score_bg = _tl_colors(score_bucket)

    circle_r  = 1.5 * cm
    circle_cx = MARGIN + circle_r
    circle_cy = top_y - 1.8 * cm

    c.setFillColor(score_bg)
    c.circle(circle_cx, circle_cy, circle_r, fill=1, stroke=0)
    c.setFillColor(score_col)
    c.setFont('Helvetica-Bold', 24)
    c.drawCentredString(circle_cx, circle_cy + 0.32 * cm, _fmt_frac(score_fraction))
    c.setFont('Helvetica-Bold', 14)
    c.setFillColor(C['txt_light'])
    c.drawCentredString(circle_cx, circle_cy - 0.65 * cm, f'/{total_metrics}')

    c.setFillColor(C['txt'])
    c.setFont('Helvetica-Bold', 11)
    c.drawString(circle_cx * 2 + 0.3 * cm, circle_cy + 0.2 * cm, 'Overall Score')
    c.setFont('Helvetica', 9)
    c.setFillColor(C['txt_light'])
    _analysed_txt = f'{n_cyc} gait cycles analysed'
    if n_excl:
        _analysed_txt += f'  ({n_excl} excluded — low TD confidence)'
    c.drawString(circle_cx * 2 + 0.3 * cm, circle_cy - 0.35 * cm, _analysed_txt)

    warnings_list = data.get('warnings', [])
    if warnings_list:
        c.setFont('Helvetica', 7.5)
        c.setFillColor(C['txt_light'])
        c.drawString(circle_cx * 2 + 0.3 * cm, circle_cy - 0.7 * cm,
                     '; '.join(warnings_list))

    raw_traces   = data.get('raw_traces', {})
    param_scores = data.get('param_scores', {})
    params_meta  = data.get('params_meta', [])
    near_side    = data.get('dominant_side') or data.get('near_side', 'right')
    backend      = data.get('backend', 'rtmpose').lower()

    card_area_x = MARGIN
    card_area_y = MARGIN
    card_area_w = CONTENT_W
    card_area_h = top_y - 3.8 * cm - MARGIN

    n_cards = len(sport_events)
    gap     = 0.35 * cm
    card_w  = (card_area_w - gap * (n_cards - 1)) / n_cards
    card_h  = card_area_h
    thumb_h = card_h - 3.5 * cm

    for i, event in enumerate(sport_events):
        cx = card_area_x + i * (card_w + gap)
        cy = card_area_y

        frames_key   = sport_fk[event]
        event_metrics = [p['key'] for p in params_meta
                         if p.get('event') == event
                         or (not p.get('event') and p.get('event_frames_key') == frames_key)]
        scores    = [param_scores.get(k) for k in event_metrics
                     if param_scores.get(k) is not None]
        ev_score  = float(np.mean(scores)) if scores else None
        ev_bucket = _tl_bucket(ev_score)
        ev_tc, ev_bg = _tl_colors(ev_bucket)
        ev_frac, ev_n_metrics = _metric_points(param_scores, event_metrics)

        _rounded_card(c, cx, cy, card_w, card_h, stroke_col=ev_tc)

        c.setFillColor(ev_tc)
        c.roundRect(cx, cy + card_h - 0.75 * cm, card_w, 0.75 * cm,
                    6, fill=1, stroke=0)
        c.rect(cx, cy + card_h - 0.75 * cm, card_w, 0.375 * cm,
               fill=1, stroke=0)
        c.setFillColor(C['white'])
        c.setFont('Helvetica-Bold', 10)
        c.drawCentredString(cx + card_w / 2,
                            cy + card_h - 0.52 * cm,
                            sport_labels[event])

        frame_list = frame_pool.get(frames_key, [])
        frame_list = [f for f in frame_list if f is not None]
        thumb_img  = None
        inner_pad  = 0.2 * cm

        if frame_list and video_path:
            candidates = [_median_frame(frame_list)] + frame_list
            for fi in candidates:
                raw_bgr = _extract_frame_raw(video_path, int(fi))
                if raw_bgr is not None:
                    kp_d = _kp_dict_at_frame(keypoints, int(fi), near_side, backend) \
                           if keypoints is not None else {}
                    if crop_box is not None:
                        raw_bgr = _apply_crop_box(raw_bgr, crop_box)
                    else:
                        raw_bgr = _smart_crop_to_person(raw_bgr, kp_d)
                    box_w = card_w - 2 * inner_pad
                    box_h = thumb_h - inner_pad
                    _fit_bgr, _fw, _fh, _ox, _oy = _fit_to_box(raw_bgr, box_w, box_h)
                    thumb_img = _frame_to_image_reader(_fit_bgr)
                    _thumb_draw_w = _fw
                    _thumb_draw_h = _fh
                    _thumb_ox     = _ox
                    _thumb_oy     = _oy
                    _thumb_box_x  = cx + inner_pad
                    _thumb_box_y  = cy + card_h - 0.75 * cm - thumb_h
                    break

        if thumb_img:
            c.drawImage(thumb_img,
                        _thumb_box_x + _thumb_ox,
                        _thumb_box_y + _thumb_oy,
                        width=_thumb_draw_w,
                        height=_thumb_draw_h,
                        mask='auto')
        else:
            c.setFillColor(C['border'])
            c.rect(cx + inner_pad,
                   cy + card_h - 0.75 * cm - thumb_h + inner_pad,
                   card_w - 2 * inner_pad, thumb_h - 2 * inner_pad,
                   fill=1, stroke=0)
            c.setFillColor(C['txt_light'])
            c.setFont('Helvetica', 8)
            c.drawCentredString(cx + card_w / 2,
                                cy + card_h - 0.75 * cm - thumb_h / 2,
                                'No video')

        pill_w, pill_h_val = 2.0 * cm, 0.55 * cm
        pill_x = cx + card_w / 2 - pill_w / 2
        pill_y = cy + 0.65 * cm
        pill_label = f'{_fmt_frac(ev_frac)}/{ev_n_metrics}' if ev_frac is not None else '—'
        _score_badge(c, pill_x, pill_y, pill_w, pill_h_val, ev_bucket, pill_label)

        c.setFillColor(C['txt_light'])
        c.setFont('Helvetica', 7)
        c.drawCentredString(cx + card_w / 2, cy + 0.3 * cm, 'Event Score')

        metric_y  = cy + 1.4 * cm
        dot_r     = 0.12 * cm
        dot_gap   = 0.22 * cm
        mini_font, mini_size = 'Helvetica', 7
        max_chars = int(card_w / (0.18 * cm))

        for k in event_metrics:
            s = param_scores.get(k)
            b = _tl_bucket(s)
            dot_col, _ = _tl_colors(b)
            label = _strip_bracket_suffix(next((p['label'] for p in params_meta if p['key'] == k), k))
            lbl_trunc = label[:max_chars] if len(label) > max_chars else label

            lbl_w   = c.stringWidth(lbl_trunc, mini_font, mini_size)
            block_w = 2 * dot_r + dot_gap + lbl_w
            start_x = cx + (card_w - block_w) / 2

            c.setFillColor(dot_col)
            c.circle(start_x + dot_r, metric_y + 0.1 * cm, dot_r, fill=1, stroke=0)
            c.setFillColor(C['txt'])
            c.setFont(mini_font, mini_size)
            c.drawString(start_x + 2 * dot_r + dot_gap, metric_y, lbl_trunc)
            metric_y += 0.38 * cm


def _build_rowing_overview_page(c: canvas.Canvas, data: dict,
                                 keypoints: np.ndarray, video_path: str,
                                 crop_box=None):
    """
    Landscape title page for rowing — layout matches PowerPoint reference:
      Col A  (~18%)  Coloured event label block (square, bold, centred text)
      Col B  (~38%)  Event image (clean, no overlay strip)
      Col C  (~44%)  Rounded metric card with value + pass/fail per metric
    Four rows (Catch / Drive / Finish / Recovery) tightly stacked.
    Overall score pill top-right above the cards column.
    """
    _page_background(c)

    backend   = data.get('backend', '').upper()
    side      = (data.get('dominant_side') or data.get('near_side') or 'unknown').capitalize()
    n_cyc     = data.get('n_cycles', 0)
    fps       = data.get('fps', 30)
    total     = data.get('total_score', 0)
    date_s    = datetime.now().strftime('%d %b %Y')
    near_side = (data.get('dominant_side') or data.get('near_side', 'right'))
    bknd      = data.get('backend', 'rtmpose').lower()

    subtitle = f'{date_s}   |   {backend}   |   {side} side   |   {n_cyc} cycles   |   {fps} fps'
    _header_bar(c, 'Rowing Biomechanical Assessment', subtitle)

    HEADER_H  = 1.1 * cm
    LEGEND_H  = 1.25 * cm
    top_y     = PAGE_H - HEADER_H - MARGIN
    avail_h   = top_y - MARGIN - LEGEND_H
    avail_w   = CONTENT_W

    _draw_rowing_legend(c, MARGIN, MARGIN * 0.55, avail_w, LEGEND_H - 0.25 * cm)

    cfg          = SPORT_CONFIGS['rowing']
    events       = cfg['events']
    labels       = cfg['labels']
    fk           = cfg['frames_key']
    key_frames   = data.get('key_frames', {})
    param_scores = data.get('param_scores', {})
    params_meta  = data.get('params_meta', [])

    n_events = len(events)
    ROW_GAP  = 0.20 * cm
    row_h    = (avail_h - ROW_GAP * (n_events - 1)) / n_events

    LABEL_W = avail_w * 0.18
    IMG_W   = avail_w * 0.38
    GAP_LI  = 0.20 * cm
    GAP_IC  = 0.30 * cm
    CARD_W  = avail_w - LABEL_W - IMG_W - GAP_LI - GAP_IC

    label_x = MARGIN
    img_x   = MARGIN + LABEL_W + GAP_LI
    card_x  = img_x + IMG_W + GAP_IC

    all_keys       = [p['key'] for p in params_meta]
    points, n_mets = _metric_points(param_scores, all_keys)

    score_bucket = _tl_bucket(total)
    sc, sbg = _tl_colors(score_bucket)
    pill_w, pill_h = 3.2 * cm, 0.72 * cm
    pill_x = card_x + CARD_W - pill_w
    pill_y = top_y + 0.08 * cm
    c.setFillColor(sbg)
    c.roundRect(pill_x, pill_y, pill_w, pill_h, 4, fill=1, stroke=0)
    c.setFillColor(sc)
    c.setFont('Helvetica-Bold', 12)
    _overall_s = (f'Overall: {_fmt_frac(points)}/{n_mets}'
                  if points is not None else 'Overall: —')
    c.drawCentredString(pill_x + pill_w / 2, pill_y + 0.18 * cm, _overall_s)

    for i, event in enumerate(events):
        row_bottom = top_y - (i + 1) * row_h - i * ROW_GAP

        ev_metrics = [p['key'] for p in params_meta if p.get('event') == event]
        ev_scores  = [param_scores.get(k) for k in ev_metrics
                      if param_scores.get(k) is not None]
        ev_score   = float(np.mean(ev_scores)) if ev_scores else None
        ev_bucket  = _tl_bucket(ev_score)
        ev_tc, _   = _tl_colors(ev_bucket)
        ev_points, ev_n = _metric_points(param_scores, ev_metrics)

        c.setFillColor(ev_tc)
        c.roundRect(label_x, row_bottom, LABEL_W, row_h, 4, fill=1, stroke=0)
        c.setFillColor(C['white'])
        c.setFont('Helvetica-Bold', 11)
        c.drawCentredString(label_x + LABEL_W / 2,
                            row_bottom + row_h / 2 + 2,
                            labels[event])
        if ev_points is not None:
            c.setFont('Helvetica', 8)
            c.drawCentredString(label_x + LABEL_W / 2,
                                row_bottom + row_h / 2 - 12,
                                f'{_fmt_frac(ev_points)}/{ev_n}   Event Score')

        frames_key = fk[event]
        frame_list = key_frames.get(frames_key, [])
        frame_list = [f for f in frame_list if f is not None]
        med_frame  = _median_frame(frame_list) if frame_list else None

        img_drawn = False
        if med_frame is not None and video_path:
            raw_bgr = _extract_frame_raw(video_path, int(med_frame))
            if raw_bgr is not None:
                cropped = _apply_crop_box(raw_bgr, crop_box) if crop_box else                           _smart_crop_to_person(
                              raw_bgr,
                              _kp_dict_at_frame(keypoints, int(med_frame), near_side, bknd)
                              if keypoints is not None else {})
                fit_bgr, fw, fh, ox, oy = _fit_to_box(cropped, IMG_W, row_h)
                ir = _frame_to_image_reader(fit_bgr)
                c.drawImage(ir, img_x + ox, row_bottom + oy,
                            width=fw, height=fh, mask='auto')
                img_drawn = True

        if not img_drawn:
            c.setFillColor(C['border'])
            c.rect(img_x, row_bottom, IMG_W, row_h, fill=1, stroke=0)

        c.setFillColor(C['card_bg'])
        c.setStrokeColor(ev_tc)
        c.setLineWidth(1.2)
        c.roundRect(card_x, row_bottom, CARD_W, row_h, 6, fill=1, stroke=1)

        n_metrics = len(ev_metrics)
        if n_metrics > 0:
            metric_block_h = row_h * 0.78
            metric_row_h   = metric_block_h / n_metrics
            metric_start_y = row_bottom + (row_h + metric_block_h) / 2

            for mi, k in enumerate(ev_metrics):
                s      = param_scores.get(k)
                bucket = _tl_bucket(s)
                tc, _  = _tl_colors(bucket)
                mlabel = next((p['label'] for p in params_meta if p['key'] == k), k)
                val    = data.get('mean_values', {}).get(k)
                unit   = next((p.get('unit', '°') for p in params_meta if p['key'] == k), '°')
                val_s  = _fmt_value(val, unit, dp=0)

                my = metric_start_y - (mi + 0.5) * metric_row_h - 4

                c.setFillColor(C['txt'])
                c.setFont('Helvetica', 8.5)
                c.drawString(card_x + 0.4 * cm, my, mlabel)

                pm_k = next((p for p in params_meta if p['key'] == k), None)
                if pm_k is not None:
                    c.setFillColor(C['txt_light'])
                    c.setFont('Helvetica', 7.5)
                    c.drawString(card_x + CARD_W * 0.47, my,
                                 f'Target {_fmt_target(pm_k)}')

                BADGE_W = 1.3 * cm
                BADGE_H = min(metric_row_h * 0.60, 0.45 * cm)
                badge_x = card_x + CARD_W - BADGE_W - 0.3 * cm
                val_x   = badge_x - 0.2 * cm
                c.setFont('Helvetica-Bold', 8.5)
                c.drawRightString(val_x, my, val_s)

                _score_badge(c, badge_x, my - BADGE_H / 2 + 4, BADGE_W, BADGE_H,
                             bucket, BADGE_LABELS[bucket])


def _draw_rowing_legend(c: canvas.Canvas, x, y, w, h):
    """Traffic-light key and accuracy note along the foot of the rowing cover."""
    err = ROWING_ANGLE_ERROR_DEG
    items = [
        ('GREEN', 'PASS',  'Good. Within the target.'),
        ('AMBER', 'AMBER', 'Borderline. Close to the edge of the target, or '
                           'movements overlapping.'),
        ('RED',   'FAIL',  'Needs work. Outside the target, or moving in the '
                           'wrong order.'),
    ]
    note_w = 5.2 * cm
    col_w  = (w - note_w) / 3
    BADGE_W, BADGE_H = 1.15 * cm, 0.42 * cm
    for i, (bucket, lab, text) in enumerate(items):
        bx = x + i * col_w
        _score_badge(c, bx, y + h - BADGE_H, BADGE_W, BADGE_H, bucket, lab)
        c.setFillColor(C['txt'])
        c.setFont('Helvetica', 7.2)
        lines = _wrap_text_to_width(c, text, 'Helvetica', 7.2,
                                    col_w - BADGE_W - 0.45 * cm)
        for li, ln in enumerate(lines[:3]):
            c.drawString(bx + BADGE_W + 0.2 * cm, y + h - 8 - li * 8.6, ln)
    nx = x + 3 * col_w
    c.setFillColor(C['txt_light'])
    c.setFont('Helvetica-Oblique', 6.8)
    for li, ln in enumerate(_wrap_text_to_width(
            c, f'Angles are approximate (about ±{err}°). New to this? See '
               '"How to read this report" on the last page.',
            'Helvetica-Oblique', 6.8, note_w)[:3]):
        c.drawString(nx, y + h - 8 - li * 8.2, ln)


# Plain verdict for the sequencing pages: one overall word, then one yes /
# amber / no question per handover.  Written for athletes and coaches.
SEQ_VERDICT_WORD = {'GREEN': 'GOOD', 'AMBER': 'NEEDS WORK', 'RED': 'POOR'}
SEQ_ANSWER_WORD  = {'GREEN': 'YES', 'AMBER': 'AMBER', 'RED': 'NO'}

SEQ_ORDER_TEXT = {
    'drive':    'Legs, then back, then arms.',
    'recovery': 'Arms, then back, then legs.',
}

# Body-part names used in the questions.
SEQ_QUESTION_PART = {'leg': 'Legs', 'trunk': 'back', 'arm': 'arms'}


def _seq_question(ho: dict) -> str:
    a = SEQ_QUESTION_PART[ho['from']]
    b = SEQ_QUESTION_PART[ho['to']]
    return f'{a[0].upper() + a[1:]} before {b.lower()}?'


def _draw_seq_verdict_card(c: canvas.Canvas, x, y_top, w, phase: str, seq: dict):
    """
    Overall GOOD / NEEDS WORK / POOR for the phase, then one question per
    handover ("Legs before back?") with a YES / AMBER / NO badge.
    Returns the card height.
    """
    PAD   = 0.4 * cm
    label = seq.get('label', '—') if seq else '—'
    tc, bg = _tl_colors(label)
    handovers = (seq or {}).get('handovers') or []

    Q_H = 0.85 * cm
    h = 1.75 * cm + 0.2 * cm + max(len(handovers), 1) * Q_H + 0.9 * cm
    y = y_top - h

    c.setFillColor(bg)
    c.setStrokeColor(tc)
    c.setLineWidth(1.2)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=1)

    c.setFillColor(C['txt'])
    c.setFont('Helvetica-Bold', 9.5)
    c.drawString(x + PAD, y_top - 0.6 * cm,
                 f'{phase.capitalize()}: is the order right?')
    c.setFillColor(tc)
    c.setFont('Helvetica-Bold', 22)
    c.drawString(x + PAD, y_top - 1.45 * cm, SEQ_VERDICT_WORD.get(label, '—'))

    qy = y_top - 1.75 * cm - 0.2 * cm
    BW, BH = 1.6 * cm, 0.55 * cm
    if not handovers:
        c.setFillColor(C['txt'])
        c.setFont('Helvetica', 9)
        c.drawString(x + PAD, qy - Q_H / 2, 'Could not be measured on this video.')
    for i, ho in enumerate(handovers):
        ry = qy - (i + 1) * Q_H
        c.setFillColor(C['card_bg'])
        c.roundRect(x + PAD * 0.6, ry + 0.08 * cm, w - PAD * 1.2, Q_H - 0.16 * cm,
                    4, fill=1, stroke=0)
        c.setFillColor(C['txt'])
        c.setFont('Helvetica-Bold', 11)
        c.drawString(x + PAD, ry + Q_H / 2 - 4, _seq_question(ho))
        _score_badge(c, x + w - PAD - BW, ry + (Q_H - BH) / 2, BW, BH,
                     ho['label'], SEQ_ANSWER_WORD.get(ho['label'], '—'))

    c.setStrokeColor(tc)
    c.setLineWidth(0.5)
    c.line(x + PAD, y + 0.75 * cm, x + w - PAD, y + 0.75 * cm)
    c.setFillColor(C['txt'])
    c.setFont('Helvetica-Bold', 8)
    c.drawString(x + PAD, y + 0.3 * cm, 'Correct order:')
    c.setFont('Helvetica', 8)
    c.drawString(x + PAD + c.stringWidth('Correct order: ', 'Helvetica-Bold', 8),
                 y + 0.3 * cm, SEQ_ORDER_TEXT[phase])
    return h


def _build_rowing_howto_page(c: canvas.Canvas, data: dict):
    """Closing page: a plain guide to the report for athletes and coaches."""
    _page_background(c)
    _header_bar(c, 'How to read this report', 'Rowing Biomechanical Assessment')

    err = ROWING_ANGLE_ERROR_DEG
    top = PAGE_H - 1.1 * cm - 0.8 * cm

    # --- Intro line
    c.setFillColor(C['txt'])
    c.setFont('Helvetica', 10)
    c.drawString(MARGIN, top,
                 'This report checks your rowing technique from one video filmed '
                 'from the side. No sensors or markers are needed.')

    # --- Three steps across the page
    steps = [
        ('1', 'Start on page 1',
         'Page 1 is your summary. Each row is one part of the stroke: catch, '
         'drive, finish and recovery.'),
        ('2', 'Look at the colours',
         'Focus on red first, then amber. Green means that part is fine.'),
        ('3', 'Turn to that page',
         'Each part of the stroke has its own page, with a picture and a '
         'box saying what good looks like.'),
    ]
    gap   = 0.5 * cm
    box_w = (CONTENT_W - 2 * gap) / 3
    box_h = 3.1 * cm
    by    = top - 0.6 * cm - box_h
    for i, (num, title, body) in enumerate(steps):
        bx = MARGIN + i * (box_w + gap)
        _rounded_card(c, bx, by, box_w, box_h)
        c.setFillColor(C['header'])
        c.circle(bx + 0.75 * cm, by + box_h - 0.75 * cm, 0.38 * cm, fill=1, stroke=0)
        c.setFillColor(C['white'])
        c.setFont('Helvetica-Bold', 12)
        c.drawCentredString(bx + 0.75 * cm, by + box_h - 0.75 * cm - 4, num)
        c.setFillColor(C['header'])
        c.setFont('Helvetica-Bold', 11)
        c.drawString(bx + 1.4 * cm, by + box_h - 0.75 * cm - 4, title)
        c.setFillColor(C['txt'])
        c.setFont('Helvetica', 9)
        for li, ln in enumerate(_wrap_text_to_width(c, body, 'Helvetica', 9,
                                                    box_w - 0.9 * cm)):
            c.drawString(bx + 0.45 * cm, by + box_h - 1.65 * cm - li * 11.5, ln)

    # --- Colours (left) and order (right)
    row_top = by - 0.5 * cm
    half_w  = (CONTENT_W - gap) / 2
    card_h  = 4.3 * cm

    lx = MARGIN
    _rounded_card(c, lx, row_top - card_h, half_w, card_h)
    c.setFillColor(C['header'])
    c.setFont('Helvetica-Bold', 11)
    c.drawString(lx + 0.45 * cm, row_top - 0.7 * cm, 'What the colours mean')
    rows = [('GREEN', 'PASS',  'Good. Keep doing what you are doing.'),
            ('AMBER', 'AMBER', 'Borderline. Close to the limit. Worth a look.'),
            ('RED',   'FAIL',  'Needs work. This is the thing to fix first.')]
    for i, (bucket, lab, text) in enumerate(rows):
        ry = row_top - 1.55 * cm - i * 0.85 * cm
        _score_badge(c, lx + 0.45 * cm, ry - 0.1 * cm, 1.4 * cm, 0.5 * cm, bucket, lab)
        c.setFillColor(C['txt'])
        c.setFont('Helvetica', 9.5)
        c.drawString(lx + 2.15 * cm, ry + 0.02 * cm, text)

    rx = MARGIN + half_w + gap
    _rounded_card(c, rx, row_top - card_h, half_w, card_h)
    c.setFillColor(C['header'])
    c.setFont('Helvetica-Bold', 11)
    c.drawString(rx + 0.45 * cm, row_top - 0.7 * cm, 'Moving in the right order')
    c.setFillColor(C['txt'])
    c.setFont('Helvetica', 9.5)
    c.drawString(rx + 0.45 * cm, row_top - 1.35 * cm,
                 'The drive and recovery pages check the order your body moves in.')

    # Two little step diagrams: bars following each other like stairs.
    for j, (ttl, parts) in enumerate((('Drive', ('Legs', 'Back', 'Arms')),
                                      ('Recovery', ('Arms', 'Back', 'Legs')))):
        dx = rx + 0.45 * cm + j * (half_w - 0.9 * cm) / 2
        dw = (half_w - 0.9 * cm) / 2 - 0.4 * cm
        c.setFillColor(C['txt'])
        c.setFont('Helvetica-Bold', 9)
        c.drawString(dx, row_top - 2.05 * cm, ttl)
        seg_keys = {'Legs': 'leg', 'Back': 'trunk', 'Arms': 'arm'}
        for k, p in enumerate(parts):
            yy = row_top - 2.55 * cm - k * 0.45 * cm
            c.setFillColor(C['txt'])
            c.setFont('Helvetica', 8)
            c.drawString(dx, yy, p)
            bx0 = dx + 1.0 * cm + k * (dw - 1.0 * cm) / 3
            c.setFillColor(SEQ_SEGMENT_COLORS[seg_keys[p]])
            c.roundRect(bx0, yy - 0.02 * cm, (dw - 1.0 * cm) / 3, 0.22 * cm,
                        0.11 * cm, fill=1, stroke=0)
    c.setFillColor(C['txt_light'])
    c.setFont('Helvetica', 8)
    c.drawString(rx + 0.45 * cm, row_top - card_h + 0.3 * cm,
                 'Good technique looks like steps. Bars stacked on top of each other mean overlap.')

    # --- Good to know
    gy_top = row_top - card_h - 0.5 * cm
    g_h    = 1.35 * cm + 3 * 0.6 * cm
    _rounded_card(c, MARGIN, gy_top - g_h, CONTENT_W, g_h)
    c.setFillColor(C['header'])
    c.setFont('Helvetica-Bold', 11)
    c.drawString(MARGIN + 0.45 * cm, gy_top - 0.7 * cm, 'Good to know')
    tips = [
        f'Angles are a close estimate, about ±{err}°, so small differences '
        'are not worth worrying about.',
        'For the best results, film from the side with the camera level and '
        'your whole body in view.',
        'To track progress, film again the same way after working on a fault '
        'and compare the two reports.',
    ]
    c.setFont('Helvetica', 9.5)
    for i, t in enumerate(tips):
        ty = gy_top - 1.35 * cm - i * 0.6 * cm
        c.setFillColor(C['header'])
        c.circle(MARGIN + 0.6 * cm, ty + 3, 2, fill=1, stroke=0)
        c.setFillColor(C['txt'])
        c.drawString(MARGIN + 0.9 * cm, ty, t)


def _draw_sequencing_strip(c: canvas.Canvas, x, y, w, h,
                           phase: str, data: dict):
    """
    The sequencing timeline: one bar per segment across the phase.

    Each bar spans the middle half of that segment's work (from a quarter done
    to three quarters done) with a mark at its centroid, all on a common
    0-100% phase scale.  Read left to right: bars staggered in the labelled
    order is a clean sequence, bars stacked above one another is a segment
    firing at the same time as its neighbour, bars in the wrong order is a
    handover happening backwards.  It is the evidence behind the single number
    in the table, and it needs no legend because every bar is named.
    """
    seq   = (data.get('sequencing') or {}).get(phase) or {}
    prof  = seq.get('profiles') or {}
    order = ('leg', 'trunk', 'arm') if phase == 'drive' else ('arm', 'trunk', 'leg')

    _rounded_card(c, x, y, w, h)

    PAD      = 0.3 * cm
    LABEL_W  = 1.35 * cm
    TITLE_H  = 0.5 * cm
    AXIS_H   = 0.42 * cm
    x0 = x + PAD + LABEL_W
    x1 = x + w - PAD
    track = x1 - x0

    c.setFillColor(C['txt'])
    c.setFont('Helvetica-Bold', 8.5)
    title = f'When each body part moves during the {phase}'
    c.drawString(x + PAD, y + h - PAD - 7, title)

    if not prof or any(prof.get(k, {}).get('cent') is None for k in order):
        c.setFillColor(C['txt_light'])
        c.setFont('Helvetica', 8)
        c.drawCentredString(x + w / 2, y + h / 2, 'Sequencing not measurable')
        return

    def _px(pct):
        return x0 + max(0.0, min(100.0, float(pct))) / 100.0 * track

    if seq.get('method') == 'completion' and seq.get('angles'):
        _draw_completion_strip_body(c, x, y, w, h, phase, seq, order,
                                    x0, x1, _px, PAD, LABEL_W, TITLE_H, AXIS_H)
        return

    ANNOT_H  = 0.42 * cm
    rows_top = y + h - TITLE_H - PAD
    rows_h   = rows_top - (y + PAD + AXIS_H + ANNOT_H)
    row_h    = rows_h / len(order)
    BAR_H    = min(row_h * 0.55, 0.32 * cm)

    cents = {}
    for i, name in enumerate(order):
        p  = prof[name]
        cy = rows_top - (i + 0.5) * row_h - BAR_H / 2

        c.setFillColor(C['txt'])
        c.setFont('Helvetica', 7.5)
        c.drawRightString(x + PAD + LABEL_W - 0.18 * cm, cy + BAR_H / 2 - 2.6,
                          SEQ_SEGMENT_LABELS[name])

        # faint full-width rail so an early or late bar is read against the phase
        c.setFillColor(C['bg'])
        c.roundRect(x0, cy + BAR_H / 2 - 0.6, track, 1.2, 0.6, fill=1, stroke=0)

        bx0, bx1 = _px(p['q25']), _px(p['q75'])
        c.setFillColor(SEQ_SEGMENT_COLORS[name])
        c.roundRect(bx0, cy, max(bx1 - bx0, 1.5), BAR_H, BAR_H / 2,
                    fill=1, stroke=0)

        cxp = _px(p['cent'])
        cents[name] = (cxp, cy + BAR_H / 2)
        c.setStrokeColor(C['white'])
        c.setLineWidth(1.4)
        c.line(cxp, cy + 0.6, cxp, cy + BAR_H - 0.6)


    # One connector per handover, between the centroid marks, coloured by that
    # handover's own result.  No numbers: the questions on the left carry the
    # verdict, this just shows where it comes from.
    ho_lab = {(h['from'], h['to']): h['label'] for h in (seq.get('handovers') or [])}
    ay = y + PAD + AXIS_H + 0.14 * cm
    for k, (a, b) in enumerate(zip(order, order[1:])):
        xa, xb = cents[a][0], cents[b][0]
        yk  = ay + 3.5 - k * 5.0          # stacked so they never hide each other
        col = _tl_colors(ho_lab.get((a, b), seq.get('label', 'AMBER')))[0]
        c.setStrokeColor(col)
        c.setLineWidth(1.4)
        c.line(xa, yk, xb, yk)
        for xe in (xa, xb):
            c.line(xe, yk - 1.8, xe, yk + 1.8)

    # Axis
    ay = y + PAD + 0.1 * cm
    c.setStrokeColor(C['border'])
    c.setLineWidth(0.6)
    c.line(x0, ay + 0.22 * cm, x1, ay + 0.22 * cm)
    c.setFillColor(C['txt_light'])
    c.setFont('Helvetica', 6.5)
    for pct, lab in ((0, 'catch' if phase == 'drive' else 'finish'),
                     (25, '25%'), (50, '50%'), (75, '75%'),
                     (100, 'finish' if phase == 'drive' else 'catch')):
        px = _px(pct)
        c.line(px, ay + 0.14 * cm, px, ay + 0.22 * cm)
        if pct == 0:
            c.drawString(px, ay, lab)
        elif pct == 100:
            c.drawRightString(px, ay, lab)
        else:
            c.drawCentredString(px, ay, lab)


def _draw_strip_axis(c, x0, x1, y, PAD, phase, _px):
    ay = y + PAD + 0.1 * cm
    c.setStrokeColor(C['border'])
    c.setLineWidth(0.6)
    c.line(x0, ay + 0.22 * cm, x1, ay + 0.22 * cm)
    c.setFillColor(C['txt_light'])
    c.setFont('Helvetica', 6.5)
    for pct, lab in ((0, 'catch' if phase == 'drive' else 'finish'),
                     (25, '25%'), (50, '50%'), (75, '75%'),
                     (100, 'finish' if phase == 'drive' else 'catch')):
        px = _px(pct)
        c.line(px, ay + 0.14 * cm, px, ay + 0.22 * cm)
        if pct == 0:
            c.drawString(px, ay, lab)
        elif pct == 100:
            c.drawRightString(px, ay, lab)
        else:
            c.drawCentredString(px, ay, lab)


def _draw_completion_strip_body(c, x, y, w, h, phase, seq, order,
                                x0, x1, _px, PAD, LABEL_W, TITLE_H, AXIS_H):
    """
    Angle-based timeline: each bar runs from when that joint starts changing
    angle to when its change is complete (the green cut-off), so overlap
    between bars is overlap between movements.  A dashed marker at each next
    joint's start is labelled with how far through the previous joint was.
    """
    angles    = seq['angles']
    handovers = {ho['to']: ho for ho in (seq.get('handovers') or [])}
    track     = x1 - x0
    rows_top  = y + h - TITLE_H - PAD
    rows_h    = rows_top - (y + PAD + AXIS_H + 0.1 * cm)
    row_h     = rows_h / len(order)
    BAR_H     = min(row_h * 0.45, 0.3 * cm)

    centres = {}
    for i, name in enumerate(order):
        a  = angles[name]
        cy = rows_top - (i + 0.5) * row_h - BAR_H / 2
        centres[name] = cy
        c.setFillColor(C['txt'])
        c.setFont('Helvetica', 7.5)
        c.drawRightString(x + PAD + LABEL_W - 0.18 * cm, cy + BAR_H / 2 - 2.6,
                          SEQ_SEGMENT_LABELS[name])
        c.setFillColor(C['bg'])
        c.roundRect(x0, cy + BAR_H / 2 - 0.6, track, 1.2, 0.6, fill=1, stroke=0)
        bx0, bx1 = _px(a['onset']), _px(a['done'])
        c.setFillColor(SEQ_SEGMENT_COLORS[name])
        c.roundRect(bx0, cy, max(bx1 - bx0, 1.5), BAR_H, BAR_H / 2,
                    fill=1, stroke=0)

    for i, name in enumerate(order[1:], start=1):
        ho = handovers.get(name)
        if not ho:
            continue
        prev = order[i - 1]
        xs   = _px(angles[name]['onset'])
        y_hi = centres[prev] + BAR_H + 0.08 * cm
        y_lo = centres[name] + BAR_H / 2
        col  = _tl_colors(ho['label'])[0]
        c.setStrokeColor(col)
        c.setLineWidth(0.9)
        c.setDash(2, 1.5)
        c.line(xs, y_lo, xs, y_hi)
        c.setDash()
        c.setFillColor(col)
        c.setFont('Helvetica-Bold', 6.8)
        txt = f'{SEQ_SEGMENT_LABELS[name].lower()} starts'
        tw  = c.stringWidth(txt, 'Helvetica-Bold', 6.8)
        tx  = min(max(xs - tw / 2, x0), x1 - tw)
        c.drawString(tx, y_hi + 0.06 * cm, txt)

    _draw_strip_axis(c, x0, x1, y, PAD, phase, _px)


def _build_rowing_progression_page(c: canvas.Canvas, event: str, data: dict,
                                    keypoints: np.ndarray, video_path: str,
                                    crop_box=None):
    """
    Rowing drive/recovery page: metric table left, 3 progression frames
    (25%, 50%, 75% through the phase) stacked in a column on the right.
    """
    _page_background(c)

    sport    = 'rowing'
    cfg      = SPORT_CONFIGS['rowing']
    ev_label = cfg['labels'].get(event, event.capitalize())
    _header_bar(c, f'Rowing Assessment — {ev_label}', '')

    HEADER_H  = 1.1 * cm
    top_y     = PAGE_H - HEADER_H - 0.5 * cm
    near_side = data.get('dominant_side') or data.get('near_side', 'right')
    backend   = data.get('backend', 'rtmpose').lower()
    key_frames = data.get('key_frames', {})
    param_scores = data.get('param_scores', {})
    mean_values  = data.get('mean_values', {})
    params_meta  = data.get('params_meta', [])

    catch_list  = key_frames.get('catch', [])
    finish_list = key_frames.get('finish', [])
    catch_list  = [f for f in catch_list if f is not None]
    finish_list = [f for f in finish_list if f is not None]

    rep_catch      = key_frames.get('rep_catch')
    rep_finish     = key_frames.get('rep_finish')
    rep_next_catch = key_frames.get('rep_next_catch')

    # Drive runs catch -> finish of the representative cycle.  Recovery runs
    # from that finish forward to the NEXT catch -- not back to the same
    # cycle's catch, which is the drive window traversed backwards.
    if rep_next_catch is None and rep_finish is not None:
        later = [f for f in catch_list if f > int(rep_finish)]
        rep_next_catch = min(later) if later else None

    start_f = end_f = None
    if event == 'drive':
        if rep_catch is not None and rep_finish is not None:
            start_f, end_f = int(rep_catch), int(rep_finish)
    else:
        if rep_finish is not None and rep_next_catch is not None:
            start_f, end_f = int(rep_finish), int(rep_next_catch)

    if start_f is None or end_f is None or end_f <= start_f:
        def _pair_frames(starts, ends):
            pairs = []
            used = set()
            for s in sorted(starts):
                for e in sorted(ends):
                    if e > s and e not in used:
                        pairs.append((s, e))
                        used.add(e)
                        break
            return pairs

        if event == 'drive':
            pairs = _pair_frames(catch_list, finish_list)
        else:
            pairs = _pair_frames(finish_list, catch_list)

        if pairs:
            start_f, end_f = pairs[len(pairs) // 2]
        else:
            start_f = _median_frame(catch_list if event == 'drive' else finish_list) or 0
            end_f   = _median_frame(finish_list if event == 'drive' else catch_list) or 0

    # Never reorder the window: the direction of travel is the point of the
    # strip.  A window that is still inverted here means event detection
    # failed, and repeating one frame says so rather than hiding it.
    if end_f > start_f:
        prog_frames = [int(start_f + (end_f - start_f) * frac)
                       for frac in (0.25, 0.50, 0.75)]
    else:
        prog_frames = [int(start_f)] * 3

    event_metrics = [p for p in params_meta if p.get('event') == event]

    TABLE_W  = 8.2 * cm
    right_w  = PAGE_W - 2 * MARGIN - TABLE_W - 0.4 * cm
    images_x = MARGIN + TABLE_W + 0.4 * cm
    avail_h  = top_y - MARGIN

    ROW_H     = 1.0 * cm
    COL_STRIP = 0.18 * cm
    COL_PAD   = 0.25 * cm
    COL_NAME  = 3.6 * cm
    COL_ACT   = 1.4 * cm
    COL_TGT   = 1.5 * cm
    COL_BADGE = 1.3 * cm
    HDR_H     = 0.6 * cm

    table_x   = MARGIN
    table_top = top_y

    seq_ph = (data.get('sequencing') or {}).get(event) or {}
    if seq_ph:
        # Plain good / needs-work verdict with coaching feedback, in the
        # left-hand column where the metric table used to sit.
        table_h_used = _draw_seq_verdict_card(c, table_x, table_top, TABLE_W,
                                              event, seq_ph)
        event_metrics_table = []
    else:
        table_h_used = None
        event_metrics_table = event_metrics

    if table_h_used is None:
        c.setFillColor(C['header'])
        c.rect(table_x, table_top - HDR_H, TABLE_W, HDR_H, fill=1, stroke=0)
        c.setFillColor(C['white'])
        c.setFont('Helvetica-Bold', 8)
        hx = table_x + COL_STRIP + COL_PAD
        for hdr, cw in [('Metric', COL_NAME), ('Actual', COL_ACT),
                        ('Target', COL_TGT), ('', COL_BADGE)]:
            c.drawString(hx, table_top - HDR_H + 0.18 * cm, hdr)
            hx += cw

    row_y = table_top - HDR_H
    for idx, pm in enumerate(event_metrics_table):
        key   = pm['key']
        label = _strip_bracket_suffix(pm.get('label', key))
        unit  = pm.get('unit', '°')

        score  = param_scores.get(key)
        bucket = _tl_bucket(score)
        tc, _  = _tl_colors(bucket)

        directional = pm.get('directional_labels')
        if directional:
            if len(directional) == 3:
                pass_text, amber_text, fail_text = directional
            else:
                pass_text, fail_text = directional
                amber_text = fail_text
            actual_s = {'GREEN': pass_text, 'AMBER': amber_text}.get(bucket, fail_text)
            target_s = ''
        else:
            target_s = _fmt_target(pm)
            actual_s = _fmt_value(mean_values.get(key), unit, dp=0)
        row_bg = C['card_bg'] if idx % 2 == 0 else C['bg']
        c.setFillColor(row_bg)
        c.rect(table_x, row_y - ROW_H, TABLE_W, ROW_H, fill=1, stroke=0)
        c.setFillColor(tc)
        c.rect(table_x, row_y - ROW_H, COL_STRIP, ROW_H, fill=1, stroke=0)
        text_y = row_y - ROW_H + (ROW_H - 8) / 2
        c.setFillColor(C['txt'])
        c.setFont('Helvetica', 8)
        rx = table_x + COL_STRIP + COL_PAD
        c.drawString(rx, text_y, label); rx += COL_NAME
        c.setFont('Helvetica-Bold', 7 if directional else 8.5)
        c.drawString(rx, text_y, actual_s); rx += COL_ACT
        c.setFont('Helvetica', 8)
        c.setFillColor(C['txt_light'])
        c.drawString(rx, text_y, target_s); rx += COL_TGT
        bw = COL_BADGE - 0.15 * cm; bh = 0.48 * cm
        _score_badge(c, rx, row_y - ROW_H + (ROW_H - bh) / 2, bw, bh,
                     bucket, BADGE_LABELS[bucket])
        row_y -= ROW_H

    if table_h_used is None:
        n_rows      = len(event_metrics)
        table_h_tot = HDR_H + n_rows * ROW_H
        c.setStrokeColor(C['border']); c.setLineWidth(0.5)
        c.rect(table_x, table_top - table_h_tot, TABLE_W, table_h_tot, fill=0, stroke=1)
    else:
        table_h_tot = table_h_used

    _tips = LAYMAN_TIPS_BY_SPORT.get('rowing', {})
    if _tips and any(_tips.get(pm['key']) for pm in event_metrics):
        tip_box_y = MARGIN
        tip_avail = (table_top - table_h_tot) - 0.4 * cm - tip_box_y
        tip_box_h = min(tip_avail, _tip_box_height(c, TABLE_W, event_metrics, _tips))
        if tip_box_h > 1.2 * cm:
            tip_box_y = (table_top - table_h_tot) - 0.4 * cm - tip_box_h
            _draw_tip_box(c, table_x, tip_box_y, TABLE_W, tip_box_h,
                          event_metrics, param_scores, _tips)

    LABEL_H   = 0.45 * cm
    n_imgs    = 3
    cell_gap  = 0.25 * cm
    STRIP_H   = 3.5 * cm
    STRIP_GAP = 0.3 * cm
    frames_h  = avail_h - STRIP_H - STRIP_GAP
    cell_h    = (frames_h - cell_gap * (n_imgs - 1)) / n_imgs
    img_h     = cell_h - LABEL_H

    pct_labels = ['25%', '50%', '75%']

    # The strip shows one metric developing across the phase, so the overlay
    # metric is chosen per phase rather than per frame.  Falls back to the
    # first metric on the page if the phase is not in the map.
    draw_key = ROWING_PROGRESSION_DRAW_METRIC.get(event)
    if draw_key not in {p['key'] for p in event_metrics}:
        draw_key = event_metrics[0]['key'] if event_metrics else None

    draw_pm     = next((p for p in event_metrics if p['key'] == draw_key), None)
    draw_label  = _strip_bracket_suffix(draw_pm.get('label', draw_key)) if draw_pm else ev_label
    draw_bucket = _tl_bucket(param_scores.get(draw_key)) if draw_key else 'AMBER'
    strip_tc, strip_bg = _tl_colors(draw_bucket)

    for pi, (fi, pct) in enumerate(zip(prog_frames, pct_labels)):
        cell_y = top_y - (pi + 1) * cell_h - pi * cell_gap

        raw_bgr = _extract_frame_raw(video_path, fi) if video_path else None
        if raw_bgr is not None:
            cropped = _apply_crop_box(raw_bgr, crop_box) if crop_box else raw_bgr
            if draw_key and keypoints is not None:
                disp_h = min(cropped.shape[0], 1600)
                sf     = disp_h / max(cropped.shape[0], 1)
                disp_w = int(cropped.shape[1] * sf)
                disp   = cv2.resize(cropped, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                kp_d   = _kp_dict_at_frame(keypoints, fi, near_side, backend)
                if crop_box:
                    cx1, cy1 = crop_box[0], crop_box[1]
                    kp_crop = {k: (np.array([v[0]-cx1, v[1]-cy1]) if v is not None else None)
                               for k, v in kp_d.items()}
                else:
                    kp_crop = kp_d
                kp_sc  = _scale_kp_to_frame(kp_crop, cropped.shape[0], disp_h)
                disp   = _annotate_frame(disp, kp_sc, draw_key, draw_bucket, sport='rowing')
                if not crop_box:
                    disp = _smart_crop_to_person(disp, kp_sc)
                fit_bgr, fw, fh, ox, oy = _fit_to_box(disp, right_w, img_h)
            else:
                fit_bgr, fw, fh, ox, oy = _fit_to_box(cropped, right_w, img_h)
            ir = _frame_to_image_reader(fit_bgr)
            c.drawImage(ir, images_x + ox, cell_y + LABEL_H + oy,
                        width=fw, height=fh, mask='auto')
        else:
            c.setFillColor(C['border'])
            c.rect(images_x, cell_y + LABEL_H, right_w, img_h, fill=1, stroke=0)

        # Label strip and cell border carry the metric's traffic-light colour,
        # matching the per-metric image cells on the running pages.
        c.setFillColor(strip_bg)
        c.setStrokeColor(strip_tc)
        c.setLineWidth(1.2)
        c.roundRect(images_x, cell_y, right_w, LABEL_H, 4, fill=1, stroke=1)
        c.setFillColor(strip_tc)
        c.setFont('Helvetica-Bold', 8)
        c.drawCentredString(images_x + right_w / 2, cell_y + (LABEL_H - 8) / 2,
                            f'{draw_label} — {pct} of {ev_label.lower()}')

        c.setStrokeColor(strip_tc)
        c.setLineWidth(1.2)
        c.roundRect(images_x, cell_y, right_w, cell_h, 4, fill=0, stroke=1)

    _draw_sequencing_strip(c, images_x, MARGIN, right_w, STRIP_H, event, data)


def _build_cycling_progression_page(c: canvas.Canvas, event: str, data: dict,
                                     keypoints: np.ndarray, video_path: str,
                                     crop_box=None):
    """
    Cycling drive/recovery page: metric table left, 3 progression frames
    (25%, 50%, 75% between TDC and BDC) in a row (portrait frames).
    """
    _page_background(c)

    sport    = 'cycling'
    cfg      = SPORT_CONFIGS['cycling']
    ev_label = cfg['labels'].get(event, event.capitalize())
    _header_bar(c, f'Cycling Assessment — {ev_label}', '')

    HEADER_H  = 1.1 * cm
    top_y     = PAGE_H - HEADER_H - 0.5 * cm
    near_side = data.get('dominant_side') or data.get('near_side', 'right')
    backend   = data.get('backend', 'rtmpose').lower()
    key_frames   = data.get('key_frames', {})
    param_scores = data.get('param_scores', {})
    mean_values  = data.get('mean_values', {})
    params_meta  = data.get('params_meta', [])

    tdc_list = sorted([f for f in key_frames.get('tdc', []) if f is not None])
    bdc_list = sorted([f for f in key_frames.get('bdc', []) if f is not None])

    def _next_after(anchors, targets):
        """For each anchor return the nearest target strictly after it."""
        pairs = []
        used = set()
        for a in anchors:
            best = None
            for t in targets:
                if t > a and t not in used:
                    best = t
                    break
            if best is not None:
                pairs.append((a, best))
                used.add(best)
        return pairs

    if event == 'drive':
        pairs = _next_after(tdc_list, bdc_list)
    else:
        pairs = _next_after(bdc_list, tdc_list)

    if pairs:
        start_f, end_f = pairs[len(pairs) // 2]
    else:
        tdc_f = _median_frame(tdc_list) if tdc_list else 0
        bdc_f = _median_frame(bdc_list) if bdc_list else 0
        if event == 'drive':
            start_f = tdc_f
            later = [b for b in bdc_list if b > tdc_f]
            end_f = later[0] if later else tdc_f + 15
        else:
            start_f = bdc_f
            later = [t for t in tdc_list if t > bdc_f]
            end_f = later[0] if later else bdc_f + 15

    if end_f > start_f:
        prog_frames = [int(start_f + (end_f - start_f) * frac)
                       for frac in (0.25, 0.50, 0.75)]
    else:
        prog_frames = [start_f, start_f, end_f]

    event_metrics = [p for p in params_meta if p.get('event') == event]

    TABLE_W  = 8.2 * cm
    right_w  = PAGE_W - 2 * MARGIN - TABLE_W - 0.4 * cm
    images_x = MARGIN + TABLE_W + 0.4 * cm
    avail_h  = top_y - MARGIN

    ROW_H = 1.0 * cm; COL_STRIP = 0.18 * cm; COL_PAD = 0.25 * cm
    COL_NAME = 3.6 * cm; COL_ACT = 1.4 * cm; COL_TGT = 1.5 * cm
    COL_BADGE = 1.3 * cm; HDR_H = 0.6 * cm
    table_x = MARGIN; table_top = top_y

    c.setFillColor(C['header'])
    c.rect(table_x, table_top - HDR_H, TABLE_W, HDR_H, fill=1, stroke=0)
    c.setFillColor(C['white']); c.setFont('Helvetica-Bold', 8)
    hx = table_x + COL_STRIP + COL_PAD
    for hdr, cw in [('Metric', COL_NAME), ('Actual', COL_ACT),
                    ('Target', COL_TGT), ('', COL_BADGE)]:
        c.drawString(hx, table_top - HDR_H + 0.18 * cm, hdr); hx += cw

    row_y = table_top - HDR_H
    for idx, pm in enumerate(event_metrics):
        key = pm['key']; label = pm.get('label', key); unit = pm.get('unit', '°')
        target_s = _fmt_target(pm)
        actual_s = _fmt_value(mean_values.get(key), unit)
        score = param_scores.get(key); bucket = _tl_bucket(score)
        tc, _ = _tl_colors(bucket)
        row_bg = C['card_bg'] if idx % 2 == 0 else C['bg']
        c.setFillColor(row_bg)
        c.rect(table_x, row_y-ROW_H, TABLE_W, ROW_H, fill=1, stroke=0)
        c.setFillColor(tc)
        c.rect(table_x, row_y-ROW_H, COL_STRIP, ROW_H, fill=1, stroke=0)
        text_y = row_y - ROW_H + (ROW_H-8)/2
        c.setFillColor(C['txt']); c.setFont('Helvetica', 8)
        rx = table_x + COL_STRIP + COL_PAD
        c.drawString(rx, text_y, label); rx += COL_NAME
        c.setFont('Helvetica-Bold', 8.5); c.drawString(rx, text_y, actual_s); rx += COL_ACT
        c.setFont('Helvetica', 8); c.setFillColor(C['txt_light'])
        c.drawString(rx, text_y, target_s); rx += COL_TGT
        bw = COL_BADGE - 0.15*cm; bh = 0.48*cm
        _score_badge(c, rx, row_y-ROW_H+(ROW_H-bh)/2, bw, bh,
                     bucket, BADGE_LABELS[bucket])
        row_y -= ROW_H

    c.setStrokeColor(C['border']); c.setLineWidth(0.5)
    c.rect(table_x, table_top - HDR_H - len(event_metrics)*ROW_H,
           TABLE_W, HDR_H + len(event_metrics)*ROW_H, fill=0, stroke=1)

    LABEL_H  = 0.45 * cm
    n_imgs   = 3
    cell_gap = 0.25 * cm
    cell_w   = (right_w - cell_gap * (n_imgs - 1)) / n_imgs
    img_h    = avail_h - LABEL_H

    pct_labels = ['25%', '50%', '75%']
    for pi, (fi, pct) in enumerate(zip(prog_frames, pct_labels)):
        cell_x = images_x + pi * (cell_w + cell_gap)

        raw_bgr = _extract_frame_raw(video_path, fi) if video_path else None
        if raw_bgr is not None:
            orig_h  = raw_bgr.shape[0]
            cropped = _apply_crop_box(raw_bgr, crop_box) if crop_box else raw_bgr
            if event_metrics and keypoints is not None:
                key    = event_metrics[0]['key']
                bucket = _tl_bucket(param_scores.get(key))
                disp_h = min(cropped.shape[0], 1600)
                sf     = disp_h / max(cropped.shape[0], 1)
                disp_w = int(cropped.shape[1] * sf)
                disp   = cv2.resize(cropped, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                kp_d   = _kp_dict_at_frame(keypoints, fi, near_side, backend)
                if crop_box:
                    cx1, cy1 = crop_box[0], crop_box[1]
                    kp_crop = {k: (np.array([v[0]-cx1, v[1]-cy1]) if v is not None else None)
                               for k, v in kp_d.items()}
                else:
                    kp_crop = kp_d
                kp_sc = _scale_kp_to_frame(kp_crop, cropped.shape[0], disp_h)
                disp  = _annotate_frame(disp, kp_sc, key, bucket, sport='cycling')
                fit_bgr, fw, fh, ox, oy = _fit_to_box(disp, cell_w, img_h)
            else:
                fit_bgr, fw, fh, ox, oy = _fit_to_box(cropped, cell_w, img_h)
            ir = _frame_to_image_reader(fit_bgr)
            c.drawImage(ir, cell_x + ox, MARGIN + LABEL_H + oy,
                        width=fw, height=fh, mask='auto')
        else:
            c.setFillColor(C['border'])
            c.rect(cell_x, MARGIN + LABEL_H, cell_w, img_h, fill=1, stroke=0)

        c.setFillColor(C['header_lt'])
        c.roundRect(cell_x, MARGIN, cell_w, LABEL_H, 3, fill=1, stroke=0)
        c.setFillColor(C['header']); c.setFont('Helvetica-Bold', 8)
        c.drawCentredString(cell_x + cell_w / 2, MARGIN + (LABEL_H - 8) / 2,
                            f'{ev_label} — {pct}')


def _build_event_page(c: canvas.Canvas, event: str, data: dict,
                      keypoints: np.ndarray, video_path: str,
                      crop_box=None):
    sport = data.get('sport', 'running')

    if sport == 'rowing' and event in ('drive', 'recovery'):
        _build_rowing_progression_page(c, event, data, keypoints, video_path, crop_box)
        return

    if sport == 'cycling' and event in ('drive', 'recovery'):
        _build_cycling_progression_page(c, event, data, keypoints, video_path, crop_box)
        return

    _page_background(c)

    cfg      = SPORT_CONFIGS.get(sport, SPORT_CONFIGS['running'])
    ev_label = cfg['labels'].get(event, event.capitalize())
    _header_bar(c, f'{sport.capitalize()} Assessment — {ev_label}', '')

    HEADER_H = 1.1 * cm
    top_y    = PAGE_H - HEADER_H - 0.5 * cm

    frames_source = cfg['frames_source']
    frame_pool    = data.get(frames_source, {})
    param_scores  = data.get('param_scores', {})
    param_scores  = data.get('param_scores', {})
    mean_values   = data.get('mean_values', {})
    params_meta   = data.get('params_meta', [])
    near_side     = data.get('dominant_side') or data.get('near_side', 'right')
    backend       = data.get('backend', 'rtmpose').lower()

    frames_key    = cfg['frames_key'].get(event, event)
    frame_list    = frame_pool.get(frames_key, [])
    frame_list    = [f for f in frame_list if f is not None]
    med_frame     = _median_frame(frame_list) if frame_list else 0
    event_metrics = [p for p in params_meta
                     if p.get('event') == event
                     or (not p.get('event') and p.get('event_frames_key') == frames_key)]

    n_imgs = len(event_metrics)
    if n_imgs == 0:
        return

    TABLE_W  = 8.2 * cm
    right_w  = PAGE_W - 2 * MARGIN - TABLE_W - 0.4 * cm
    images_x = MARGIN + TABLE_W + 0.4 * cm
    avail_h  = top_y - MARGIN

    ROW_H     = 1.0 * cm
    COL_STRIP = 0.18 * cm
    COL_PAD   = 0.25 * cm
    COL_NAME  = 3.6 * cm
    COL_ACT   = 1.4 * cm
    COL_TGT   = 1.5 * cm
    COL_BADGE = 1.3 * cm
    HDR_H     = 0.6 * cm

    table_x   = MARGIN
    table_top = top_y

    c.setFillColor(C['header'])
    c.rect(table_x, table_top - HDR_H, TABLE_W, HDR_H, fill=1, stroke=0)
    c.setFillColor(C['white'])
    c.setFont('Helvetica-Bold', 8)
    hx = table_x + COL_STRIP + COL_PAD
    for hdr, cw in [('Metric', COL_NAME), ('Actual', COL_ACT),
                    ('Target', COL_TGT), ('', COL_BADGE)]:
        c.drawString(hx, table_top - HDR_H + 0.18 * cm, hdr)
        hx += cw

    row_y = table_top - HDR_H

    for idx, pm in enumerate(event_metrics):
        key   = pm['key']
        label = _strip_bracket_suffix(pm.get('label', key))
        unit  = pm.get('unit', '°')

        score  = param_scores.get(key)
        bucket = _tl_bucket(score)
        tc, bg = _tl_colors(bucket)

        directional = pm.get('directional_labels')
        if directional:
            if len(directional) == 3:
                pass_text, amber_text, fail_text = directional
            else:
                pass_text, fail_text = directional
                amber_text = fail_text
            actual_s = {'GREEN': pass_text, 'AMBER': amber_text}.get(bucket, fail_text)
            target_s = ''
        else:
            target_s = _fmt_target(pm)
            actual_s = _fmt_value(mean_values.get(key), unit,
                                  dp=VALUE_DP_BY_SPORT.get(sport, 1))

        row_bg = C['card_bg'] if idx % 2 == 0 else C['bg']
        c.setFillColor(row_bg)
        c.rect(table_x, row_y - ROW_H, TABLE_W, ROW_H, fill=1, stroke=0)

        c.setFillColor(tc)
        c.rect(table_x, row_y - ROW_H, COL_STRIP, ROW_H, fill=1, stroke=0)

        text_y = row_y - ROW_H + (ROW_H - 8) / 2

        c.setFillColor(C['txt'])
        c.setFont('Helvetica', 8)
        rx = table_x + COL_STRIP + COL_PAD
        c.drawString(rx, text_y, label)
        rx += COL_NAME

        c.setFont('Helvetica-Bold', 7 if directional else 8.5)
        c.setFillColor(C['txt'])
        c.drawString(rx, text_y, actual_s)
        rx += COL_ACT

        c.setFont('Helvetica', 8)
        c.setFillColor(C['txt_light'])
        c.drawString(rx, text_y, target_s)
        rx += COL_TGT

        bw = COL_BADGE - 0.15 * cm
        bh = 0.48 * cm
        _score_badge(c, rx, row_y - ROW_H + (ROW_H - bh) / 2,
                     bw, bh, bucket, BADGE_LABELS[bucket])

        row_y -= ROW_H

    n_rows = len(event_metrics)
    table_h_total = HDR_H + n_rows * ROW_H
    c.setStrokeColor(C['border'])
    c.setLineWidth(0.5)
    c.rect(table_x, table_top - table_h_total, TABLE_W, table_h_total,
           fill=0, stroke=1)

    _tips = LAYMAN_TIPS_BY_SPORT.get(sport, {})
    if _tips and any(_tips.get(pm['key']) for pm in event_metrics):
        tip_gap_below_table = 0.4 * cm
        tip_box_y = MARGIN
        tip_avail = (table_top - table_h_total) - tip_gap_below_table - tip_box_y
        tip_box_h = min(tip_avail, _tip_box_height(c, TABLE_W, event_metrics, _tips))
        if tip_box_h > 1.2 * cm:
            tip_box_y = (table_top - table_h_total) - tip_gap_below_table - tip_box_h
            _draw_tip_box(c, table_x, tip_box_y, TABLE_W, tip_box_h,
                          event_metrics, param_scores, _tips)

    LABEL_BOX_H = 0.5 * cm
    if sport == 'rowing' and event in ('catch', 'finish'):
        cols = 1
    else:
        cols = 2 if n_imgs > 2 else n_imgs
    rows     = math.ceil(n_imgs / cols)
    cell_gap = 0.3 * cm
    cell_w   = (right_w - cell_gap * (cols - 1)) / cols
    cell_h   = (avail_h - cell_gap * (rows - 1)) / rows
    img_h    = cell_h - LABEL_BOX_H - 0.1 * cm

    orig_h   = 0
    orig_bgr = None

    if video_path and os.path.isfile(video_path) and frame_list:
        candidates = [med_frame] + [f for f in frame_list if f != med_frame]
        for fi in candidates:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    orig_h   = frame.shape[0]
                    orig_bgr = frame
                    med_frame = fi
                    break
            else:
                cap.release()

        if orig_bgr is None:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                limit = min(int(med_frame) + 1, 2000)
                for i in range(limit):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    orig_bgr = frame
                cap.release()
                if orig_bgr is not None:
                    orig_h = orig_bgr.shape[0]

    for mi, pm in enumerate(event_metrics):
        row_i = mi // cols
        col_i = mi % cols

        cell_x = images_x + col_i * (cell_w + cell_gap)
        cell_y = top_y - (row_i + 1) * (cell_h + cell_gap) + cell_gap

        key    = pm['key']
        bucket = _tl_bucket(param_scores.get(key))
        tc, bc = _tl_colors(bucket)
        label  = _strip_bracket_suffix(pm.get('label', key))

        annotated = None
        if orig_bgr is not None and orig_h > 0 and keypoints is not None:
            display_h    = min(orig_h, 1600)
            scale_factor = display_h / orig_h
            disp_w       = int(orig_bgr.shape[1] * scale_factor)
            raw_bgr      = cv2.resize(orig_bgr, (disp_w, display_h),
                                      interpolation=cv2.INTER_AREA)
            kp_dict   = _kp_dict_at_frame(keypoints, med_frame, near_side, backend)
            kp_scaled = _scale_kp_to_frame(kp_dict, orig_h, display_h)
            annotated = _annotate_frame(raw_bgr, kp_scaled, key, bucket, sport=sport)
            if crop_box is not None:
                sf = display_h / orig_h
                cb_scaled = (int(crop_box[0]*sf), int(crop_box[1]*sf),
                             int(crop_box[2]*sf), int(crop_box[3]*sf))
                annotated = _apply_crop_box(annotated, cb_scaled)
            else:
                annotated = _smart_crop_to_person(annotated, kp_scaled)

        img_area_y = cell_y + LABEL_BOX_H

        if annotated is not None:
            fit_bgr, fw, fh, ox, oy = _fit_to_box(annotated, cell_w, img_h)
            ir = _frame_to_image_reader(fit_bgr)
            c.drawImage(ir,
                        cell_x + ox, img_area_y + oy,
                        width=fw, height=fh, mask='auto')
        else:
            c.setFillColor(C['border'])
            c.rect(cell_x, img_area_y, cell_w, img_h, fill=1, stroke=0)
            c.setFillColor(C['txt_light'])
            c.setFont('Helvetica', 8)
            c.drawCentredString(cell_x + cell_w / 2,
                                img_area_y + img_h / 2, 'No video')

        c.setFillColor(bc)
        c.setStrokeColor(tc)
        c.setLineWidth(1.2)
        c.roundRect(cell_x, cell_y, cell_w, LABEL_BOX_H, 4, fill=1, stroke=1)
        c.setFillColor(tc)
        c.setFont('Helvetica-Bold', 8)
        c.drawCentredString(cell_x + cell_w / 2,
                            cell_y + (LABEL_BOX_H - 8) / 2, label)

        c.setStrokeColor(tc)
        c.setLineWidth(1.2)
        c.roundRect(cell_x, cell_y, cell_w, cell_h, 4, fill=0, stroke=1)


def generate_report(data: dict,
                    keypoints: np.ndarray,
                    output_path: str,
                    athlete_name: Optional[str] = None) -> str:
    """
    Generate a landscape PDF biomechanical assessment report.

    Parameters
    ----------
    data         : dict        — RunningScorer.to_dict() output
    keypoints    : np.ndarray  — raw keypoint array shape (n_frames, n_kp, 2+)
    output_path  : str         — destination .pdf file path
    athlete_name : str | None  — printed in the header (optional)

    Returns
    -------
    output_path  : str  — path to the generated PDF
    """
    video_path = data.get('video_path') or ''
    if video_path and '_annotated' in os.path.basename(video_path).lower():
        video_path = ''
    sport = data.get('sport', 'running')
    cfg   = SPORT_CONFIGS.get(sport, SPORT_CONFIGS['running'])

    crop_box = None
    if keypoints is not None and video_path:
        cap = cv2.VideoCapture(video_path)
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        pad_x = 0.30 if sport == 'cycling' else 0.18
        crop_box = _compute_video_crop_box(
            keypoints, pad_x_frac=pad_x, pad_y_frac=0.12,
            frame_w=fw, frame_h=fh)

    c = canvas.Canvas(output_path, pagesize=landscape(A4))
    c.setTitle(f'{sport.capitalize()} Biomechanical Assessment Report')
    c.setAuthor('Automated Movement Assessment System')

    if sport == 'rowing':
        _build_rowing_overview_page(c, data, keypoints, video_path, crop_box)
    else:
        _build_overview_page(c, data, keypoints, video_path, crop_box)
    c.showPage()

    for event in cfg['events']:
        _build_event_page(c, event, data, keypoints, video_path, crop_box=crop_box)
        c.showPage()

    if sport == 'rowing':
        _build_rowing_howto_page(c, data)
        c.showPage()

    c.save()
    return output_path


if __name__ == '__main__':
    _mock_params_meta = [
        {'key': 'thigh_sep_mvp',     'label': 'Thigh Sep. (MVP)',    'unit': '°',
         'optimal': [80, 100],  'event_frames_key': 'mvp_frames'},
        {'key': 'trailing_shin_mvp', 'label': 'Trailing Shin (MVP)', 'unit': 'score',
         'optimal': [0.5, 1.0], 'event_frames_key': 'mvp_frames'},
        {'key': 'leading_knee_mvp',  'label': 'Leading Knee (MVP)',  'unit': '°',
         'optimal': [110, 180], 'event_frames_key': 'mvp_frames'},
        {'key': 'trailing_foot_to',  'label': 'Trailing Foot (TO)',  'unit': '°',
         'optimal': [-10, 10],  'event_frames_key': 'to_frames'},
        {'key': 'thigh_sep_to',      'label': 'Thigh Sep. (TO)',     'unit': '°',
         'optimal': [80, 100],  'event_frames_key': 'to_frames'},
        {'key': 'leading_ankle_to',  'label': 'Leading Ankle (TO)',  'unit': '°',
         'optimal': [80, 100],  'event_frames_key': 'to_frames'},
        {'key': 'trailing_knee_td',  'label': 'Trailing Knee (TD)',  'unit': '°',
         'optimal': [0, 45],    'event_frames_key': 'td_frames'},
        {'key': 'landing_foot_td',   'label': 'Landing Foot (TD)',   'unit': '°',
         'optimal': [-20, 20],  'event_frames_key': 'td_frames'},
        {'key': 'shin_ext_td',       'label': 'Shin Extension (TD)', 'unit': '°',
         'optimal': [-45, 0],   'event_frames_key': 'td_frames'},
        {'key': 'torso_tilt_td',     'label': 'Torso Tilt (TD)',     'unit': '°',
         'optimal': [0, 15],    'event_frames_key': 'td_frames'},
        {'key': 'thigh_gap_ms',      'label': 'Thigh Gap (MS)',      'unit': '°',
         'optimal': [0, 20],    'event_frames_key': 'ms_frames'},
        {'key': 'sinking_ms',        'label': 'Sinking (MS)',        'unit': '°',
         'optimal': [-45, 0],   'event_frames_key': 'ms_frames'},
    ]

    _mock_data = {
        'sport': 'running',
        'backend': 'RTMPOSE',
        'dominant_side': 'left',
        'fps': 30,
        'n_frames': 900,
        'n_cycles': 14,
        'total_score': 42.0,
        'param_scores': {
            'thigh_sep_mvp': 100, 'trailing_shin_mvp': 50,
            'leading_knee_mvp': 0, 'trailing_foot_to': 100,
            'thigh_sep_to': 50, 'leading_ankle_to': 0,
            'trailing_knee_td': 0, 'landing_foot_td': 100,
            'shin_ext_td': 0, 'torso_tilt_td': 50,
            'thigh_gap_ms': 100, 'sinking_ms': 0,
        },
        'mean_values': {
            'thigh_sep_mvp': 91.2, 'trailing_shin_mvp': 0.8,
            'leading_knee_mvp': 95.3, 'trailing_foot_to': 4.1,
            'thigh_sep_to': 84.7, 'leading_ankle_to': 68.2,
            'trailing_knee_td': -12.4, 'landing_foot_td': 3.5,
            'shin_ext_td': 11.9, 'torso_tilt_td': 7.2,
            'thigh_gap_ms': 15.3, 'sinking_ms': 11.2,
        },
        'params_meta': _mock_params_meta,
        'raw_traces': {
            'td_frames':  [30, 90, 150, 210, 270],
            'ms_frames':  [45, 105, 165, 225, 285],
            'to_frames':  [60, 120, 180, 240, 300],
            'mvp_frames': [75, 135, 195, 255, 315],
        },
        'video_path': '',
        'warnings': [],
    }

    out = generate_report(_mock_data, None, '/tmp/test_report.pdf')
    print(f'Report written to {out}')
