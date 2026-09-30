"""
rowing_scorer/coordination.py
=============================
Drive and recovery sequencing, from segment contribution profiles.

Primitive
---------
Over any phase, handle travel relative to the footplate decomposes exactly
into three segment contributions:

    (wrist - ankle)  =  (hip - ankle)  +  (shoulder - hip)  +  (wrist - shoulder)
     handle travel        legs (seat)        trunk swing         arm draw

Each segment's frame-to-frame increment in the direction the handle is
travelling is the work that segment did at that instant.  The cumulative
profile of those increments gives, on one common 0-100% phase scale:

    q25     when the segment had done a quarter of its work   (onset)
    cent    contribution-weighted centroid                    (when it worked)
    q75     when it had done three quarters                   (completion)
    share   its fraction of total handle travel               (how much)

This replaces the previous threshold-crossing onset detector, which
normalised each segment by its own excursion and therefore compared three
timestamps that were never on the same scale.

Scoring
-------
A phase has two handovers.  Each is scored by the gap between consecutive
centroids, in percent of phase, so the measure is invariant to stroke rate:

    drive     leg -> trunk -> arm
    recovery  arm -> trunk -> leg

The phase metric is the SMALLER of the two gaps: a stroke is only as good
as its weakest handover, and reporting the minimum names which handover
failed.  Negative means that handover is out of order.  A gap smaller than
the timing resolution of the capture means the two segments are
indistinguishable in time, which reads as a simultaneous, unsequenced
movement.

Validation (24 clips, 6 rowers, 95 strokes, Sept 2026)
------------------------------------------------------
The one experienced rower in the capture set ranks first of six on recovery
and second on drive, with every clip above the resolution floor on both
phases, and his centroids fall in the positions the technical model
predicts without the method being tuned to produce that.  Stroke-to-stroke
scatter within a clip has a median SD below 2% of phase against a
between-clip spread of ~30 points, so the per-clip mean is the stable
quantity.  Thresholds below are motivated by that set, not validated on it:
one expert is not a population.

Angle-based completion method (default since 18 Sept 2026)
----------------------------------------------------------
Agreed with Ciarán: each handover is judged by how far the previous segment
is through its own angle change when the next segment starts moving.  Knee
angle stands for the legs, signed trunk angle for the trunk, elbow angle for
the arms.  Progress is the cumulative angle change in the direction of the
segment's net change over the phase, rectified, so a brief reversal does not
count as un-doing the movement.

    next segment "starts"   when it reaches SEQ_ONSET_FRAC of its change
    GREEN                   previous >= SEQ_GREEN_FRAC complete   (in sequence)
    AMBER                   in between                            (overlap)
    RED                     previous <  SEQ_RED_FRAC complete     (out of sequence)

The phase figure is the smaller of the two handovers, as before.  The
centroid profiles are still computed for the timeline strip and the
dashboard; set SEQ_METHOD = 'centroid' in params.py to score with them.

Source: Kleshnev 2010, 2016 (segment contribution decomposition);
        British Rowing Technical Model (sequence).
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from .params import (SEQ_METHOD, SEQ_ONSET_FRAC, SEQ_GREEN_FRAC, SEQ_RED_FRAC,
                     SEQ_LAST_QUARTER, SEQ_ANGLE_SMOOTH)

# Joint angle that stands for each segment in the angle-based method.
SEGMENT_ANGLE_NAME = {'leg': 'knee', 'trunk': 'trunk', 'arm': 'elbow'}

# Combined catch/finish timing uncertainty, in frames.  The floor below is
# derived from this, so it is the single number to revisit once rowing event
# detection has been validated against digitised ground truth the way running
# TD/TO was.  Provisional: the two phase boundaries are each taken as ~1 frame,
# combined in quadrature and rounded up.
EVENT_UNCERTAINTY_FRAMES = 1.5

# Ordering of segments for each phase.
PHASE_ORDER = {
    'drive':    ('leg', 'trunk', 'arm'),
    'recovery': ('arm', 'trunk', 'leg'),
}

_QUANTILES = (0.25, 0.75)


def _gap_label(gap: float, floor: float) -> str:
    """Centroid-method colour for one handover: out of order / unresolved / clean."""
    if gap < 0:
        return 'RED'
    if gap < floor:
        return 'AMBER'
    return 'GREEN'


def sequencing_floor(phase_frames: int) -> float:
    """
    Smallest centroid gap this capture can resolve, in percent of phase.

    Expressed per clip rather than as a fixed percentage because a fixed
    percentage is a different number of frames at every stroke rate: a fast
    drive of 13 frames cannot resolve what a slow one of 46 frames can.
    """
    n = max(int(phase_frames), 1)
    return float(EVENT_UNCERTAINTY_FRAMES / n * 100.0)


def _segment_signals(seg, near_side):
    """Horizontal segment extents whose sum is handle travel from the footplate."""
    from scorer_utils import _jxy
    pre = 'l' if near_side == 'left' else 'r'
    hip = _jxy(seg, f'{pre}_hip')
    ank = _jxy(seg, f'{pre}_ankle')
    sho = _jxy(seg, f'{pre}_shoulder')
    wri = _jxy(seg, f'{pre}_wrist')
    return (
        {
            'leg':   hip[:, 0] - ank[:, 0],
            'trunk': sho[:, 0] - hip[:, 0],
            'arm':   wri[:, 0] - sho[:, 0],
        },
        wri[:, 0] - ank[:, 0],          # handle
    )


def _profile(signal, direction):
    """
    Cumulative contribution profile of one segment over one phase.

    Increments are taken in the direction the handle travels and rectified:
    a segment briefly moving against the stroke contributes no work rather
    than negative work, which would make a weighted centroid meaningless.
    Returns None fields when the segment did not move at all.
    """
    d = np.clip(np.diff(np.asarray(signal, float)) * direction, 0.0, None)
    total = float(d.sum())
    n = len(d)
    if n == 0 or total < 1e-9:
        return {'q25': None, 'cent': None, 'q75': None, 'total': 0.0}

    t   = (np.arange(n) + 0.5) / n * 100.0
    cum = np.cumsum(d) / total
    q25, q75 = (float(np.interp(q, cum, t)) for q in _QUANTILES)
    return {
        'q25':   q25,
        'cent':  float((t * d).sum() / total),
        'q75':   q75,
        'total': total,
    }


def compute_phase_sequencing(seg, start_local, end_local, near_side, phase):
    """
    Sequencing of one phase of one stroke.

    Parameters
    ----------
    seg         : list[dict]  normalised keypoints for the whole cycle
    start_local : int         first frame of the phase, local to the cycle
    end_local   : int         last frame of the phase, local to the cycle
    near_side   : 'left' | 'right'
    phase       : 'drive' | 'recovery'

    Returns
    -------
    dict or None (phase too short to profile), with:
      metric       smaller of the two centroid gaps, % of phase
      gaps         (first handover, second handover), % of phase
      weakest      name of the handover the metric came from, e.g. 'trunk->arm'
      floor        smallest gap this clip can resolve, % of phase
      label        'GREEN' | 'AMBER' | 'RED'
      profiles     per segment: q25, cent, q75, share
      n_frames     phase length in frames
    """
    order = PHASE_ORDER[phase]
    n_fr  = int(end_local) - int(start_local)
    if n_fr < 5:
        return None

    signals, handle = _segment_signals(seg, near_side)
    sl = slice(int(start_local), int(end_local) + 1)

    direction = float(np.sign(handle[sl][-1] - handle[sl][0]))
    if direction == 0.0:
        direction = 1.0

    profiles = {name: _profile(signals[name][sl], direction) for name in order}
    if any(p['cent'] is None for p in profiles.values()):
        return None

    grand = sum(p['total'] for p in profiles.values())
    for p in profiles.values():
        p['share'] = float(p['total'] / grand * 100.0) if grand > 1e-9 else None
        p.pop('total')

    gaps = tuple(profiles[b]['cent'] - profiles[a]['cent']
                 for a, b in zip(order, order[1:]))
    names  = tuple(f'{a}->{b}' for a, b in zip(order, order[1:]))
    worst  = int(np.argmin(gaps))
    metric = float(gaps[worst])
    floor  = sequencing_floor(n_fr)

    if metric < 0:
        label = 'RED'
    elif metric < floor:
        label = 'AMBER'
    else:
        label = 'GREEN'

    out = {
        'method':   'centroid',
        'metric':   round(metric, 2),
        'gaps':     tuple(round(g, 2) for g in gaps),
        'weakest':  names[worst],
        'floor':    round(floor, 2),
        'label':    label,
        'profiles': {k: {kk: (round(vv, 2) if vv is not None else None)
                         for kk, vv in v.items()}
                     for k, v in profiles.items()},
        'n_frames': n_fr,
        'pass_lo':  round(floor, 2),
        'handovers': [
            {'from': a, 'to': b, 'name': nm, 'gap': round(g, 2),
             'label': _gap_label(g, floor)}
            for (a, b), nm, g in zip(zip(order, order[1:]), names, gaps)
        ],
    }

    if SEQ_METHOD == 'completion':
        comp = _completion_sequencing(seg, int(start_local), int(end_local),
                                      near_side, order)
        if comp is None:
            return None
        out.update(comp)
    return out


# ---------------------------------------------------------------------------
# Angle-based completion method
# ---------------------------------------------------------------------------

def _segment_angles(seg, near_side):
    """Knee, signed trunk and elbow angle series over the whole cycle, smoothed."""
    from scorer_utils import _jxy, angle_at_joint, signed_trunk_angle
    pre = 'l' if near_side == 'left' else 'r'
    hip = _jxy(seg, f'{pre}_hip')
    kne = _jxy(seg, f'{pre}_knee')
    ank = _jxy(seg, f'{pre}_ankle')
    sho = _jxy(seg, f'{pre}_shoulder')
    elb = _jxy(seg, f'{pre}_elbow')
    wri = _jxy(seg, f'{pre}_wrist')
    n = len(seg)
    raw = {
        'leg':   np.array([angle_at_joint(hip[i], kne[i], ank[i]) for i in range(n)]),
        'trunk': np.array([signed_trunk_angle(hip[i], sho[i]) for i in range(n)]),
        'arm':   np.array([angle_at_joint(sho[i], elb[i], wri[i]) for i in range(n)]),
    }
    w = max(int(SEQ_ANGLE_SMOOTH), 1)
    return {k: (uniform_filter1d(v, size=w, mode='nearest') if w > 1 else v)
            for k, v in raw.items()}


def _angle_progress(angle):
    """
    Fraction of the net angle change completed at each frame of the phase.

    Increments are taken in the direction of the net change and rectified, so
    the curve is non-decreasing from 0 at the first frame to 1 at the last.
    Returns None when the segment did not move.
    """
    a = np.asarray(angle, float)
    direction = float(np.sign(a[-1] - a[0])) or 1.0
    d = np.clip(np.diff(a) * direction, 0.0, None)
    total = float(d.sum())
    if len(d) == 0 or total < 1e-6:
        return None
    return np.concatenate([[0.0], np.cumsum(d) / total])


def _time_at(progress, frac):
    """Fractional frame index at which progress first reaches frac."""
    idx = int(np.searchsorted(progress, frac, side='left'))
    if idx <= 0:
        return 0.0
    if idx >= len(progress):
        return float(len(progress) - 1)
    p0, p1 = progress[idx - 1], progress[idx]
    return float(idx - 1 + (frac - p0) / max(p1 - p0, 1e-12))


def _completion_label(frac):
    if frac >= SEQ_GREEN_FRAC:
        return 'GREEN'
    if frac < SEQ_RED_FRAC:
        return 'RED'
    return 'AMBER'


def _completion_sequencing(seg, start_local, end_local, near_side, order):
    """
    Angle-based handover scoring for one phase of one stroke.

    Returns a dict of fields that override / extend the centroid record, or
    None if any segment's angle did not change over the phase.
    """
    angles = _segment_angles(seg, near_side)
    sl     = slice(start_local, end_local + 1)
    n_fr   = end_local - start_local

    prog, ang = {}, {}
    for name in order:
        a = angles[name][sl]
        p = _angle_progress(a)
        if p is None:
            return None
        prog[name] = p
        ang[name] = {
            'start':  float(a[0]),
            'end':    float(a[-1]),
            'change': float(a[-1] - a[0]),
            # timing on the 0-100% phase scale, for the report strip
            'onset':  _time_at(p, SEQ_ONSET_FRAC) / n_fr * 100.0,
            'done':   _time_at(p, SEQ_GREEN_FRAC) / n_fr * 100.0,
        }

    handovers = []
    for a, b in zip(order, order[1:]):
        t_on = _time_at(prog[b], SEQ_ONSET_FRAC)
        comp = float(np.interp(t_on, np.arange(len(prog[a])), prog[a]))
        handovers.append({
            'from':       a,
            'to':         b,
            'name':       f'{a}->{b}',
            'completion': round(comp * 100.0, 1),
            'label':      _completion_label(comp),
        })

    worst  = min(range(len(handovers)), key=lambda i: handovers[i]['completion'])
    metric = handovers[worst]['completion']
    first  = handovers[0]

    return {
        'method':       'completion',
        'metric':       round(metric, 1),
        'weakest':      handovers[worst]['name'],
        'label':        _completion_label(metric / 100.0),
        'handovers':    handovers,
        'last_quarter': bool(first['completion'] >= SEQ_LAST_QUARTER * 100.0),
        'angles':       {k: {kk: round(vv, 1) for kk, vv in v.items()}
                         for k, v in ang.items()},
        'pass_lo':      round(SEQ_GREEN_FRAC * 100.0, 1),
        'red_lo':       round(SEQ_RED_FRAC * 100.0, 1),
    }


def compute_drive_sequencing(seg, catch_local, finish_local, near_side):
    return compute_phase_sequencing(seg, catch_local, finish_local,
                                    near_side, 'drive')


def compute_recovery_sequencing(seg, finish_local, near_side):
    return compute_phase_sequencing(seg, finish_local, len(seg) - 1,
                                    near_side, 'recovery')


def aggregate_sequencing(records):
    """
    Combine the per-stroke results of one phase into the clip-level figure.

    The per-clip mean is the quantity to report: within a clip the measure is
    repeatable to well under the resolution floor, while between clips it
    spans tens of percent.  When the strokes of a clip disagree by more than
    the clip can resolve, the verdict is not trustworthy however clean the
    mean looks, so it is flagged rather than coloured confidently — that is
    usually a sign the catch or finish was detected inconsistently, not that
    the rower changed technique mid-piece.

    Returns None for an empty list.
    """
    recs = [r for r in records if r]
    if not recs:
        return None

    vals  = np.array([r['metric'] for r in recs], dtype=float)
    mean  = float(vals.mean())
    sd    = float(vals.std())
    floor = float(np.mean([r['floor'] for r in recs]))

    if mean < 0:
        label = 'RED'
    elif mean < floor:
        label = 'AMBER'
    else:
        label = 'GREEN'

    names   = [r['weakest'] for r in recs]
    weakest = max(set(names), key=names.count)

    if recs[0].get('method') == 'completion':
        return _aggregate_completion(recs, vals, mean, sd, floor, weakest)

    prof = {}
    for seg in recs[0]['profiles']:
        for field in ('q25', 'cent', 'q75', 'share'):
            xs = [r['profiles'][seg][field] for r in recs
                  if r['profiles'][seg].get(field) is not None]
            prof.setdefault(seg, {})[field] = (round(float(np.mean(xs)), 2)
                                               if xs else None)

    return {
        'metric':     round(mean, 2),
        'sd':         round(sd, 2),
        'floor':      round(floor, 2),
        'label':      label,
        'weakest':    weakest,
        'unstable':   bool(sd > floor),
        'n_strokes':  len(recs),
        'profiles':   prof,
        'method':     'centroid',
        'pass_lo':    round(floor, 2),
        'handovers':  _mean_gap_handovers(recs, floor),
    }


def _mean_gap_handovers(recs, floor):
    """Clip-level gap and colour for each handover (centroid method)."""
    first = recs[0].get('handovers') or []
    out = []
    for i, h0 in enumerate(first):
        g = float(np.mean([r['handovers'][i]['gap'] for r in recs]))
        out.append({'from': h0['from'], 'to': h0['to'], 'name': h0['name'],
                    'gap': round(g, 2), 'label': _gap_label(g, floor)})
    return out


def _mean_profiles(recs):
    prof = {}
    for seg in recs[0]['profiles']:
        for field in ('q25', 'cent', 'q75', 'share'):
            xs = [r['profiles'][seg][field] for r in recs
                  if r['profiles'][seg].get(field) is not None]
            prof.setdefault(seg, {})[field] = (round(float(np.mean(xs)), 2)
                                               if xs else None)
    return prof


def _aggregate_completion(recs, vals, mean, sd, floor, weakest):
    """
    Clip-level figure for the angle-based method.

    The clip label comes from the mean of the per-stroke minima.  A clip is
    flagged unstable when more than a third of its strokes disagree with that
    label, which on this method usually means the catch or finish was placed
    inconsistently rather than that technique changed mid-piece.
    """
    label = _completion_label(mean / 100.0)
    disagree = sum(r['label'] != label for r in recs) / len(recs)

    order = [h['from'] for h in recs[0]['handovers']] + [recs[0]['handovers'][-1]['to']]
    angles = {}
    for segname in order:
        angles[segname] = {
            f: round(float(np.mean([r['angles'][segname][f] for r in recs])), 1)
            for f in ('start', 'end', 'change', 'onset', 'done')
        }

    handovers = []
    for i, h0 in enumerate(recs[0]['handovers']):
        m = float(np.mean([r['handovers'][i]['completion'] for r in recs]))
        handovers.append({
            'from': h0['from'], 'to': h0['to'], 'name': h0['name'],
            'completion': round(m, 1),
            'label': _completion_label(m / 100.0),
        })

    return {
        'method':       'completion',
        'metric':       round(mean, 1),
        'sd':           round(sd, 1),
        'floor':        round(floor, 2),
        'label':        label,
        'weakest':      weakest,
        'unstable':     bool(disagree > 1 / 3),
        'n_strokes':    len(recs),
        'profiles':     _mean_profiles(recs),
        'handovers':    handovers,
        'last_quarter': bool(handovers[0]['completion'] >= SEQ_LAST_QUARTER * 100.0),
        'angles':       angles,
        'pass_lo':      round(SEQ_GREEN_FRAC * 100.0, 1),
        'red_lo':       round(SEQ_RED_FRAC * 100.0, 1),
    }
