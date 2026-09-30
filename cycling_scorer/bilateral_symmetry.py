"""
Parameter 6 - bilateral symmetry index: |(near_mean - far_mean) / avg| * 100 for knee angle, each leg sampled at its OWN BDC.

The crank arms are 180 deg out of phase, so comparing both knees in the same
frame measures crank phase offset rather than asymmetry. The far-side BDC is
therefore located independently within each cycle as the frame of maximum
far-ankle Y, not assumed to be half a cycle later.

Validation caveat: this is the only cycling parameter requiring far-side
(occluded-limb) landmarks, which cannot be validated against digitised markers
from a single sagittal camera. Frames below MIN_FAR_CONF are excluded and the
returned n_valid / n_total let the caller report coverage rather than an
unqualified figure.

Optimal: below 10% asymmetry (Bini et al. 2008).
"""

import numpy as np

from scorer_utils import angle_at_joint

MIN_FAR_CONF = 0.30

MIN_VALID_CYCLES = 3


def _knee_angle(frame, kp_map, side_prefix):
    """Included knee angle for one side in one frame, or None if unusable."""
    hip = kp_map[f'{side_prefix}_hip']
    knee = kp_map[f'{side_prefix}_knee']
    ank = kp_map[f'{side_prefix}_ankle']
    ang = angle_at_joint(frame[hip, :2], frame[knee, :2], frame[ank, :2])
    if ang is None or not np.isfinite(ang):
        return None
    return float(ang)


def _min_conf(frame, kp_map, side_prefix):
    """Lowest confidence across the three joints defining the knee angle."""
    if frame.shape[1] < 3:
        return 1.0
    idxs = [kp_map[f'{side_prefix}_hip'],
            kp_map[f'{side_prefix}_knee'],
            kp_map[f'{side_prefix}_ankle']]
    return float(np.min([frame[i, 2] for i in idxs]))


def compute_bsi(kp_seq, kp_map, cycles, bdc_frames, near_side):
    """
    Compute global BSI with each leg sampled at its own BDC.

    Parameters
    ----------
    kp_seq     : np.ndarray  shape (n_frames, n_kp, 3)
    kp_map     : dict        keypoint name → index map
    cycles     : list[(start, end)]  cycle windows (near-side BDC to BDC)
    bdc_frames : sequence[int]       near-side BDC frame indices
    near_side  : str         'left' or 'right'

    Returns
    -------
    (bsi, n_valid, n_total)
        bsi     : float | None  asymmetry as a percentage; None if fewer than
                                MIN_VALID_CYCLES cycles yielded a usable pair.
        n_valid : int           cycles contributing to the figure
        n_total : int           cycles attempted
    """
    near = 'r' if near_side == 'right' else 'l'
    far = 'l' if near == 'r' else 'r'
    far_ank = kp_map[f'{far}_ankle']

    n_total = len(cycles)
    near_vals, far_vals = [], []

    bdc_sorted = sorted(int(b) for b in bdc_frames if b is not None)

    for (start, end) in cycles:
        start, end = int(start), int(end)
        if end <= start or end > len(kp_seq):
            continue

        in_cycle = [b for b in bdc_sorted if start <= b < end]
        near_f = in_cycle[0] if in_cycle else start

        far_y = kp_seq[start:end, far_ank, 1]
        if not np.any(np.isfinite(far_y)):
            continue
        far_f = start + int(np.nanargmax(far_y))

        if _min_conf(kp_seq[far_f], kp_map, far) < MIN_FAR_CONF:
            continue
        if _min_conf(kp_seq[near_f], kp_map, near) < MIN_FAR_CONF:
            continue

        near_ang = _knee_angle(kp_seq[near_f], kp_map, near)
        far_ang = _knee_angle(kp_seq[far_f], kp_map, far)
        if near_ang is None or far_ang is None:
            continue

        near_vals.append(near_ang)
        far_vals.append(far_ang)

    n_valid = len(near_vals)
    if n_valid < MIN_VALID_CYCLES:
        return None, n_valid, n_total

    near_mean = float(np.mean(near_vals))
    far_mean = float(np.mean(far_vals))
    avg = (near_mean + far_mean) / 2.0
    if abs(avg) < 1e-9:
        return None, n_valid, n_total

    return abs(near_mean - far_mean) / avg * 100.0, n_valid, n_total
