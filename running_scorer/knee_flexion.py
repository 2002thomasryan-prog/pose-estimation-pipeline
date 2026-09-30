"""
Parameter 4 - knee flexion at mid-stance, averaged over a +/-3 frame window.

Angle convention: 0 deg = fully extended, positive = flexed.
Optimal: 35-55 deg (Souza 2016; Dierks et al. 2011; Wille et al. 2014).
"""

import numpy as np
from scorer_utils import angle_at_joint

_MS_HALF_WINDOW = 3


def compute_knee_flexion_midstance(seg, kp_map, side='right', ms_local=None):
    """
    Parameters
    ----------
    seg      : np.ndarray  shape (n, n_kp, 3)  keypoints for one gait cycle
    kp_map   : dict        keypoint name → index map
    side     : str         'right' | 'left'  dominant side
    ms_local : int | None  LOCAL frame index of mid-stance within seg.
               If None, falls back to the fixed 10–20% cycle window used
               previously.

    Returns
    -------
    float  Mean knee flexion angle (degrees) across the mid-stance window.
           0° = straight, positive = flexed.
    """
    prefix  = 'r' if side == 'right' else 'l'
    hip_i   = kp_map[f'{prefix}_hip']
    knee_i  = kp_map[f'{prefix}_knee']
    ankle_i = kp_map[f'{prefix}_ankle']

    n = len(seg)

    if ms_local is not None:
        ms   = int(np.clip(ms_local, 0, n - 1))
        i_lo = max(0, ms - _MS_HALF_WINDOW)
        i_hi = min(n, ms + _MS_HALF_WINDOW + 1)
    else:
        i_lo = max(0, int(np.floor(0.10 * n)))
        i_hi = min(n, int(np.ceil( 0.20 * n)))
        if i_hi <= i_lo:
            i_lo, i_hi = 0, n

    flexion_angles = []
    for i in range(i_lo, i_hi):
        hip   = seg[i, hip_i,   :2]
        knee  = seg[i, knee_i,  :2]
        ankle = seg[i, ankle_i, :2]
        included = angle_at_joint(hip, knee, ankle)
        flexion  = 180.0 - included
        flexion_angles.append(flexion)

    if not flexion_angles:
        return float('nan')

    return float(np.mean(flexion_angles))
