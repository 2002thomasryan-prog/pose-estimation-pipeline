"""
Parameter 3 - trailing limb angle from vertical at toe-off (hip-to-ankle segment).

Optimal: 30-60 deg from vertical.
"""

import numpy as np
from scorer_utils import angle_from_vertical


def compute_trailing_limb_angle(seg, kp_map, side='right', to_local=None):
    """
    Parameters
    ----------
    seg      : np.ndarray  shape (n, n_kp, 3)  keypoints for one gait cycle
    kp_map   : dict        keypoint name → index map
    side     : str         'right' | 'left'
    to_local : int | None  LOCAL frame index of toe-off within seg.
               If None, falls back to the mid-swing heuristic used previously
               (mid_swing // 2) so the function degrades gracefully when event
               detection could not find a valid toe-off frame.

    Returns
    -------
    float  Trailing limb angle from vertical in degrees.
    """
    prefix    = 'r' if side == 'right' else 'l'
    hip_idx   = kp_map[f'{prefix}_hip']
    ankle_idx = kp_map[f'{prefix}_ankle']

    n = len(seg)

    if to_local is not None:
        toe_off_loc = int(np.clip(to_local, 0, n - 1))
    else:
        ankle_y    = seg[:, ankle_idx, 1]
        search_end = max(1, int(0.70 * n))
        mid_swing  = int(np.argmin(ankle_y[:search_end]))
        toe_off_loc = max(0, mid_swing // 2)

    f   = seg[toe_off_loc]
    tla = angle_from_vertical(f[hip_idx, :2], f[ankle_idx, :2])

    return float(tla)
