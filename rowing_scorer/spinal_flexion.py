"""
Parameter 5 - spinal flexion, as the min/max shoulder-to-hip distance ratio across the cycle.

Near-side keypoints only, avoiding planar projection error from contralateral
joints. 1.0 is a perfectly stable trunk length; the ratio is scale-free but
cannot detect a rower who is rounded throughout the stroke.
Optimal: 0.88-1.0 (Nugent et al. 2021; Caldwell et al. 2003).
"""

import numpy as np
from scorer_utils import _jxy, segment_length


def compute_spinal_flexion(seg, near_side):
    """
    Compute shoulder-hip distance stability ratio across one stroke cycle.

    Parameters
    ----------
    seg       : list[dict]  keypoints for one cycle (normalised)
    near_side : str         'left' | 'right'

    Returns
    -------
    float  Ratio in [0, 1].  1.0 = no rounding detected.
    """
    prefix = 'l' if near_side == 'left' else 'r'
    sh_xy  = _jxy(seg, f'{prefix}_shoulder')
    hip_xy = _jxy(seg, f'{prefix}_hip')

    n     = len(seg)
    dists = np.array([
        segment_length(sh_xy[i], hip_xy[i])
        for i in range(n)
    ])

    max_d = np.nanmax(dists)
    if max_d < 1e-9:
        return 1.0

    return float(np.nanmin(dists) / max_d)


def compute_sh_distance_series(seg, near_side):
    """
    Return shoulder-hip distance time series normalised to cycle maximum.

    Used by visualise_detail only — not called during scoring.

    Returns
    -------
    np.ndarray  shape (n,)  normalised distance [0, 1]
    """
    prefix = 'l' if near_side == 'left' else 'r'
    sh_xy  = _jxy(seg, f'{prefix}_shoulder')
    hip_xy = _jxy(seg, f'{prefix}_hip')

    n     = len(seg)
    dists = np.array([segment_length(sh_xy[i], hip_xy[i]) for i in range(n)])
    max_d = np.nanmax(dists) + 1e-9

    return dists / max_d
