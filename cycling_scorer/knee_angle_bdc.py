"""
Knee angle at Bottom Dead Centre, the frame of maximum ankle Y within each crank cycle.

Returns the INCLUDED angle (hip-knee-ankle), not clinical flexion: a straight
leg is approximately 180 deg. Primary indicator of saddle height.
Optimal: 140-155 deg included, near straight without locking out
(Bini et al. 2011; Peveler 2008).
"""

import numpy as np
from scorer_utils import angle_at_joint


def compute_knee_angle_bdc(seg, kp_map, side='right'):
    """
    Knee flexion angle at BDC within one crank cycle.

    BDC is identified as the maximum Y (lowest point) of the dominant-side ankle.

    Parameters
    ----------
    seg     : np.ndarray  shape (n, n_kp, 3)  keypoints for one cycle
    kp_map  : dict        keypoint name → index map
    side    : str         'right' | 'left'

    Returns
    -------
    float  Knee flexion angle in degrees at BDC.
    """
    prefix    = 'r' if side == 'right' else 'l'
    hip_idx   = kp_map[f'{prefix}_hip']
    knee_idx  = kp_map[f'{prefix}_knee']
    ankle_idx = kp_map[f'{prefix}_ankle']

    bdc_local = int(np.argmax(seg[:, ankle_idx, 1]))
    f = seg[bdc_local]

    included = angle_at_joint(f[hip_idx, :2], f[knee_idx, :2], f[ankle_idx, :2])
    return float(included)
