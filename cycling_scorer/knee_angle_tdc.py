"""
Knee angle at Top Dead Centre, the frame of minimum ankle Y within each crank cycle.

Returns the INCLUDED angle (hip-knee-ankle), not clinical flexion: a straight
leg is approximately 180 deg and the compressed limb at TDC is lower.
Optimal: 55-75 deg included. Amber 45-55 or 75-85; red below 45 or above 85.
Derived from crank geometry relative to the BDC target combined with typical
seat-height kinematics (Bini et al. 2011; Swart & Holliday 2019).
"""

import numpy as np
from scorer_utils import angle_at_joint


def compute_knee_angle_tdc(seg, kp_map, side='right'):
    """
    Knee flexion angle at TDC within one crank cycle.

    TDC is identified as the minimum Y (highest image position) of the
    dominant-side ankle — the top of the pedal stroke.

    Parameters
    ----------
    seg     : np.ndarray  shape (n, n_kp, 3)  keypoints for one cycle
    kp_map  : dict        keypoint name → index map
    side    : str         'right' | 'left'

    Returns
    -------
    float  Knee flexion angle in degrees at TDC (180° − included angle).
    None   if the segment is empty or keypoints are degenerate.
    """
    if seg is None or len(seg) == 0:
        return None

    prefix    = 'r' if side == 'right' else 'l'
    hip_idx   = kp_map[f'{prefix}_hip']
    knee_idx  = kp_map[f'{prefix}_knee']
    ankle_idx = kp_map[f'{prefix}_ankle']

    ankle_y   = seg[:, ankle_idx, 1]
    tdc_local = int(np.argmin(ankle_y))
    f         = seg[tdc_local]

    hip   = f[hip_idx,   :2]
    knee  = f[knee_idx,  :2]
    ankle = f[ankle_idx, :2]

    if (np.linalg.norm(hip - knee) < 1e-6
            or np.linalg.norm(ankle - knee) < 1e-6):
        return None

    included = angle_at_joint(hip, knee, ankle)
    if not np.isfinite(included):
        return None

    return float(included)
