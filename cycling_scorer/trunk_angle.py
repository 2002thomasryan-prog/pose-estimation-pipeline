"""
Parameter 5 - mean forward trunk lean from vertical across the crank cycle.

Uses the bilateral shoulder-hip midpoint vector, and the unsigned
angle_from_vertical helper so the result holds regardless of which way the
cyclist faces the camera. Cycling trunk lean is largely static, so the mean
is the appropriate statistic rather than a range.
Optimal: 10-45 deg from vertical (Bini et al. 2011, 2014; Barratt et al. 2011).
"""

import numpy as np
from scorer_utils import angle_from_vertical


def compute_trunk_angle(seg, kp_map):
    """
    Compute mean forward trunk lean from vertical across one crank cycle.

    Uses bilateral shoulder and hip midpoints to reduce sensitivity to
    out-of-plane rotation and asymmetric posture.

    Parameters
    ----------
    seg    : np.ndarray  shape (n, n_kp, 3)  keypoints for one cycle
    kp_map : dict        keypoint name → index map

    Returns
    -------
    float  Mean trunk lean in degrees from vertical (always positive).
    """
    l_sh  = kp_map['l_shoulder']
    r_sh  = kp_map['r_shoulder']
    l_hip = kp_map['l_hip']
    r_hip = kp_map['r_hip']

    angles = []
    for f in seg:
        sh_mid  = (f[l_sh,  :2] + f[r_sh,  :2]) / 2.0
        hip_mid = (f[l_hip, :2] + f[r_hip, :2]) / 2.0
        angles.append(angle_from_vertical(sh_mid, hip_mid))

    return float(np.mean(angles))
