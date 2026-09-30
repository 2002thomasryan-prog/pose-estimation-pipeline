"""
Parameter 6 - forward trunk lean, signed mean angle from vertical across the cycle.

Positive = forward lean. Uses bilateral hip and shoulder midpoints.
Optimal: 5-20 deg forward lean (Bramah et al. 2024 S-MAS); note this range is
calibrated at near-maximal sprint speeds, so a smaller lean is expected and
normal at submaximal speeds.
"""

import numpy as np
from scorer_utils import signed_trunk_angle


def compute_trunk_lean(seg, kp_map):
    """
    Parameters
    ----------
    seg    : np.ndarray  shape (n, n_kp, 3)  keypoints for one cycle
    kp_map : dict        keypoint name → index map

    Returns
    -------
    float  Mean signed trunk lean in degrees across the cycle.
    """
    l_sh  = kp_map['l_shoulder']; r_sh  = kp_map['r_shoulder']
    l_hip = kp_map['l_hip'];      r_hip = kp_map['r_hip']

    n = len(seg)
    leans = np.array([
        signed_trunk_angle(
            (seg[i, l_hip, :2] + seg[i, r_hip, :2]) / 2,
            (seg[i, l_sh,  :2] + seg[i, r_sh,  :2]) / 2,
        )
        for i in range(n)
    ])

    return float(np.mean(leans))
