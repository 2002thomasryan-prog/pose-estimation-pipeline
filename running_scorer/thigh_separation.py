"""
Parameter 5 - thigh separation at maximum vertical propulsion (MVP).

Both thigh vectors are anchored at the left hip, per Baida (2025), measuring
the angular spread of both legs from one reference.
Optimal: 80-100 deg (Bramah et al. 2024 S-MAS; Baida 2025).
"""

import numpy as np


def compute_thigh_separation(mvp_frame, kp_map, side='right'):
    """
    Parameters
    ----------
    mvp_frame : np.ndarray  shape (n_kp, 3)  single frame at MVP event
    kp_map    : dict        keypoint name → index map
    side      : str         retained for API compatibility (unused — both legs
                            are always evaluated using Emma's L_HIP anchor)

    Returns
    -------
    float  Angle between thigh vectors in degrees.
    """
    lh_i = kp_map['l_hip']
    lk_i = kp_map['l_knee']
    rk_i = kp_map['r_knee']

    l_hip  = mvp_frame[lh_i, :2].astype(float)
    l_knee = mvp_frame[lk_i, :2].astype(float)
    r_knee = mvp_frame[rk_i, :2].astype(float)

    left_thigh  = l_knee - l_hip
    right_thigh = r_knee - l_hip

    nl = float(np.linalg.norm(left_thigh))
    nr = float(np.linalg.norm(right_thigh))

    if nl < 1e-6 or nr < 1e-6:
        return float('nan')

    cos_sep = float(np.clip(np.dot(left_thigh, right_thigh) / (nl * nr), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_sep)))
