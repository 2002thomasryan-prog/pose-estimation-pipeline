"""
Parameter 2 - peak-to-peak vertical hip displacement per crank cycle, normalised to a body-height proxy.

Uses the bilateral hip midpoint to reduce noise from asymmetric pedalling;
the height proxy is the shoulder-ankle distance.
Optimal: below 0.03, i.e. under 3% of body height (Bini et al. 2010).
"""

import numpy as np


def compute_pelvic_stability(seg, kp_map, body_h):
    """
    Parameters
    ----------
    seg    : np.ndarray  shape (n, n_kp, 3)  keypoints for one cycle
    kp_map : dict        keypoint name → index map
    body_h : float       shoulder–ankle height proxy (pixels), pre-computed
                         from the first frame at the scorer level

    Returns
    -------
    float  Normalised peak-to-peak vertical hip displacement.
    """
    l_hip = kp_map['l_hip']
    r_hip = kp_map['r_hip']

    hip_y_mid = (seg[:, l_hip, 1] + seg[:, r_hip, 1]) / 2
    return float(np.ptp(hip_y_mid)) / body_h
