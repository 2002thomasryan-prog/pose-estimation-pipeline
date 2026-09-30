"""
Parameter 4 - spinal stability, as the min/max shoulder-to-hip distance ratio across the crank cycle.

Uses bilateral midpoints rather than near-side only, since the camera angle
may not reliably isolate one side and trunk rounding in cycling is a
sagittal-plane phenomenon visible from either. A ratio near 1.0 means the
rider holds a consistent trunk length. Dimensionless, not an angle.
Optimal: 0.88-1.0 (Burt et al. 2014; Silberman et al. 2005).
"""

import numpy as np


def compute_spinal_flexion(seg, kp_map):
    """
    Compute shoulder–hip distance stability ratio across one crank cycle.

    Parameters
    ----------
    seg    : np.ndarray  shape (n, n_kp, 3)  keypoints for one cycle
    kp_map : dict        keypoint name → index map

    Returns
    -------
    float  Ratio in [0, 1].  1.0 = no trunk instability detected.
    """
    l_sh  = kp_map['l_shoulder']
    r_sh  = kp_map['r_shoulder']
    l_hip = kp_map['l_hip']
    r_hip = kp_map['r_hip']

    dists = []
    for f in seg:
        sh_mid  = (f[l_sh,  :2] + f[r_sh,  :2]) / 2.0
        hip_mid = (f[l_hip, :2] + f[r_hip, :2]) / 2.0
        dists.append(float(np.linalg.norm(sh_mid - hip_mid)))

    dists  = np.array(dists)
    max_d  = np.nanmax(dists)
    if max_d < 1e-9:
        return 1.0

    return float(np.nanmin(dists) / max_d)
