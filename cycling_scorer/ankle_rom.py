"""
Parameter 3 - dorsi/plantarflexion range through the crank cycle, computed bilaterally and averaged.

Where the backend has no dedicated foot points and heel/toe collapse onto the
ankle (e.g. YOLO), a synthetic foot direction is substituted to avoid a
degenerate zero-ROM result.
Optimal: 20-35 deg (Bini et al. 2011).
"""

import numpy as np
from scorer_utils import angle_at_joint


def compute_ankle_rom(seg, kp_map):
    """
    Parameters
    ----------
    seg    : np.ndarray  shape (n, n_kp, 3)  keypoints for one cycle
    kp_map : dict        keypoint name → index map

    Returns
    -------
    float  Mean bilateral ankle ROM in degrees.
    """
    sides = [
        (kp_map['l_knee'], kp_map['l_ankle'],
         kp_map.get('l_heel', kp_map['l_ankle']),
         kp_map.get('l_toe',  kp_map['l_ankle'])),
        (kp_map['r_knee'], kp_map['r_ankle'],
         kp_map.get('r_heel', kp_map['r_ankle']),
         kp_map.get('r_toe',  kp_map['r_ankle'])),
    ]

    roms = []
    for knee_idx, ank_idx, heel_idx, toe_idx in sides:
        angles = []
        for f in seg:
            knee_pt = f[knee_idx, :2]
            ank_pt  = f[ank_idx,  :2]
            foot_pt = f[heel_idx, :2] if heel_idx != ank_idx else f[toe_idx, :2]

            if np.allclose(foot_pt, ank_pt, atol=1.0):
                foot_pt = ank_pt + np.array([0.0, 15.0])

            angles.append(angle_at_joint(knee_pt, ank_pt, foot_pt))

        roms.append(float(np.ptp(angles)))

    return float(np.mean(roms))
