"""
Detects crank cycles from dominant-side ankle Bottom Dead Centre.

In image coordinates Y increases downward, so BDC is the frame of maximum
ankle Y (lowest point of the pedal stroke).
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from .params import BDC_PEAK_PROM_FRAC, BDC_MIN_DISTANCE


def detect_cycling_cycles(kp_seq, kp_map, side='right'):
    """
    Detect crank cycles from dominant-side ankle Y maxima (BDC events).

    Parameters
    ----------
    kp_seq : np.ndarray  shape (n_frames, n_kp, 3)
    kp_map : dict        keypoint name → index map
    side   : str         'right' | 'left'  dominant / drive side

    Returns
    -------
    cycles     : list[(start_frame, end_frame)]
    bdc_frames : np.ndarray  global BDC frame indices
    """
    prefix    = 'r' if side == 'right' else 'l'
    ankle_idx = kp_map[f'{prefix}_ankle']
    ankle_y   = uniform_filter1d(kp_seq[:, ankle_idx, 1], size=7)

    prom = BDC_PEAK_PROM_FRAC * np.ptp(ankle_y)
    bdc_frames, _ = find_peaks(ankle_y, prominence=prom,
                                distance=BDC_MIN_DISTANCE)

    cycles = [
        (int(bdc_frames[i]), int(bdc_frames[i + 1]))
        for i in range(len(bdc_frames) - 1)
    ]

    return cycles, bdc_frames
