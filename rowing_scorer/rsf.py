"""
Parameter 6 - Rowing Style Factor: seat travel as a percentage of handle travel over the first 20% of the drive.

    RSF (%) = 100 * seat_travel / handle_travel

At the catch the rower is a rigid linkage, so an ideal leg-first drive moves
seat and handle equally and scores 100%. Travel is estimated as the
least-squares slope of the x-trace over the window, signed onto the drive
direction, rather than an endpoint difference. Seat proxy is near-side hip x,
handle proxy near-side wrist x.

    90-100%   optimal (Kleshnev's reported optimum is 90-95%)
    <80%      early trunk opening; the handle outruns the seat
    >105%     bum-shoving; the seat outruns the handle (>150% severe)

Reference: Kleshnev 2010, Proc IMechE Part P 224:63-74 (n > 25,000 strokes);
Kleshnev 2016, Biomechanics of Rowing, Crowood Press.
"""

import numpy as np
from scorer_utils import _jxy

RSF_WINDOW_FRAC   = 0.20
RSF_MIN_WINDOW    = 3
RSF_MIN_DRIVE     = 5
RSF_CAP           = 200.0
RSF_MIN_SEAT_SLOPE = 0.05


def _signed_slope(x, sgn):
    """Least-squares slope of x against frame index, projected onto `sgn`."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.nan
    t = np.arange(len(x), dtype=float)[ok]
    return float(np.polyfit(t, x[ok], 1)[0]) * sgn


def compute_rsf(seg, catch_local, finish_local, near_side):
    """
    Compute Rowing Style Factor for one stroke cycle.

    Parameters
    ----------
    seg          : list[dict]  keypoints for one cycle (normalised, pixel space)
    catch_local  : int         local catch frame index (typically 0)
    finish_local : int         local finish frame index
    near_side    : str         'left' | 'right'

    Returns
    -------
    float  RSF as a percentage in [0, RSF_CAP], or None if the drive phase is
           too short or the seat is effectively stationary.
    """
    drive_len = finish_local - catch_local
    if drive_len < RSF_MIN_DRIVE:
        return None

    win_len = max(RSF_MIN_WINDOW, int(round(RSF_WINDOW_FRAC * drive_len)))
    win_len = min(win_len, drive_len)
    prefix  = 'l' if near_side == 'left' else 'r'

    hip_xy = _jxy(seg, f'{prefix}_hip')
    wr_xy  = _jxy(seg, f'{prefix}_wrist')

    sgn = np.sign(hip_xy[finish_local, 0] - hip_xy[catch_local, 0])
    if sgn == 0:
        return None

    win      = slice(catch_local, catch_local + win_len)
    seat_x   = hip_xy[win, 0]
    handle_x = wr_xy[win, 0]

    if len(seat_x) < 2:
        return None

    seat_slope   = _signed_slope(seat_x,   sgn)
    handle_slope = _signed_slope(handle_x, sgn)

    if not np.isfinite(seat_slope) or not np.isfinite(handle_slope):
        return None

    if seat_slope < RSF_MIN_SEAT_SLOPE:
        return None

    if handle_slope <= 0:
        return RSF_CAP

    return float(np.clip(100.0 * seat_slope / handle_slope, 0.0, RSF_CAP))


def compute_rsf_series(seg, catch_local, finish_local, near_side):
    """
    Return normalised seat and handle cumulative displacement traces through
    the full drive phase.

    Used by visualise_detail only - not called during scoring.  Divergence of
    the two traces AFTER the first 20% is expected and correct (trunk and arms
    are meant to engage); RSF itself is only defined on the early window.

    Returns
    -------
    tuple  (seat_disp, handle_disp)  each normalised to the larger of the two
           maxima, so the dominant trace reaches 1.0.
    None   if the drive phase is too short.
    """
    drive_len = finish_local - catch_local
    if drive_len < RSF_MIN_DRIVE:
        return None

    prefix = 'l' if near_side == 'left' else 'r'
    hip_xy = _jxy(seg, f'{prefix}_hip')
    wr_xy  = _jxy(seg, f'{prefix}_wrist')

    sgn = np.sign(hip_xy[finish_local, 0] - hip_xy[catch_local, 0])
    if sgn == 0:
        sgn = 1.0

    win      = slice(catch_local, finish_local + 1)
    seat_x   = hip_xy[win, 0]
    handle_x = wr_xy[win, 0]

    seat_disp   = (seat_x   - seat_x[0])   * sgn
    handle_disp = (handle_x - handle_x[0]) * sgn

    total_max = max(np.nanmax(np.abs(seat_disp)),
                    np.nanmax(np.abs(handle_disp)), 1e-9)

    return seat_disp / total_max, handle_disp / total_max
