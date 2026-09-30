"""
Parameter 3 - included knee angle (hip-knee-ankle) at the catch, near-side keypoints only.

Straight leg is approximately 180 deg. Note this is the INCLUDED angle, not
the clinical flexion angle (180 - included) sometimes cited as 80-105 deg.
Optimal: 40-55 deg (Kleshnev 2023 World Championships mean 47.3 deg, gold
medallists 45.4 deg; Buckeridge et al. 2012).
"""

from scorer_utils import _jxy, angle_at_joint


def compute_knee_angle_catch(seg, catch_local, near_side):
    """
    Compute the included knee angle at the catch position.

    Parameters
    ----------
    seg         : list[dict]  keypoints for one cycle (normalised)
    catch_local : int         local frame index of catch within seg.
                              For catch-to-catch cycles this is always 0.
    near_side   : str         'left' | 'right'

    Returns
    -------
    float  Included knee angle in degrees at catch.
    """
    prefix  = 'l' if near_side == 'left' else 'r'
    hip_xy  = _jxy(seg, f'{prefix}_hip')
    knee_xy = _jxy(seg, f'{prefix}_knee')
    ank_xy  = _jxy(seg, f'{prefix}_ankle')

    return float(angle_at_joint(
        hip_xy[catch_local],
        knee_xy[catch_local],
        ank_xy[catch_local],
    ))
