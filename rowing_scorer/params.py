"""
Parameter targets, scoring tolerances and cycle-detection constants for the rowing scorer.

METRIC_RULES   traffic-light target strings + uncertainty per parameter
ROWING_PARAMS  dashboard-compatible definitions (radar, tables, event cards)

Derivation of the uncertainty bands, the RSF definition correction, and the
split of trunk angle into separate catch and finish measurements are recorded
in CALIBRATION_NOTES_rowing.md.

Drive and recovery sequencing are NOT in METRIC_RULES.  They are scored inside
coordination.py against the SEQ_* constants below (or, with SEQ_METHOD =
'centroid', against the clip's own timing resolution), and their records are
built there.

The report quotes the rowing angles as approximate to +/-5 deg (18 Sept 2026).
Trunk angle at finish keeps its 3 deg scoring band: with 5 deg on a 10 deg wide
target no value could ever score GREEN.  (Trunk angle at catch already has
that problem at 5 deg -- see CALIBRATION_NOTES_rowing.md.)  RSF is still
computed and exposed in to_dict(), but it is no longer separately scored: the
fault it detects -- the handle outrunning the seat off the catch -- is the
trunk contributing before the legs, which drive sequencing measures directly,
and two metrics reading one fault were able to disagree.
"""

CYCLE_MIN_FRAMES      = 15

CATCH_SMOOTH_WINDOW   = 9
CATCH_PEAK_PROM_FRAC  = 0.08
CATCH_PEAK_MIN_DIST   = 20
CATCH_PLATEAU_TOL     = 0.02
CATCH_PLATEAU_MIN     = 3

ONSET_THRESHOLD       = 0.05


# --- Sequencing method (meeting with Ciarán, 18 Sept 2026) -----------------
# 'completion'  angle-based: how far through its angle change the previous
#               segment is (knee -> trunk -> elbow on the drive, reversed on
#               the recovery) at the moment the next segment starts moving.
# 'centroid'    the previous method: gap between contribution centroids,
#               passed against the clip's timing resolution.
# Switch back to 'centroid' to reproduce reports generated before this change.
# Back on 'centroid' (28 Sept 2026): on real clips the angle-based bars showed
# far more overlap than the centroid method.  'completion' is kept for testing.
SEQ_METHOD        = 'centroid'

# The next segment counts as "started" once it has done this fraction of its
# angle change over the phase.
SEQ_ONSET_FRAC    = 0.05

# Previous segment's completion when the next one starts:
#   >= SEQ_GREEN_FRAC            GREEN  (in sequence)
#   <  SEQ_RED_FRAC              RED    (next started before previous was halfway)
#   anything in between          AMBER  (segments overlap)
# Ciarán suggested also trying 0.80 as the green cut-off: change it here.
SEQ_GREEN_FRAC    = 0.95
SEQ_RED_FRAC      = 0.50

# Reported (not scored): does the second segment start in the last quarter of
# the first segment's movement, e.g. trunk in the last quarter of leg drive.
SEQ_LAST_QUARTER  = 0.75

# Angle smoothing window (frames) applied over the whole cycle before slicing
# a phase, so the phase edges are not smoothed against padding.
SEQ_ANGLE_SMOOTH  = 5

SMOOTH_WINDOW         = CATCH_SMOOTH_WINDOW
PEAK_PROM_FRAC        = CATCH_PEAK_PROM_FRAC
PEAK_MIN_DISTANCE     = CATCH_PEAK_MIN_DIST
PLATEAU_TOL           = CATCH_PLATEAU_TOL
PLATEAU_MIN_FRAMES    = CATCH_PLATEAU_MIN
CATCH_MIN_DISTANCE    = CATCH_PEAK_MIN_DIST
CATCH_CROSS_VAL_TOL   = 8
EDGE_CATCH_MARGIN     = 10


METRIC_RULES = {
    'knee_angle_catch': {
        'label':       'Knee Angle at Catch',
        'target':      '35-65°',
        'uncertainty': 5,
    },
    'rsf': {
        'label':       'Rowing Style Factor',
        'target':      '90-100',
        'uncertainty': 5,
    },
    'trunk_angle_catch': {
        'label':       'Trunk Angle at Catch',
        'target':      '25-35°',
        'uncertainty': 5,
    },
    'trunk_angle_finish': {
        'label':       'Trunk Angle at Finish',
        'target':      '-30--20°',
        'uncertainty': 3,
    },
    'spinal_flexion': {
        'label':       'Spinal Flexion Ratio',
        'target':      '>0.88',
        'uncertainty': 0.05,
    },
}


ROWING_PARAMS = [
    {
        'key':              'knee_angle_catch',
        'label':            'Knee Angle at Catch',
        'unit':             '°',
        'optimal':          [35, 65],
        'description':      'Included knee angle at catch (leg compression position). '
                            'Target 35–65°. Too closed (<35°) limits leg drive; '
                            'too open (>65°) indicates inadequate reach.',
        'source':           'Kleshnev 2023; Buckeridge et al. 2012',
        'event':            'catch',
        'event_frames_key': 'catch',
    },
    {
        'key':              'trunk_angle_catch',
        'label':            'Trunk Angle at Catch',
        'unit':             '°',
        'optimal':          [25, 35],
        'description':      'Signed trunk angle at the catch, 0° upright. Positive = '
                            'forward lean toward the footplate. 25–35° optimal. Under '
                            '25° is short reach; over 35° over-reaches and rounds the '
                            'lumbar spine.',
        'source':           'British Rowing Plus (Clark 2022); Soper & Hume 2004',
        'event':            'catch',
        'event_frames_key': 'catch',
    },
    {
        'key':              'drive_sequencing',
        'label':            'Drive Sequencing',
        'unit':             '%',
        'optimal':          [0, 100],      # replaced per clip by the resolution floor
        'description':      'How far through its angle change each segment is when the '
                            'next one starts (knee, then trunk, then elbow); the weaker '
                            'of the two handovers is reported. 95% or more is in '
                            'sequence; under 50% is out of sequence; in between the '
                            'segments overlap.',
        'source':           'Kleshnev 2010, 2016; British Rowing Technical Model',
        'event':            'drive',
        'event_frames_key': 'finish',
    },
    {
        'key':              'trunk_angle_finish',
        'label':            'Trunk Angle at Finish',
        'unit':             '°',
        'optimal':          [-30, -20],
        'description':      'Signed trunk angle at the finish, 0° upright. Negative = '
                            'layback. −30 to −20° optimal. Shallower than −20° wastes '
                            'stroke length; beyond −30° loads the lumbar spine and '
                            'delays the recovery.',
        'source':           'British Rowing Plus (Clark 2022); Soper & Hume 2004',
        'event':            'finish',
        'event_frames_key': 'finish',
    },
    {
        'key':              'spinal_flexion',
        'label':            'Spinal Flexion',
        'unit':             'ratio',
        'optimal':          [0.88, 1.0],
        'description':      'Minimum / maximum shoulder-to-hip distance ratio across '
                            'cycle. >0.88 = minimal rounding. <0.88 = significant '
                            'lumbar flexion (injury risk).',
        'source':           'Nugent et al. 2021; Caldwell et al. 2003',
        'event':            'finish',
        'event_frames_key': 'finish',
    },
    {
        'key':              'recovery_sequencing',
        'label':            'Recovery Sequencing',
        'unit':             '%',
        'optimal':          [0, 100],      # replaced per clip by the resolution floor
        'description':      'How far through its angle change each segment is when the '
                            'next one starts on the recovery (elbow, then trunk, then '
                            'knee). A low figure on trunk to knee means the slide '
                            'starts before the body is over, the most common recovery '
                            'fault.',
        'source':           'Kleshnev 2016; British Rowing Technical Model',
        'event':            'recovery',
        'event_frames_key': 'catch',
    },
]
