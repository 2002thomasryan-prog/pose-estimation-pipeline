"""
Parameter targets, scoring tolerances and event-detection constants for the running scorer.

METRIC_RULES    target strings + uncertainty for traffic-light scoring
RUNNING_PARAMS  dashboard-compatible definitions used by the Dash UI

Calibration provenance for the constants below is recorded in
CALIBRATION_NOTES.md.
"""

CYCLE_MIN_FRAMES        = 10


PLATEAU_DELTA_TD        = 0.20

PLATEAU_DELTA_TO        = 0.09

PLATEAU_GAP_TOL         = 2

PLATEAU_MIN_DUR         = 3

DELTA_TO_ANGLE_REF      = 3.0
DELTA_TO_MAX_BOOST      = 4.4
DELTA_TO_POWER          = 0.6

STANCE_SWING_OUTLIER_STD_MULT = 0.5
STANCE_SWING_OUTLIER_MIN_FRAMES = 0.5
CROSSING_JUMP_LOW_Z            = -0.75

TD_APPROACH_OUTLIER_Z          = 0.5

EXCLUDE_LOW_CONFIDENCE_CYCLES = True


METRIC_RULES = {
    'thigh_sep_mvp': {
        'label':       'Thigh Separation (MVP)',
        'target':      '90°±10°',
        'uncertainty': 10,
    },
    'trailing_shin_mvp': {
        'label':       'Trailing Shin Angle (MVP)',
        'target':      'shin downward',
        'uncertainty': None,
    },
    'leading_knee_mvp': {
        'label':       'Leading Knee Angle (MVP)',
        'target':      '>110°',
        'uncertainty': 5,
    },
    'trailing_foot_to': {
        'label':       'Trailing Foot Perp. (TO)',
        'target':      '0°±10°',
        'uncertainty': 10,
    },
    'thigh_sep_to': {
        'label':       'Thigh Separation (TO)',
        'target':      '90°±10°',
        'uncertainty': 10,
    },
    'leading_ankle_to': {
        'label':       'Leading Ankle Flex. (TO)',
        'target':      '80-100°',
        'uncertainty': 11,
    },
    'trailing_knee_td': {
        'label':       'Trailing Knee Pos. (TD)',
        'target':      '>0°',
        'uncertainty': 0,
    },
    'landing_foot_td': {
        'label':       'Landing Foot Angle (TD)',
        'target':      '-20-20°',
        'uncertainty': 10,
    },
    'shin_ext_td': {
        'label':       'Shin Extension (TD)',
        'target':      '<0°',
        'uncertainty': 5,
    },
    'torso_tilt_td': {
        'label':       'Torso Tilt (TD)',
        'target':      '<15°',
        'uncertainty': 3,
    },
    'thigh_gap_ms': {
        'label':       'Thigh Gap (MS)',
        'target':      '>0°',
        'uncertainty': 0,
    },
    'sinking_ms': {
        'label':       'Sinking (MS)',
        'target':      '<0°',
        'uncertainty': 7,
    },
}

RUNNING_PARAMS = [
    {
        'key':              'thigh_sep_mvp',
        'label':            'Thigh Sep. (MVP)',
        'unit':             '°',
        'optimal':          [80, 100],
        'description':      'Angle between thigh vectors at max velocity phase. Ideal 90±10°.',
        'event_frames_key': 'mvp_frames',
    },
    {
        'key':              'trailing_shin_mvp',
        'label':            'Trailing Shin (MVP)',
        'unit':             'score',
        'optimal':          [0.5, 1.0],
        'description':      'Trailing shin angled downward at MVP (1=pass, 0=fail).',
        'event_frames_key': 'mvp_frames',
        'directional_labels': ('Ankle below knee', 'Ankle above knee'),
    },
    {
        'key':              'leading_knee_mvp',
        'label':            'Leading Knee (MVP)',
        'unit':             '°',
        'optimal':          [110, 180],
        'description':      'Leading knee should be extended (>110°) at MVP.',
        'event_frames_key': 'mvp_frames',
    },
    {
        'key':              'trailing_foot_to',
        'label':            'Trailing Foot (TO)',
        'unit':             '°',
        'optimal':          [-10, 10],
        'description':      'Trailing foot deviation from vertical at toe-off. Ideal 0±10°.',
        'event_frames_key': 'to_frames',
    },
    {
        'key':              'thigh_sep_to',
        'label':            'Thigh Sep. (TO)',
        'unit':             '°',
        'optimal':          [80, 100],
        'description':      'Angle between thigh vectors at toe-off. Ideal 90±10°.',
        'event_frames_key': 'to_frames',
    },
    {
        'key':              'leading_ankle_to',
        'label':            'Leading Ankle (TO)',
        'unit':             '°',
        'optimal':          [80, 100],
        'description':      'Leading foot-to-shank angle at toe-off. Ideal 80-100°.',
        'event_frames_key': 'to_frames',
    },
    {
        'key':              'trailing_knee_td',
        'label':            'Trailing Knee (TD)',
        'unit':             '°',
        'optimal':          [0, 45],
        'description':      'Trailing thigh signed angle relative to vertical at touchdown. Should be behind hip (>0°).',
        'event_frames_key': 'td_frames',
        'directional_labels': ('Knee behind hip', 'Knee ahead of hip'),
    },
    {
        'key':              'landing_foot_td',
        'label':            'Landing Foot (TD)',
        'unit':             '°',
        'optimal':          [-20, 20],
        'description':      'Signed toe->heel angle at touchdown. Ideal +-20°.',
        'event_frames_key': 'td_frames',
    },
    {
        'key':              'shin_ext_td',
        'label':            'Shin Extension (TD)',
        'unit':             '°',
        'optimal':          [-45, 0],
        'description':      'Signed shin angle relative to vertical at touchdown. Should be forward-angled (<0°).',
        'event_frames_key': 'td_frames',
        'directional_labels': ('Ankle behind knee', 'Barely ahead of knee', 'Ankle ahead of knee'),
    },
    {
        'key':              'torso_tilt_td',
        'label':            'Torso Tilt (TD)',
        'unit':             '°',
        'optimal':          [0, 15],
        'description':      'Hip-to-shoulder lean angle at touchdown. Should be upright (<15°).',
        'event_frames_key': 'td_frames',
    },
    {
        'key':              'thigh_gap_ms',
        'label':            'Thigh Gap (MS)',
        'unit':             '°',
        'optimal':          [0, 180],
        'description':      'Signed thigh-separation angle at mid-stance. Positive = far knee has driven ahead of the near knee (pass). Negative = far knee still behind (fail).',
        'event_frames_key': 'ms_frames',
    },
    {
        'key':              'sinking_ms',
        'label':            'Sinking (MS)',
        'unit':             '°',
        'optimal':          [-45, 0],
        'description':      'Ankle->knee angle at mid-stance. Negative = knee behind toe (correct). Positive = knee ahead of toe (collapsing).',
        'event_frames_key': 'ms_frames',
        'directional_labels': ('Knee behind toe', 'Knee ahead of toe'),
    },
]
