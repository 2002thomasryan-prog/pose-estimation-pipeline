"""
Parameter targets, scoring tolerances and cycle-detection constants for the cycling scorer.

METRIC_RULES    target strings + uncertainty for traffic-light scoring
CYCLING_PARAMS  dashboard-compatible definitions used by the Dash UI

Derivation of the uncertainty bands is recorded in CALIBRATION_NOTES_cycling.md.
"""

CYCLE_MIN_FRAMES     = 10
BDC_PEAK_PROM_FRAC   = 0.30
BDC_MIN_DISTANCE     = 15


METRIC_RULES = {
    'knee_angle_tdc': {
        'label':       'Knee Angle at TDC',
        'target':      '55-75°',
        'uncertainty': 5,
    },
    'knee_angle_bdc': {
        'label':       'Knee Angle at BDC',
        'target':      '140-155°',
        'uncertainty': 5,
    },
    'trunk_angle': {
        'label':       'Trunk Angle',
        'target':      '10-45°',
        'uncertainty': 3,
    },
    'spinal_flexion': {
        'label':       'Spinal Flexion Ratio',
        'target':      '>0.88',
        'uncertainty': 0.05,
    },
    'pelvic_stability': {
        'label':       'Pelvic Stability',
        'target':      '<0.03',
        'uncertainty': 0.01,
    },
    'ankle_rom': {
        'label':       'Ankle ROM',
        'target':      '20-35°',
        'uncertainty': 5,
    },
    'bilateral_symmetry': {
        'label':       'Bilateral Symmetry',
        'target':      '<10',
        'uncertainty': 2,
    },
}


CYCLING_PARAMS = [
    {
        'key':              'knee_angle_tdc',
        'label':            'Knee Angle at TDC',
        'unit':             '°',
        'optimal':          [55, 75],
        'description':      'Included knee angle at the top of the pedal stroke (Top '
                            'Dead Centre). The leg is compressed at TDC; an included '
                            'angle of 55–75° is typical for well-fitted road geometry.',
        'source':           'Bini et al. 2011; Swart & Holliday 2019',
        'event':            'tdc',
        'event_frames_key': 'tdc',
    },
    {
        'key':              'knee_angle_bdc',
        'label':            'Knee Angle at BDC',
        'unit':             '°',
        'optimal':          [140, 155],
        'description':      'Included knee angle at the bottom of the pedal stroke '
                            '(Bottom Dead Centre). Primary indicator of saddle height. '
                            'Optimal 140–155° included (near straight without locking out).',
        'source':           'Bini et al. 2011; Peveler 2008',
        'event':            'bdc',
        'event_frames_key': 'bdc',
    },
    {
        'key':              'trunk_angle',
        'label':            'Trunk Angle',
        'unit':             '°',
        'optimal':          [10, 45],
        'description':      'Forward lean of the trunk from vertical. Influences '
                            'aerodynamics and power transfer. Optimal 10–45° for '
                            'road cycling.',
        'source':           'Bini et al. 2011',
        'event':            'drive',
        'event_frames_key': 'bdc',
    },
    {
        'key':              'spinal_flexion',
        'label':            'Spinal Flexion',
        'unit':             'ratio',
        'optimal':          [0.88, 1.0],
        'description':      'Minimum / maximum shoulder-to-hip distance ratio across '
                            'the crank cycle, using bilateral midpoints. A ratio near '
                            '1.0 indicates the rider holds a consistent trunk length; '
                            '<0.88 indicates meaningful trunk compression or lumbar '
                            'rounding. Dimensionless — not an angle.',
        'source':           'Burt et al. 2014; Silberman et al. 2005',
        'event':            'drive',
        'event_frames_key': 'bdc',
    },
    {
        'key':              'pelvic_stability',
        'label':            'Pelvic Stability',
        'unit':             'norm.',
        'optimal':          [0.0, 0.03],
        'description':      'Peak-to-peak vertical hip displacement per cycle, '
                            'normalised to body height. Excessive rocking indicates '
                            'saddle too high or hip weakness. Optimal <3%.',
        'source':           'Bini et al. 2010',
        'event':            'recovery',
        'event_frames_key': 'tdc',
    },
    {
        'key':              'ankle_rom',
        'label':            'Ankle ROM',
        'unit':             '°',
        'optimal':          [20, 35],
        'description':      'Range of ankle dorsiflexion/plantarflexion across the '
                            'crank cycle. Active ankle motion improves power transfer '
                            'at the pedal. Optimal 20–35°.',
        'source':           'Bini et al. 2011',
        'event':            'recovery',
        'event_frames_key': 'tdc',
    },
    {
        'key':              'bilateral_symmetry',
        'label':            'Bilateral Symmetry',
        'unit':             '%',
        'optimal':          [0.0, 10.0],
        'description':      'Asymmetry index comparing left and right knee extension, '
                            'each sampled at its own BDC (the crank arms are 180° out '
                            'of phase). Low values indicate balanced power delivery. '
                            'Optimal <10%. Requires far-side landmarks — see the '
                            'validation limitation in the accompanying report.',
        'source':           'Bini et al. 2008',
        'event':            'symmetry',
        'event_frames_key': 'bdc',
    },
]
