"""
Modular running biomechanical scorer.

    from running_scorer import RunningScorer
    scorer = RunningScorer(keypoints_array, backend='mmpose', fps=30)
    results = scorer.run(save_path='running_radar.png')
"""

from .running_scorer import RunningScorer

__all__ = ['RunningScorer']
