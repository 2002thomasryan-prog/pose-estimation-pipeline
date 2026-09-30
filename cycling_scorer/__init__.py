"""
Modular cycling biomechanical scorer.

    from cycling_scorer import CyclingScorer
    scorer = CyclingScorer(keypoints_array, backend='mmpose', fps=30)
    results = scorer.run(save_path='cycling_radar.png')
"""

from .cycling_scorer import CyclingScorer

__all__ = ['CyclingScorer']
