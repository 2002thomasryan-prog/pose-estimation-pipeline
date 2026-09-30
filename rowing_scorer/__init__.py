"""
Modular rowing biomechanical scorer.

    from rowing_scorer import RowingScorer
    scorer = RowingScorer(keypoints_array, backend='mmpose', fps=30)
    results = scorer.run(save_path='rowing_radar.png')

After run(), check scorer.warnings for quality alerts to surface in the GUI.
"""

from .rowing_scorer import RowingScorer


def visualise_rowing_detail(rowing_ts, param_scores, total_score,
                             n_cycles, backend,
                             frame_imgs=None, save_path=None, log_cb=None):
    """
    Stub — detailed rowing visualisation is deferred to the interactive
    visualisation overhaul.  report_generator.py imports this function
    directly so it must exist; it is a no-op until the new vis is built.
    """
    _log = log_cb or print
    _log('[RowingScorer] visualise_rowing_detail: deferred to visualisation overhaul — skipped.')


__all__ = ['RowingScorer', 'visualise_rowing_detail']
