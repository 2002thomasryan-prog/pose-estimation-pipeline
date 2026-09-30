"""Determines which body side faces the camera, via a YOLOv8 confidence pass with a sport-specific geometric fallback."""

import numpy as np
import cv2

_yolo_model = None
_YOLO_MODEL_NAME = 'yolov8n-pose.pt'

_LEFT_IDX  = [11, 13, 15]
_RIGHT_IDX = [12, 14, 16]

_N_SAMPLE_FRAMES = 20


def detect_side(kp_seq, video_path=None, kp_map=None, sport=None):
    """
    Unified side-detection entry point for all three sports.

    Parameters
    ----------
    kp_seq     : np.ndarray  (n_frames, n_kp, 2+)  raw keypoint array
    video_path : str | None  path to source video (used for YOLO confidence pass)
    kp_map     : dict | None  backend keypoint name→index map.
                 Used by the geometric fallback so the correct joint indices
                 are read regardless of backend (COCO-17, Halpe26, MHR70).
    sport      : str | None  'rowing' | 'running' | 'cycling'
                 Selects the sport-specific geometric fallback.

    Returns
    -------
    str  'left' | 'right'
    """
    return _detect(kp_seq, video_path, kp_map, sport)


def detect_camera_side(kp_seq, backend=None, kp_norm=None,
                       video_path=None, kp_map=None, sport='rowing'):
    """Alias for detect_side — used by RowingScorer."""
    return _detect(kp_seq, video_path, kp_map, sport)


def detect_lateral_side(kp_seq, backend=None, kp_norm=None,
                        video_path=None, kp_map=None, sport=None):
    """Alias for detect_side — used by RunningScorer and CyclingScorer."""
    return _detect(kp_seq, video_path, kp_map, sport)


def _detect(kp_seq, video_path, kp_map, sport):
    if video_path:
        try:
            return _yolo_confidence(video_path)
        except Exception:
            pass
    return _geometric_fallback(kp_seq, kp_map, sport)


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(_YOLO_MODEL_NAME)
    return _yolo_model


def _yolo_confidence(video_path):
    """
    Sample N evenly-spaced frames from the video, run YOLOv8-Pose on each,
    and compare mean confidence of left vs right lower-body joints.

    Always uses COCO-17 indices since YOLO always returns COCO-17 layout
    regardless of which primary backend was used for the full pipeline.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {video_path}')

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, _N_SAMPLE_FRAMES, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError('No frames could be read from video.')

    model       = _load_yolo()
    left_confs  = []
    right_confs = []

    for frame in frames:
        results = model(frame, verbose=False)
        if not results or results[0].keypoints is None:
            continue
        kps = results[0].keypoints.data
        if kps.shape[0] == 0:
            continue
        person_conf = kps[:, :, 2].mean(dim=1)
        best = int(person_conf.argmax())
        kp   = kps[best].cpu().numpy()

        if kp.shape[0] > max(_RIGHT_IDX):
            left_confs.append(np.nanmean(kp[_LEFT_IDX,  2]))
            right_confs.append(np.nanmean(kp[_RIGHT_IDX, 2]))

    if not left_confs:
        raise RuntimeError('YOLO produced no usable detections.')

    return 'left' if np.mean(left_confs) >= np.mean(right_confs) else 'right'


def _geometric_fallback(kp_seq, kp_map, sport):
    """
    Dispatch to the correct geometric fallback for the given sport.

    Rowing  → wrist x-range  (ankles are footplate-fixed, useless signal)
    Running → ankle x-range  (near-side stance ankle has larger swing arc)
    Cycling → ankle x-range  (near-side pedalling ankle has larger arc)
    None    → wrist first, ankle second (safest default)
    """
    if sport == 'rowing':
        return _wrist_xrange(kp_seq, kp_map)
    else:
        return _ankle_xrange(kp_seq, kp_map)


def _wrist_xrange(kp_seq, kp_map=None):
    """
    Compare full-video x-displacement range of left vs right wrist.

    The near-side wrist has a much larger anterior-posterior swing arc
    than the far-side wrist (typically 100+ px difference for rowing).
    Uses kp_map to resolve the correct backend-specific wrist indices.
    Falls back to 'right' if indices are out of range.
    """
    kp_seq = np.asarray(kp_seq, dtype=float)

    if kp_map is not None:
        l_idx = kp_map.get('l_wrist', 9)
        r_idx = kp_map.get('r_wrist', 10)
    else:
        l_idx, r_idx = 9, 10

    if kp_seq.shape[1] > max(l_idx, r_idx):
        l_range = np.nanmax(kp_seq[:, l_idx, 0]) - np.nanmin(kp_seq[:, l_idx, 0])
        r_range = np.nanmax(kp_seq[:, r_idx, 0]) - np.nanmin(kp_seq[:, r_idx, 0])
        return 'left' if l_range >= r_range else 'right'

    return 'right'


def _ankle_xrange(kp_seq, kp_map=None):
    """
    Compare full-video x-displacement range of left vs right ankle.

    Uses kp_map to resolve ankle indices for the actual backend so that
    SAM-3D (ankle = 13/14) and RTMPose (ankle = 15/16) are both handled
    correctly.  Falls back to COCO-17 indices (15/16) only when kp_map
    is not supplied.
    """
    kp_seq = np.asarray(kp_seq, dtype=float)

    if kp_map is not None:
        l_idx = kp_map.get('l_ankle', 15)
        r_idx = kp_map.get('r_ankle', 16)
    else:
        l_idx, r_idx = 15, 16

    if kp_seq.shape[1] > max(l_idx, r_idx):
        l_range = np.nanmax(kp_seq[:, l_idx, 0]) - np.nanmin(kp_seq[:, l_idx, 0])
        r_range = np.nanmax(kp_seq[:, r_idx, 0]) - np.nanmin(kp_seq[:, r_idx, 0])
        return 'left' if l_range >= r_range else 'right'

    return 'right'
