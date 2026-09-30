"""Renders an annotated MP4 overlaying the pose skeleton and event markers onto the source video, for all three sports."""

import os
import cv2
import numpy as np


_SKELETON_PAIRS = [
    ('nose', 'l_eye'), ('nose', 'r_eye'),
    ('l_eye', 'l_ear'), ('r_eye', 'r_ear'),
    ('l_shoulder', 'r_shoulder'),
    ('l_shoulder', 'l_hip'), ('r_shoulder', 'r_hip'),
    ('l_hip', 'r_hip'),
    ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
    ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'),
    ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'),
    ('r_hip', 'r_knee'), ('r_knee', 'r_ankle'),
    ('l_ankle', 'l_heel'), ('l_heel', 'l_toe'),
    ('r_ankle', 'r_heel'), ('r_heel', 'r_toe'),
]

_COL_SKELETON = (180, 180, 180)
_COL_JOINT    = (255, 255, 255)
_STRIP_H      = 36


def _is_sam3d(backend: str) -> bool:
    b = backend.lower().replace('-', '').replace(' ', '')
    return b in ('sam3dbody', 'sam3d', 'mhr70')


def _resolve_skeleton(backend: str):
    """
    Return (skeleton, idx_map) where:
      skeleton : list of (idx_a, idx_b, bgr)  — for drawing lines
      idx_map  : dict canonical_name→int       — for resolving named joints
                 (used only for event highlight dots)

    SAM-3D: reads the module-level globals set when the model was loaded.
    RTMPose / YOLOv8: resolves name pairs from KEYPOINT_MAPS.
    """
    if _is_sam3d(backend):
        skeleton = []
        idx_map  = {}
        try:
            from pose_extractor import get_sam3d_connections, get_sam3d_name_to_idx
            conns = get_sam3d_connections()
            n2i   = get_sam3d_name_to_idx()
            if conns:
                skeleton = list(conns)
            if n2i:
                from scorer_utils import _build_kp_sam3d
                idx_map = _build_kp_sam3d()
                if not idx_map:
                    required = {
                        'nose': 'nose',
                        'l_eye': 'left_eye', 'r_eye': 'right_eye',
                        'l_ear': 'left_ear', 'r_ear': 'right_ear',
                        'l_shoulder': 'left_shoulder', 'r_shoulder': 'right_shoulder',
                        'l_elbow': 'left_elbow',       'r_elbow': 'right_elbow',
                        'l_wrist': 'left_wrist',       'r_wrist': 'right_wrist',
                        'l_hip': 'left_hip',           'r_hip': 'right_hip',
                        'l_knee': 'left_knee',         'r_knee': 'right_knee',
                        'l_ankle': 'left_ankle',       'r_ankle': 'right_ankle',
                        'l_heel': 'left_heel',         'r_heel': 'right_heel',
                        'l_toe': 'left_big_toe',       'r_toe': 'right_big_toe',
                    }
                    for canonical, mhr70_name in required.items():
                        if mhr70_name in n2i:
                            idx_map[canonical] = n2i[mhr70_name]
        except Exception:
            pass
        return skeleton, idx_map

    from scorer_utils import KEYPOINT_MAPS
    b = backend.lower().replace('-', '').replace(' ', '')
    alias = {'yolov8': 'yolov8', 'yolo': 'yolov8',
             'rtmpose': 'rtmpose', 'halpe26': 'rtmpose',
             'mmpose': 'mmpose'}
    key     = alias.get(b, 'rtmpose')
    idx_map = KEYPOINT_MAPS.get(key, KEYPOINT_MAPS.get('rtmpose', {}))
    skeleton = []
    for (a_name, b_name) in _SKELETON_PAIRS:
        ia = idx_map.get(a_name)
        ib = idx_map.get(b_name)
        if ia is not None and ib is not None:
            skeleton.append((ia, ib, _COL_SKELETON))
    return skeleton, idx_map


def _build_event_map_running(scorer_data):
    raw = scorer_data.get('raw_traces', {})
    definitions = [
        ('td_frames',  'TD',  (0,   0, 220), 'near_ankle'),
        ('ms_frames',  'MS',  (0, 200,  80), 'near_ankle'),
        ('to_frames',  'TO',  (0, 140, 255), 'near_toe'),
        ('mvp_frames', 'MVP', (220, 100,  0), 'near_toe'),
    ]
    event_map = {}
    for key, label, colour, joint in definitions:
        for f in raw.get(key, []):
            if f is None:
                continue
            f = int(f)
            if f in event_map:
                event_map[f]['label'] += f'/{label}'
            else:
                event_map[f] = {'label': label, 'colour': colour, 'joint': joint}
    return event_map


def _build_event_map_rowing(scorer_data):
    key_frames = scorer_data.get('key_frames', {})
    event_map  = {}
    for f in key_frames.get('catch', []):
        event_map[int(f)] = {'label': 'CATCH',  'colour': (0,   200, 220), 'joint': 'near_wrist'}
    for f in key_frames.get('finish', []):
        event_map[int(f)] = {'label': 'FINISH', 'colour': (220,   0, 200), 'joint': 'near_wrist'}
    return event_map


def _build_event_map_cycling(scorer_data):
    key_frames = scorer_data.get('key_frames', {})
    event_map  = {}
    for f in key_frames.get('tdc', []):
        event_map[int(f)] = {'label': 'TDC', 'colour': (0,   220, 220), 'joint': 'near_ankle'}
    for f in key_frames.get('bdc', []):
        event_map[int(f)] = {'label': 'BDC', 'colour': (200, 200, 200), 'joint': 'near_ankle'}
    return event_map


def _resolve_xy(frame_kps, idx, n_kps):
    if idx is None or idx >= n_kps:
        return None
    xy = frame_kps[idx, :2]
    if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
        return None
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def _near_side_prefix(scorer_data):
    side = scorer_data.get('dominant_side') or scorer_data.get('near_side') or 'r'
    return 'l' if str(side).lower().startswith('l') else 'r'


def _resolve_near_joint(frame_kps, idx_map, joint_template, side_prefix, n_kps):
    if joint_template and joint_template.startswith('near_'):
        name = side_prefix + '_' + joint_template[5:]
    else:
        name = joint_template or ''
    idx = idx_map.get(name)
    return _resolve_xy(frame_kps, idx, n_kps)


def _draw_skeleton(frame, frame_kps, skeleton, n_kps):
    h, w = frame.shape[:2]
    drawn = set()
    for (ia, ib, bgr) in skeleton:
        a = _resolve_xy(frame_kps, ia, n_kps)
        b = _resolve_xy(frame_kps, ib, n_kps)
        if a is None or b is None:
            continue
        if not (0 <= a[0] < w and 0 <= a[1] < h):
            continue
        if not (0 <= b[0] < w and 0 <= b[1] < h):
            continue
        cv2.line(frame, a, b, bgr, 2, cv2.LINE_AA)
        drawn.add(ia)
        drawn.add(ib)
    for idx in drawn:
        pt = _resolve_xy(frame_kps, idx, n_kps)
        if pt is None or not (0 <= pt[0] < w and 0 <= pt[1] < h):
            continue
        cv2.circle(frame, pt, 4, _COL_JOINT, -1, cv2.LINE_AA)


def _draw_event_strip(frame, label, colour, alpha=0.75):
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, _STRIP_H), colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    font = cv2.FONT_HERSHEY_DUPLEX
    fs   = 0.75
    th   = 2
    (tw, fh), _ = cv2.getTextSize(label, font, fs, th)
    tx = (w - tw) // 2
    ty = (_STRIP_H + fh) // 2 - 2
    cv2.putText(frame, label, (tx + 1, ty + 1), font, fs, (0, 0, 0),       th + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (tx,     ty),     font, fs, (255, 255, 255),  th,     cv2.LINE_AA)


def _draw_highlight_dot(frame, pt, colour, radius=10):
    cv2.circle(frame, pt, radius + 2, (255, 255, 255), 2,  cv2.LINE_AA)
    cv2.circle(frame, pt, radius,     colour,           -1, cv2.LINE_AA)


def _draw_frame_counter(frame, frame_idx, fps):
    h    = frame.shape[0]
    text = f'fr {frame_idx}  |  {frame_idx / max(fps, 1):.2f}s'
    cv2.putText(frame, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0),       2, cv2.LINE_AA)
    cv2.putText(frame, text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)


def write_annotated_video(video_path, kp_arr, scorer_data, output_path,
                          progress_cb=None):
    """
    Render annotated video and write to output_path.

    Parameters
    ----------
    video_path  : str          Path to original input video.
    kp_arr      : np.ndarray   Cleaned keypoints, shape (n_frames, n_kp, 2+).
    scorer_data : dict         Output of scorer.to_dict() — any sport.
    output_path : str          Destination .mp4 path.
    progress_cb : callable     Optional log callback.

    Returns
    -------
    str  output_path written (raises IOError on failure).
    """
    def _log(msg):
        if progress_cb:
            progress_cb(msg)

    sport   = scorer_data.get('sport', 'running').lower()
    backend = scorer_data.get('backend', 'rtmpose')
    fps     = float(scorer_data.get('fps', 30))
    n_kps   = kp_arr.shape[1]

    _log(f'[Annotator] sport={sport}  backend={backend}  '
         f'kp_arr={kp_arr.shape}  fps={fps:.1f}')

    if sport == 'running':
        event_map = _build_event_map_running(scorer_data)
    elif sport == 'rowing':
        event_map = _build_event_map_rowing(scorer_data)
    elif sport == 'cycling':
        event_map = _build_event_map_cycling(scorer_data)
    else:
        raise ValueError(f'[Annotator] Unknown sport: {sport!r}')

    _log(f'[Annotator] {len(event_map)} event frames identified.')

    skeleton, idx_map = _resolve_skeleton(backend)
    side_prefix       = _near_side_prefix(scorer_data)

    _log(f'[Annotator] skeleton={len(skeleton)} connections  '
         f'idx_map={len(idx_map)} joints  side={side_prefix}')

    if not skeleton:
        _log('[Annotator] WARNING: no skeleton connections resolved.')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f'[Annotator] Cannot open video: {video_path}')

    orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps  = cap.get(cv2.CAP_PROP_FPS) or fps

    _log(f'[Annotator] Input: {orig_w}x{orig_h}  {n_frames} frames  {vid_fps:.1f} fps')

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, vid_fps, (orig_w, orig_h))
    if not writer.isOpened():
        cap.release()
        raise IOError(f'[Annotator] Cannot open VideoWriter: {output_path}')

    n_kp_frames = len(kp_arr)
    written     = 0
    log_every   = max(1, n_frames // 20)

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        if fi < n_kp_frames and skeleton:
            _draw_skeleton(frame, kp_arr[fi], skeleton, n_kps)

        if fi in event_map:
            ev = event_map[fi]
            _draw_event_strip(frame, ev['label'], ev['colour'])
            if fi < n_kp_frames and ev.get('joint'):
                pt = _resolve_near_joint(
                    kp_arr[fi], idx_map, ev['joint'], side_prefix, n_kps)
                if pt is not None:
                    hh, ww = frame.shape[:2]
                    if 0 <= pt[0] < ww and 0 <= pt[1] < hh:
                        _draw_highlight_dot(frame, pt, ev['colour'])

        _draw_frame_counter(frame, fi, vid_fps)
        writer.write(frame)
        written += 1

        if fi % log_every == 0:
            _log(f'[Annotator]   {int(fi/max(n_frames,1)*100):3d}%  '
                 f'frame {fi}/{n_frames}')

    cap.release()
    writer.release()
    _log(f'[Annotator] ✓  Written {written} frames → {output_path}')
    return output_path
