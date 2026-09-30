"""Extracts per-frame 2D keypoints from video using YOLOv8-pose, RTMPose/Halpe26 or SAM-3D-Body."""

import os
import sys
import contextlib
import io
import warnings
import numpy as np

if not hasattr(np, 'core') or not hasattr(np.core, 'multiarray'):
    import types
    if not hasattr(np, 'core'):
        np.core = types.ModuleType('numpy.core')
        sys.modules['numpy.core'] = np.core
    if not hasattr(np.core, 'multiarray'):
        np.core.multiarray = types.ModuleType('numpy.core.multiarray')
        sys.modules['numpy.core.multiarray'] = np.core.multiarray
    for _attr in ('_reconstruct', 'scalar', 'ndarray'):
        if not hasattr(np.core.multiarray, _attr):
            setattr(np.core.multiarray, _attr, getattr(np, _attr, None))

import cv2

warnings.filterwarnings("ignore")

# Optional SAM-3D-Body backend: set the SAM3D_ROOT environment variable to the
# folder containing a local sam-3d-body checkout. If unset or missing, the
# backend is simply disabled and YOLOv8 / RTMPose are used instead.
SAM3D_ROOT     = os.environ.get("SAM3D_ROOT", "")
NOTEBOOK_DIR   = os.path.join(SAM3D_ROOT, "notebook")
SAM3D_AVAILABLE = bool(SAM3D_ROOT) and os.path.isdir(SAM3D_ROOT)

_yolo_model        = None
_rtm_model         = None
_sam3d_model       = None
_sam3d_name_to_idx = None
_sam3d_connections = None


def get_sam3d_name_to_idx():
    """Return the MHR70 name→index dict (None if SAM-3D not yet loaded)."""
    return _sam3d_name_to_idx

def get_sam3d_connections():
    """Return native MHR70 skeleton connections (None if SAM-3D not yet loaded)."""
    return _sam3d_connections


def _load_yolo(log):
    global _yolo_model
    if _yolo_model is None:
        log("  Loading YOLOv8-pose (yolov8x-pose.pt)...")
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8x-pose.pt")
        log("  YOLOv8 ready.")
    return _yolo_model


def _load_rtmpose(log):
    global _rtm_model
    if _rtm_model is None:
        log("  Loading RTMPose-M (Halpe26 / ONNX)...")
        from rtmlib import Custom
        _rtm_model = Custom(
            det_class="YOLOX",
            det=(
                "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
            ),
            det_input_size=(640, 640),
            pose_class="RTMPose",
            pose=(
                "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip"
            ),
            pose_input_size=(192, 256),
            backend="onnxruntime",
            device="cpu",
        )
        log("  RTMPose ready.")
    return _rtm_model


def _load_sam3d(log):
    global _sam3d_model, _sam3d_name_to_idx, _sam3d_connections
    if not SAM3D_AVAILABLE:
        raise RuntimeError(
            f"SAM-3D-Body is not available on this machine "
            f"(expected directory: {SAM3D_ROOT!r}). "
            "Please select YOLOv8 or RTMPose."
        )
    if _sam3d_model is None:
        log("  Loading SAM-3D-Body (may take ~30s on first run)...")

        os.environ["HUGGINGFACE_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]  = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS"]          = "1"
        os.environ["PYOPENGL_PLATFORM"]                = "win32"

        sys.path.insert(0, SAM3D_ROOT)
        sys.path.insert(0, NOTEBOOK_DIR)
        os.chdir(NOTEBOOK_DIR)

        from utils import setup_sam_3d_body
        from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

        with contextlib.redirect_stderr(io.StringIO()):
            _sam3d_model = setup_sam_3d_body(
                hf_repo_id="facebook/sam-3d-body-dinov3",
                detector_name=None,
                segmentor_name=None,
                fov_name=None,
            )

        _sam3d_name_to_idx = {
            v["name"]: k for k, v in mhr70_pose_info["keypoint_info"].items()
        }

        _sam3d_connections = []
        for link in mhr70_pose_info["skeleton_info"].values():
            n1, n2 = link["link"][0], link["link"][1]
            if n1 in _sam3d_name_to_idx and n2 in _sam3d_name_to_idx:
                rgb = link["color"]
                bgr = (rgb[2], rgb[1], rgb[0])
                _sam3d_connections.append(
                    (_sam3d_name_to_idx[n1], _sam3d_name_to_idx[n2], bgr)
                )

        log(f"  SAM-3D ready. {len(_sam3d_name_to_idx)} joints, "
            f"{len(_sam3d_connections)} skeleton connections.")
    return _sam3d_model


def _run_yolo(cap, process_up, every_n, log):
    model = _load_yolo(log)
    kps_list, frame_idx = [], []
    fi = processed = 0

    while True:
        ret, frame = cap.read()
        if not ret or fi >= process_up:
            break
        if fi % every_n == 0:
            try:
                res = model(frame, verbose=False)
                if res and len(res[0].keypoints) > 0:
                    kps  = res[0].keypoints.xy[0].cpu().numpy()
                    conf = res[0].keypoints.conf[0].cpu().numpy()
                    kps_list.append(np.column_stack([kps, conf]))
                else:
                    kps_list.append(np.full((17, 3), np.nan))
            except Exception:
                kps_list.append(np.full((17, 3), np.nan))
            frame_idx.append(fi)
            processed += 1
            if processed % 50 == 0:
                log(f"  YOLOv8: {processed} frames processed...")
        fi += 1

    log(f"  YOLOv8 done: {processed} frames.")
    return kps_list, frame_idx


def _run_rtmpose(cap, process_up, every_n, log):
    model = _load_rtmpose(log)
    kps_list, frame_idx = [], []
    fi = processed = 0

    while True:
        ret, frame = cap.read()
        if not ret or fi >= process_up:
            break
        if fi % every_n == 0:
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                keypoints, scores = model(img_rgb)
                if keypoints is not None and len(keypoints) > 0:
                    kps_list.append(np.column_stack([keypoints[0], scores[0]]))
                else:
                    kps_list.append(np.full((26, 3), np.nan))
            except Exception:
                kps_list.append(np.full((26, 3), np.nan))
            frame_idx.append(fi)
            processed += 1
            if processed % 50 == 0:
                log(f"  RTMPose: {processed} frames processed...")
        fi += 1

    log(f"  RTMPose done: {processed} frames.")
    return kps_list, frame_idx


def _run_sam3d(cap, process_up, every_n, log):
    model = _load_sam3d(log)
    kps_list, frame_idx = [], []
    fi = processed = 0

    while True:
        ret, frame = cap.read()
        if not ret or fi >= process_up:
            break
        if fi % every_n == 0:
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                outputs = model.process_one_image(img_rgb)
                if outputs:
                    kps_2d = np.array(outputs[0]["pred_keypoints_2d"])
                    if kps_2d.ndim == 2 and kps_2d.shape[1] == 2:
                        kps_2d = np.column_stack([kps_2d, np.ones(len(kps_2d))])
                    kps_list.append(kps_2d)
                else:
                    kps_list.append(np.full((70, 3), np.nan))
            except Exception:
                kps_list.append(np.full((70, 3), np.nan))
            frame_idx.append(fi)
            processed += 1
            if processed % 10 == 0:
                log(f"  SAM-3D: {processed} frames processed...")
        fi += 1

    log(f"  SAM-3D done: {processed} frames.")
    return kps_list, frame_idx


def extract_keypoints(video_path: str, model_name: str,
                      progress_cb=None, max_frames: int = None, every_n: int = 1):
    """
    Extract pose keypoints from a video using the selected model.

    Parameters
    ----------
    video_path   : str   Path to input video.
    model_name   : str   'YOLOv8' | 'RTMPose' | 'SAM-3D-Body'
    progress_cb  : callable(str)  Optional log callback.
    max_frames   : int | None     Limit frames processed.
    every_n      : int            Process every Nth frame (default 1 = all).

    Returns
    -------
    kps_array    : np.ndarray  (n_processed, n_kp, 3)
    fps          : float
    backend      : str         'yolov8' | 'rtmpose' | 'sam3d'
    frame_indices: list[int]
    """
    log = progress_cb or print

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    orig_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_f     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    process_up = min(max_frames, total_f) if max_frames else total_f
    log(f"  {os.path.basename(video_path)}  |  {total_f} frames  |  {orig_fps:.1f} fps  |  processing {process_up}")

    mn = model_name.upper()

    if "YOLO" in mn:
        kps_list, fidx = _run_yolo(cap, process_up, every_n, log)
        cap.release()
        kps_array = np.stack(kps_list) if kps_list else np.empty((0, 17, 3))
        return kps_array, orig_fps, "yolov8", fidx

    elif "RTM" in mn:
        kps_list, fidx = _run_rtmpose(cap, process_up, every_n, log)
        cap.release()
        kps_array = np.stack(kps_list) if kps_list else np.empty((0, 26, 3))
        return kps_array, orig_fps, "rtmpose", fidx

    elif "SAM" in mn:
        kps_list, fidx = _run_sam3d(cap, process_up, every_n, log)
        cap.release()
        kps_array = np.stack(kps_list) if kps_list else np.empty((0, 70, 3))
        return kps_array, orig_fps, "sam3d", fidx

    else:
        cap.release()
        raise ValueError(f"Unknown model: {model_name!r}. "
                         f"Expected 'YOLOv8', 'RTMPose', or 'SAM-3D-Body'.")
