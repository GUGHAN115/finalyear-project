"""
inference.py — Unified inference engine for image and video deepfake detection.

Modes:
  • Image  : Single-frame detection with bounding-box visualisation.
  • Video  : Equidistant frame sampling + temporal aggregation
             (mean fake-probability + peak anomaly tracking).
"""

import os
import cv2
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.model import DeepfakeDetector
from src.preprocessor import preprocess_image, annotate_image


# ── Device selection ──────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(
    checkpoint_path: Optional[str] = None,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[DeepfakeDetector, torch.device]:
    """
    Instantiate the DeepfakeDetector and optionally load a checkpoint.

    Args:
        checkpoint_path : Path to a .pt / .pth state-dict checkpoint.
        pretrained      : Use ImageNet weights for the ResNet backbone.
        device          : Target device (auto-detected if None).

    Returns:
        (model, device)
    """
    if device is None:
        device = get_device()

    model = DeepfakeDetector(pretrained=pretrained)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        # Accept both raw state-dicts and wrapped checkpoints
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model, device


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def infer_image(
    image_bgr: np.ndarray,
    model: DeepfakeDetector,
    device: torch.device,
    margin: float = 0.15,
) -> Dict:
    """
    Run deepfake detection on a single BGR image.

    Args:
        image_bgr : OpenCV BGR image array.
        model     : Loaded DeepfakeDetector in eval mode.
        device    : Torch device.
        margin    : Face bounding-box expansion fraction.

    Returns dict with keys:
        label          : "REAL" or "FAKE"
        confidence     : float in [0, 1]
        prob_real      : float — P(REAL)
        prob_fake      : float — P(FAKE)
        bbox           : (x1, y1, x2, y2) or None
        annotated_bgr  : Annotated image array (BGR) or None
        face_crop_rgb  : (224,224,3) RGB face crop or None
        forensic_vector: (1024,) numpy array — fused feature embedding
        face_detected  : bool
    """
    tensor, face_rgb, bbox = preprocess_image(image_bgr, margin=margin)

    if tensor is None:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "prob_real": 0.0,
            "prob_fake": 0.0,
            "bbox": None,
            "annotated_bgr": None,
            "face_crop_rgb": None,
            "forensic_vector": np.zeros(1024),
            "face_detected": False,
        }

    tensor = tensor.to(device)

    with torch.no_grad():
        logits, probs, forensic = model(tensor)

    probs_np   = probs.cpu().numpy()[0]          # [p_real, p_fake]
    class_idx  = int(probs_np.argmax())
    label      = DeepfakeDetector.class_label(class_idx)
    confidence = float(probs_np[class_idx])

    annotated = annotate_image(image_bgr, bbox, label, confidence)

    return {
        "label": label,
        "confidence": confidence,
        "prob_real": float(probs_np[0]),
        "prob_fake": float(probs_np[1]),
        "bbox": bbox,
        "annotated_bgr": annotated,
        "face_crop_rgb": face_rgb,
        "forensic_vector": forensic.cpu().numpy()[0],
        "face_detected": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO INFERENCE — equidistant sampling + temporal aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def sample_frames(
    video_path: str,
    n_frames: int = 32,
) -> Tuple[List[np.ndarray], float, int]:
    """
    Extract `n_frames` equidistant frames from a video file.

    Returns:
        frames  : list of BGR numpy arrays
        fps     : original video frame rate
        total   : total frame count in the video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total <= 0:
        cap.release()
        raise ValueError(f"Video has 0 frames: {video_path}")

    # Compute equidistant sample indices (clamped)
    n_sample = min(n_frames, total)
    indices  = np.linspace(0, total - 1, n_sample, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    return frames, fps, total


def infer_video(
    video_path: str,
    model: DeepfakeDetector,
    device: torch.device,
    n_frames: int = 32,
    margin: float = 0.15,
) -> Dict:
    """
    Run temporal deepfake analysis on a video file.

    Args:
        video_path : Path to the input video.
        model      : Loaded DeepfakeDetector in eval mode.
        device     : Torch device.
        n_frames   : Number of frames to sample.
        margin     : Face bounding-box expansion fraction.

    Returns dict with keys:
        verdict           : "REAL" or "FAKE" — majority temporal vote
        mean_fake_prob    : float — mean P(FAKE) across sampled frames
        peak_fake_prob    : float — max P(FAKE) across any single frame
        peak_frame_index  : int   — sampled-frame index of the peak anomaly
        frame_results     : list of per-frame dicts (label, confidence, prob_fake)
        faces_detected    : int   — number of frames where a face was found
        total_sampled     : int   — number of frames sampled
        video_fps         : float
        video_total_frames: int
    """
    frames, fps, total = sample_frames(video_path, n_frames)

    frame_results: List[Dict] = []
    fake_probs:   List[float] = []
    faces_detected = 0

    for i, frame in enumerate(frames):
        result = infer_image(frame, model, device, margin)
        frame_results.append({
            "frame_index": i,
            "label":       result["label"],
            "confidence":  result["confidence"],
            "prob_fake":   result["prob_fake"],
            "prob_real":   result["prob_real"],
            "face_detected": result["face_detected"],
        })
        if result["face_detected"]:
            faces_detected += 1
            fake_probs.append(result["prob_fake"])

    # ── Temporal aggregation ───────────────────────────────────────────────
    if fake_probs:
        mean_fake = float(np.mean(fake_probs))
        peak_fake = float(np.max(fake_probs))
        peak_idx  = int(np.argmax(fake_probs))
    else:
        mean_fake = 0.0
        peak_fake = 0.0
        peak_idx  = 0

    verdict = "FAKE" if mean_fake >= 0.5 else "REAL"

    return {
        "verdict":            verdict,
        "mean_fake_prob":     mean_fake,
        "peak_fake_prob":     peak_fake,
        "peak_frame_index":   peak_idx,
        "frame_results":      frame_results,
        "faces_detected":     faces_detected,
        "total_sampled":      len(frames),
        "video_fps":          fps,
        "video_total_frames": total,
    }
