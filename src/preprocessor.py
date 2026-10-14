"""
preprocessor.py — OpenCV-based face detection and preprocessing pipeline.

Pipeline:
  1. Load image (BGR -> RGB)
  2. Detect face ROI using Haar Cascade (frontalface_default)
  3. Expand bounding box by 15% margin on all sides
  4. Crop and resize face ROI to 224x224
  5. Normalise with ImageNet channel mean/std
  6. Return PyTorch tensor and annotated image for visualisation
"""

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Optional, Tuple


# ── ImageNet normalisation constants ──────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Haar Cascade XML bundled with OpenCV ──────────────────────────────────────
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def _load_cascade() -> cv2.CascadeClassifier:
    """Load and validate the Haar Cascade classifier."""
    if not os.path.exists(_CASCADE_PATH):
        raise FileNotFoundError(
            f"Haar Cascade XML not found at: {_CASCADE_PATH}. "
            "Please ensure OpenCV is correctly installed."
        )
    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier.")
    return cascade


# Global singleton — loaded once per process
_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None


def get_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = _load_cascade()
    return _FACE_CASCADE


# ── Torchvision transforms ────────────────────────────────────────────────────
_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def expand_bbox(x: int, y: int, w: int, h: int,
                margin: float, img_h: int, img_w: int
                ) -> Tuple[int, int, int, int]:
    """
    Expand bounding box by `margin` fraction (e.g. 0.15 = 15%) on all sides,
    clamped to image boundaries.

    Returns (x1, y1, x2, y2) in pixel coordinates.
    """
    dx = int(w * margin)
    dy = int(h * margin)
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(img_w, x + w + dx)
    y2 = min(img_h, y + h + dy)
    return x1, y1, x2, y2


def detect_and_crop_face(
    image_bgr: np.ndarray,
    margin: float = 0.15,
    target_size: int = 224,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (60, 60),
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Detect the largest face in a BGR image, expand bounding box by `margin`,
    and return a cropped RGB patch resized to `target_size x target_size`.

    Args:
        image_bgr  : OpenCV BGR image (H x W x 3 uint8).
        margin     : Fractional padding around the detected bounding box.
        target_size: Output spatial size (pixels).
        scale_factor, min_neighbors, min_size: Haar Cascade hyperparameters.

    Returns:
        face_rgb   : (target_size, target_size, 3) RGB crop, or None.
        bbox       : (x1, y1, x2, y2) expanded bounding box, or None.
    """
    cascade = get_cascade()
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_h, img_w = image_bgr.shape[:2]

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    if len(faces) == 0:
        return None, None

    # Select the largest detected face
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = faces[0]

    x1, y1, x2, y2 = expand_bbox(x, y, w, h, margin, img_h, img_w)

    # Crop and resize
    crop_bgr = image_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, (target_size, target_size),
                          interpolation=cv2.INTER_AREA)

    return crop_rgb, (x1, y1, x2, y2)


def preprocess_image(
    image_bgr: np.ndarray,
    margin: float = 0.15,
    target_size: int = 224,
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Full preprocessing pipeline for a single image.

    Args:
        image_bgr : OpenCV BGR image.
        margin    : Bounding-box expansion fraction.
        target_size: Face crop size before normalisation.

    Returns:
        tensor      : (1, 3, 224, 224) normalised float tensor ready for the model.
        face_rgb    : (224, 224, 3) uint8 RGB crop (for display / visualisation).
        bbox        : (x1, y1, x2, y2) expanded bounding box in original coords.
                      All values are None if no face is detected.
    """
    face_rgb, bbox = detect_and_crop_face(image_bgr, margin, target_size)

    if face_rgb is None:
        return None, None, None

    tensor = _to_tensor(face_rgb).unsqueeze(0)   # (1, 3, H, W)
    return tensor, face_rgb, bbox


def load_image_bgr(path: str) -> np.ndarray:
    """Read an image file into BGR format via OpenCV."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def annotate_image(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    color_real: Tuple[int, int, int] = (0, 200, 80),
    color_fake: Tuple[int, int, int] = (0, 60, 220),
) -> np.ndarray:
    """
    Draw bounding box and confidence label on the original image.

    Returns a BGR annotated copy (suitable for cv2.imwrite / st.image).
    """
    annotated = image_bgr.copy()
    x1, y1, x2, y2 = bbox

    color = color_fake if label == "FAKE" else color_real
    thickness = 3

    # Bounding box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

    # Confidence badge background
    badge_text = f"{label}  {confidence * 100:.1f}%"
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.75
    (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, 2)
    badge_y1 = max(y1 - th - baseline - 8, 0)
    badge_y2 = badge_y1 + th + baseline + 8
    cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, -1)
    cv2.putText(annotated, badge_text,
                (x1 + 5, badge_y2 - baseline - 2),
                font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated
