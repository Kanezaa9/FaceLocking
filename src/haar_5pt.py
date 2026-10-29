# src/haar_5pt.py
"""
Haar face detection + 5-point landmarks (MediaPipe FaceMesh) + alignment.
This is the shared core module used by every later stage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mediapipe as mp


@dataclass
class FaceKpsBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32
    mesh: Optional[np.ndarray] = None  # (N,2) float32 full FaceMesh landmarks in pixels (unsmoothed)


def _estimate_norm_5pt(kps_5x2: np.ndarray, out_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Computes the 2x3 affine matrix that maps your 5 points onto the
    standard ArcFace 112x112 template positions.
    """
    k = kps_5x2.astype(np.float32)

    # standard InsightFace/ArcFace reference positions for a 112x112 face
    dst = np.array([
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose
        [41.5493, 92.3655],   # left mouth
        [70.7299, 92.2041],   # right mouth
    ], dtype=np.float32)

    out_w, out_h = int(out_size[0]), int(out_size[1])
    if (out_w, out_h) != (112, 112):
        sx = out_w / 112.0
        sy = out_h / 112.0
        dst = dst * np.array([sx, sy], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(k, dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(
            np.array([k[0], k[1], k[2]], dtype=np.float32),
            np.array([dst[0], dst[1], dst[2]], dtype=np.float32),
        )
    return M.astype(np.float32)


def align_face_5pt(
    frame_bgr: np.ndarray,
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warps frame_bgr so the 5 landmarks land on the canonical template.
    Returns (aligned_face, transform_matrix).
    """
    M = _estimate_norm_5pt(kps_5x2, out_size=out_size)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, M


def _clip_box_xyxy(b: np.ndarray, W: int, H: int) -> np.ndarray:
    bb = b.astype(np.float32).copy()
    bb[0] = np.clip(bb[0], 0, W - 1)
    bb[1] = np.clip(bb[1], 0, H - 1)
    bb[2] = np.clip(bb[2], 0, W - 1)
    bb[3] = np.clip(bb[3], 0, H - 1)
    return bb


def _bbox_from_5pt(kps: np.ndarray, pad_x: float = 0.55, pad_y_top: float = 0.85, pad_y_bot: float = 1.15) -> np.ndarray:
    """Builds a face-like box centered on the 5 keypoints (avoids Haar's 'off-center' box)."""
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average — smooths jitter between frames."""
    if prev is None:
        return cur.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float = 12.0) -> bool:
    """Sanity check: rejects obviously-wrong landmark detections."""
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < min_eye_dist:
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True


class Haar5ptDetector:
    def __init__(
        self,
        haar_xml: str = "models/haarcascade_frontalface_default.xml",
        min_size: Tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.80,
        debug: bool = True,
        max_mesh_faces: int = 1,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        self.smooth_alpha = float(smooth_alpha)

        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load cascade: {haar_xml}")

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=int(max_mesh_faces),
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = None
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            task_path = Path("models/face_landmarker.task")
            if not task_path.exists():
                raise FileNotFoundError(f"Missing model file: {task_path}")
            base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=int(max_mesh_faces),
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self.mp_face_mesh = None

        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

        self._prev_box: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE, minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32)

    def _facemesh_all(self, frame_bgr: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        idxs = [self.IDX_LEFT_EYE, self.IDX_RIGHT_EYE, self.IDX_NOSE_TIP, self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT]
        out = []

        if self.mp_face_mesh is not None:
            res = self.mp_face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                return []
            for face in res.multi_face_landmarks:
                mesh = np.array([[p.x * W, p.y * H] for p in face.landmark], dtype=np.float32)
                kps = mesh[idxs].copy()
                if kps[0, 0] > kps[1, 0]:
                    kps[[0, 1]] = kps[[1, 0]]
                if kps[3, 0] > kps[4, 0]:
                    kps[[3, 4]] = kps[[4, 3]]
                out.append((kps, mesh))
        elif self.landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_image)
            if not res.face_landmarks:
                return []
            for face in res.face_landmarks:
                mesh = np.array([[p.x * W, p.y * H] for p in face], dtype=np.float32)
                kps = mesh[idxs].copy()
                if kps[0, 0] > kps[1, 0]:
                    kps[[0, 1]] = kps[[1, 0]]
                if kps[3, 0] > kps[4, 0]:
                    kps[[3, 4]] = kps[[4, 3]]
                out.append((kps, mesh))
        return out

    def _facemesh_5pt(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        faces = self._facemesh_all(frame_bgr)
        return faces[0] if faces else (None, None)

    @staticmethod
    def _inside_frac(kps: np.ndarray, x: int, y: int, w: int, h: int, margin: float = 0.35) -> float:
        """Fraction of keypoints inside the Haar box padded by `margin` on each side."""
        x1m, y1m = x - margin * w, y - margin * h
        x2m, y2m = x + (1.0 + margin) * w, y + (1.0 + margin) * h
        inside = (
            (kps[:, 0] >= x1m) & (kps[:, 0] <= x2m) &
            (kps[:, 1] >= y1m) & (kps[:, 1] <= y2m)
        )
        return float(inside.mean())

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 1) -> List[FaceKpsBox]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_faces(gray)
        if faces.shape[0] == 0:
            return []

        areas = faces[:, 2] * faces[:, 3]
        i = int(np.argmax(areas))
        x, y, w, h = faces[i].tolist()

        kps, mesh = self._facemesh_5pt(frame_bgr)
        if kps is None:
            if self.debug:
                print("[haar_5pt] Haar face found but FaceMesh returned none -> reject")
            return []

        if self._inside_frac(kps, x, y, w, h) < 0.60:
            if self.debug:
                print("[haar_5pt] FaceMesh points not consistent with Haar box -> reject")
            return []

        if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * w)):
            if self.debug:
                print("[haar_5pt] 5pt geometry sanity failed -> reject")
            return []

        box = _bbox_from_5pt(kps)
        box = _clip_box_xyxy(box, W, H)

        box_s = _ema(self._prev_box, box, self.smooth_alpha)
        kps_s = _ema(self._prev_kps, kps, self.smooth_alpha)
        self._prev_box = box_s.copy()
        self._prev_kps = kps_s.copy()

        x1, y1, x2, y2 = box_s.tolist()

        return [
            FaceKpsBox(
                x1=int(round(x1)), y1=int(round(y1)),
                x2=int(round(x2)), y2=int(round(y2)),
                score=1.0, kps=kps_s.astype(np.float32), mesh=mesh,
            )
        ][:max_faces]

    def detect_all(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceKpsBox]:
        """
        Every FaceMesh face that is confirmed by its own Haar box (same checks as
        detect()). Needs max_mesh_faces > 1 to see more than one face. Results are
        NOT smoothed, since there is no per-face identity to smooth across frames.
        """
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        haar = self._haar_faces(gray).tolist()
        if not haar:
            return []

        used = set()
        out: List[FaceKpsBox] = []
        for kps, mesh in self._facemesh_all(frame_bgr):
            best_j, best_frac = -1, 0.60
            for j, (x, y, w, h) in enumerate(haar):
                if j in used:
                    continue
                frac = self._inside_frac(kps, x, y, w, h)
                if frac >= best_frac and _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * w)):
                    best_j, best_frac = j, frac
            if best_j < 0:
                if self.debug:
                    print("[haar_5pt] FaceMesh face without a consistent Haar box -> reject")
                continue
            used.add(best_j)
            x1, y1, x2, y2 = _clip_box_xyxy(_bbox_from_5pt(kps), W, H).tolist()
            out.append(FaceKpsBox(
                x1=int(round(x1)), y1=int(round(y1)),
                x2=int(round(x2)), y2=int(round(y2)),
                score=1.0, kps=kps.astype(np.float32), mesh=mesh,
            ))
        return out[:max_faces]
