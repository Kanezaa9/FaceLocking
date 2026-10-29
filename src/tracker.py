from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .haar_5pt import FaceKpsBox
from .recognize import MatchResult
from .yolox import BodyDetection
from .body_features import BodyFeatures, BodyFeatureExtractor


@dataclass
class TrackedBody:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    missed_frames: int = 0
    features: Optional[BodyFeatures] = None
    face: Optional[FaceKpsBox] = None
    face_match: Optional[MatchResult] = None

    @property
    def box_int(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


def box_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area1 = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area2 = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area1 + area2 - inter
    return inter / union


class SimpleBodyTracker:
    def __init__(self, iou_thresh: float = 0.35, max_lost: int = 18, ema_alpha: float = 0.75):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.ema_alpha = ema_alpha
        self.tracks: Dict[int, TrackedBody] = {}
        self.next_id = 1
        self.feature_extractor = BodyFeatureExtractor()

    def update(
        self,
        frame: np.ndarray,
        detections: List[BodyDetection],
        faces: List[Tuple[FaceKpsBox, MatchResult]],
    ) -> List[TrackedBody]:
        h, w = frame.shape[:2]

        matched_tracks = set()
        matched_dets = set()

        if self.tracks and detections:
            cost_matrix = []
            track_ids = list(self.tracks.keys())
            for tid in track_ids:
                tb = self.tracks[tid]
                t_box = (tb.x1, tb.y1, tb.x2, tb.y2)
                row = []
                for det in detections:
                    d_box = (float(det.x1), float(det.y1), float(det.x2), float(det.y2))
                    iou_val = box_iou(t_box, d_box)
                    dist = np.hypot(tb.center[0] - det.center[0], tb.center[1] - det.center[1])
                    norm_dist = dist / max(w, h)
                    score = iou_val - 0.2 * norm_dist
                    row.append(score)
                cost_matrix.append(row)

            cost_arr = np.array(cost_matrix)
            while True:
                max_val = np.max(cost_arr)
                if max_val < 0.15:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(cost_arr), cost_arr.shape)
                cost_arr[t_idx, :] = -999.0
                cost_arr[:, d_idx] = -999.0
                tid = track_ids[t_idx]
                det = detections[d_idx]

                tb = self.tracks[tid]
                tb.x1 = self.ema_alpha * tb.x1 + (1.0 - self.ema_alpha) * det.x1
                tb.y1 = self.ema_alpha * tb.y1 + (1.0 - self.ema_alpha) * det.y1
                tb.x2 = self.ema_alpha * tb.x2 + (1.0 - self.ema_alpha) * det.x2
                tb.y2 = self.ema_alpha * tb.y2 + (1.0 - self.ema_alpha) * det.y2
                tb.score = det.score
                tb.missed_frames = 0
                matched_tracks.add(tid)
                matched_dets.add(d_idx)

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = TrackedBody(
                    track_id=tid,
                    x1=float(det.x1),
                    y1=float(det.y1),
                    x2=float(det.x2),
                    y2=float(det.y2),
                    score=det.score,
                    missed_frames=0,
                )
                matched_tracks.add(tid)

        to_delete = []
        for tid, tb in self.tracks.items():
            if tid not in matched_tracks:
                tb.missed_frames += 1
                if tb.missed_frames > self.max_lost:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        for tb in self.tracks.values():
            tb.face = None
            tb.face_match = None

            bx1, by1, bx2, by2 = tb.box_int
            bx1_c = max(0, min(w - 1, bx1))
            by1_c = max(0, min(h - 1, by1))
            bx2_c = max(bx1_c + 1, min(w, bx2))
            by2_c = max(by1_c + 1, min(h, by2))

            crop = frame[by1_c:by2_c, bx1_c:bx2_c]
            # from_rear is set True when no face is visible; updated after face association
            tb.features = self.feature_extractor.extract(crop, from_rear=True)

        used_faces = set()
        for tb in self.tracks.values():
            bx1, by1, bx2, by2 = tb.box_int
            bh = max(1.0, float(by2 - by1))
            best_face = None
            best_match = None
            best_dist = 999999.0
            best_idx = -1

            for f_idx, (f, m) in enumerate(faces):
                if f_idx in used_faces:
                    continue
                fcx = (f.x1 + f.x2) / 2.0
                fcy = (f.y1 + f.y2) / 2.0

                if bx1 - 35 <= fcx <= bx2 + 35 and by1 - 30 <= fcy <= by1 + 0.75 * bh:
                    dist = np.hypot(fcx - tb.center[0], fcy - (by1 + 0.20 * bh))
                    if dist < best_dist:
                        best_dist = dist
                        best_face = f
                        best_match = m
                        best_idx = f_idx

            if best_face is not None:
                tb.face = best_face
                tb.face_match = best_match
                used_faces.add(best_idx)
                # mark features as front-view now that a face is confirmed
                if tb.features is not None:
                    tb.features.from_rear = False

        return list(self.tracks.values())
