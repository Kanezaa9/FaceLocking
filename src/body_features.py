from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BodyFeatures:
    hsv_hists: List[np.ndarray]       # per-zone HSV hists (front or rear)
    lab_moments: List[np.ndarray]     # per-zone LAB moments
    dominant_upper_rgb: Tuple[int, int, int]
    dominant_lower_rgb: Tuple[int, int, int]
    dominant_upper_lab: np.ndarray
    dominant_lower_lab: np.ndarray
    texture_energy: np.ndarray
    aspect_ratio: float
    feature_vector: np.ndarray        # full L2-normalised descriptor
    color_vec: np.ndarray             # compact color-only descriptor (robust across orientations)
    from_rear: bool = False           # True when extracted without a visible face


class BodyFeatureExtractor:
    """
    Extracts clothing-based body appearance features.

    Two descriptor levels:
      feature_vector  – full descriptor (histograms + LAB moments + texture + color)
                        used for front-to-front and rear-to-rear matching
      color_vec       – compact HSV + dominant-color descriptor, orientation-agnostic,
                        used for cross-orientation matching (front gallery vs rear query)
    """

    def __init__(self, h_bins: int = 24, s_bins: int = 8):
        self.h_bins = h_bins
        self.s_bins = s_bins

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _zone_mask(self, h: int, w: int, y0: float, y1: float,
                   x0: float = 0.05, x1: float = 0.95) -> np.ndarray:
        """Rectangular mask clipped to [x0,x1] × [y0,y1] fractions of the crop."""
        mask = np.zeros((h, w), dtype=np.uint8)
        r0 = max(0, int(h * y0))
        r1 = min(h, int(h * y1))
        c0 = max(0, int(w * x0))
        c1 = min(w, int(w * x1))
        if r1 > r0 and c1 > c0:
            mask[r0:r1, c0:c1] = 255
        return mask

    def _dominant_color(self, img_bgr: np.ndarray,
                        mask: np.ndarray) -> Tuple[Tuple[int, int, int], np.ndarray]:
        valid = img_bgr[mask > 0]
        if len(valid) < 10:
            return (128, 128, 128), np.array([50.0, 0.0, 0.0], dtype=np.float32)
        data = valid.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
        k = min(3, max(1, len(data) // 20))
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        counts = np.bincount(labels.flatten(), minlength=k)
        best_bgr = centers[int(np.argmax(counts))].astype(np.uint8)
        rgb = (int(best_bgr[2]), int(best_bgr[1]), int(best_bgr[0]))
        lab_px = cv2.cvtColor(np.uint8([[best_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        return rgb, lab_px

    def _hsv_hist(self, hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist(
            [hsv], [0, 1], mask,
            [self.h_bins, self.s_bins],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
        return hist.flatten().astype(np.float32)

    def _lab_moments(self, lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pts = lab[mask > 0]
        if len(pts) > 10:
            return np.concatenate(
                [np.mean(pts, axis=0), np.std(pts, axis=0)]
            ).astype(np.float32)
        return np.array([128.0, 128.0, 128.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def extract(self, body_crop: np.ndarray,
                from_rear: bool = False) -> Optional[BodyFeatures]:
        if body_crop is None or body_crop.size == 0:
            return None
        h, w = body_crop.shape[:2]
        if h < 40 or w < 20:
            return None

        aspect_ratio = float(w) / float(h)
        hsv = cv2.cvtColor(body_crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(body_crop, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY)

        # 6 horizontal zones spanning the full width – orientation-agnostic
        zone_defs = [
            (0.05, 0.30),   # head/shoulder region
            (0.25, 0.50),   # upper torso
            (0.45, 0.68),   # mid torso
            (0.65, 0.85),   # hip/upper leg
            (0.82, 1.00),   # lower leg
            (0.10, 0.90),   # full body
        ]

        hsv_hists: List[np.ndarray] = []
        lab_moments_list: List[np.ndarray] = []

        for y0, y1 in zone_defs:
            m = self._zone_mask(h, w, y0, y1)
            hsv_hists.append(self._hsv_hist(hsv, m))
            lab_moments_list.append(self._lab_moments(lab, m))

        # dominant color in upper/lower halves (used for cross-orientation matching)
        m_upper = self._zone_mask(h, w, 0.15, 0.48)
        m_lower = self._zone_mask(h, w, 0.50, 0.90)
        rgb_upper, lab_upper = self._dominant_color(body_crop, m_upper)
        rgb_lower, lab_lower = self._dominant_color(body_crop, m_lower)

        # texture energy (Sobel gradient magnitude in upper + lower halves)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(sobel_x, sobel_y)
        tex_vals: List[float] = []
        for m in [m_upper, m_lower]:
            pts = grad[m > 0]
            if len(pts) > 10:
                tex_vals += [float(np.mean(pts)), float(np.std(pts))]
            else:
                tex_vals += [0.0, 0.0]
        texture_energy = np.array(tex_vals, dtype=np.float32)

        # ---- full feature vector ----
        raw = np.concatenate([
            np.concatenate(hsv_hists),          # 6 × (h_bins × s_bins)
            np.concatenate(lab_moments_list) / 255.0,
            lab_upper / 255.0,
            lab_lower / 255.0,
            texture_energy / 255.0,
            np.array([np.clip(aspect_ratio, 0.1, 2.0)], dtype=np.float32),
        ]).astype(np.float32)
        norm = float(np.linalg.norm(raw) + 1e-12)
        feature_vector = (raw / norm).astype(np.float32)

        # ---- compact color-only vector (orientation-agnostic) ----
        # Just the 6-zone HSV histograms + dominant upper/lower LAB
        color_raw = np.concatenate([
            np.concatenate(hsv_hists),
            lab_upper / 255.0,
            lab_lower / 255.0,
        ]).astype(np.float32)
        cnorm = float(np.linalg.norm(color_raw) + 1e-12)
        color_vec = (color_raw / cnorm).astype(np.float32)

        return BodyFeatures(
            hsv_hists=hsv_hists,
            lab_moments=lab_moments_list,
            dominant_upper_rgb=rgb_upper,
            dominant_lower_rgb=rgb_lower,
            dominant_upper_lab=lab_upper,
            dominant_lower_lab=lab_lower,
            texture_energy=texture_energy,
            aspect_ratio=aspect_ratio,
            feature_vector=feature_vector,
            color_vec=color_vec,
            from_rear=from_rear,
        )


# ----------------------------------------------------------------------
# similarity functions
# ----------------------------------------------------------------------

def _hist_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(np.sum(np.minimum(h1, h2)))


def compute_feature_similarity(a: BodyFeatures, b: BodyFeatures) -> float:
    """Full feature similarity – best when both are same orientation."""
    hist_score = float(np.mean([
        _hist_intersection(h1, h2)
        for h1, h2 in zip(a.hsv_hists, b.hsv_hists)
    ]))

    lab_score = float(np.mean([
        max(0.0, 1.0 - float(np.linalg.norm((m1[:3] - m2[:3]) / 255.0)))
        for m1, m2 in zip(a.lab_moments, b.lab_moments)
    ]))

    diff_up = np.linalg.norm((a.dominant_upper_lab - b.dominant_upper_lab) / 255.0)
    diff_lo = np.linalg.norm((a.dominant_lower_lab - b.dominant_lower_lab) / 255.0)
    color_score = max(0.0, 1.0 - 0.5 * float(diff_up + diff_lo))

    diff_tex = np.linalg.norm((a.texture_energy - b.texture_energy) / 255.0)
    tex_score = max(0.0, 1.0 - float(diff_tex))

    cos_score = max(0.0, float(np.dot(a.feature_vector, b.feature_vector)))

    return float(np.clip(
        0.35 * hist_score
        + 0.25 * lab_score
        + 0.20 * color_score
        + 0.05 * tex_score
        + 0.15 * cos_score,
        0.0, 1.0,
    ))


def compute_color_similarity(a: BodyFeatures, b: BodyFeatures) -> float:
    """
    Color-only similarity using the compact color_vec.
    More robust across front/rear orientation changes because it
    ignores texture and shape which differ between views.
    """
    return max(0.0, float(np.dot(a.color_vec, b.color_vec)))


def _best_sim(candidate: BodyFeatures,
              gallery: List[BodyFeatures],
              top_k: int = 5) -> float:
    """Mean of the top-k similarities, using full or color-only metric."""
    if not gallery:
        return 0.0

    scores = []
    for g in gallery:
        # if orientations differ, weight color heavily
        if candidate.from_rear != g.from_rear:
            s = 0.30 * compute_feature_similarity(candidate, g) \
              + 0.70 * compute_color_similarity(candidate, g)
        else:
            s = compute_feature_similarity(candidate, g)
        scores.append(s)

    top = sorted(scores, reverse=True)[:min(top_k, len(scores))]
    return float(np.mean(top))


# ----------------------------------------------------------------------
# BodyLearner
# ----------------------------------------------------------------------

class BodyLearner:
    """
    Maintains two galleries:
      front_gallery  – samples collected while the face was visible (front view)
      rear_gallery   – samples collected while facing away (no face)

    Matching a candidate checks both galleries and takes the best score,
    so a person returning from behind is matched against their rear gallery
    and a person returning face-first is matched against the front gallery.
    """

    def __init__(
        self,
        target_name: str,
        min_samples: int = 5,
        max_gallery_size: int = 30,
        match_threshold: float = 0.42,
        ema_alpha: float = 0.80,
    ):
        self.target_name = target_name
        self.min_samples = min_samples
        self.max_gallery_size = max_gallery_size
        self.match_threshold = match_threshold
        self.ema_alpha = ema_alpha

        self.front_gallery: List[BodyFeatures] = []
        self.rear_gallery: List[BodyFeatures] = []
        self.ema_feature: Optional[BodyFeatures] = None
        self.samples_collected = 0

    # kept for compatibility with body_lock.py which reads .gallery
    @property
    def gallery(self) -> List[BodyFeatures]:
        return self.front_gallery + self.rear_gallery

    @property
    def is_learned(self) -> bool:
        return self.samples_collected >= self.min_samples

    @property
    def progress(self) -> float:
        return min(1.0, self.samples_collected / self.min_samples)

    def reset(self) -> None:
        self.front_gallery.clear()
        self.rear_gallery.clear()
        self.ema_feature = None
        self.samples_collected = 0

    def _add_to_gallery(self, features: BodyFeatures,
                        gallery: List[BodyFeatures]) -> None:
        if len(gallery) < self.max_gallery_size:
            gallery.append(features)
        else:
            # replace the entry least similar to the new sample (keeps diversity)
            sims = [compute_feature_similarity(features, g) for g in gallery]
            idx = int(np.argmin(sims))
            if sims[idx] < 0.92:
                gallery[idx] = features

    def add_sample(self, features: BodyFeatures) -> None:
        self.samples_collected += 1
        if features.from_rear:
            self._add_to_gallery(features, self.rear_gallery)
        else:
            self._add_to_gallery(features, self.front_gallery)

        # update EMA feature vector
        if self.ema_feature is None:
            self.ema_feature = features
        else:
            v = (self.ema_alpha * self.ema_feature.feature_vector
                 + (1.0 - self.ema_alpha) * features.feature_vector)
            n = float(np.linalg.norm(v) + 1e-12)
            self.ema_feature.feature_vector = (v / n).astype(np.float32)
            self.ema_feature.dominant_upper_rgb = features.dominant_upper_rgb
            self.ema_feature.dominant_lower_rgb = features.dominant_lower_rgb

    def match(self, candidate: BodyFeatures) -> Tuple[float, bool]:
        if not self.front_gallery and not self.rear_gallery:
            return 0.0, False

        scores: List[float] = []

        # score against front gallery
        if self.front_gallery:
            scores.append(_best_sim(candidate, self.front_gallery))

        # score against rear gallery
        if self.rear_gallery:
            scores.append(_best_sim(candidate, self.rear_gallery))

        # EMA cosine (orientation-agnostic color vec)
        if self.ema_feature is not None:
            ema_color = compute_color_similarity(candidate, self.ema_feature)
            ema_full = max(0.0, float(
                np.dot(candidate.feature_vector, self.ema_feature.feature_vector)
            ))
            scores.append(0.60 * ema_color + 0.40 * ema_full)

        score = float(np.max(scores))   # take the best match across all galleries
        return score, (score >= self.match_threshold)
