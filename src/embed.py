# src/embed.py
"""
Turns an aligned 112x112 face crop into a 512-D L2-normalized embedding
using the ArcFace ONNX model.

Run:
    python -m src.embed

Keys:
    q : quit
    p : print embedding stats to terminal
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort

from .haar_5pt import Haar5ptDetector, align_face_5pt


@dataclass
class EmbeddingResult:
    embedding: np.ndarray  # (512,) float32, L2-normalized
    norm_before: float
    dim: int


class ArcFaceEmbedderONNX:
    """
    Input: aligned 112x112 BGR image.
    Output: L2-normalized 512-D embedding.
    """

    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.in_w, self.in_h = input_size
        self.debug = debug

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

        self.in_shape = self.sess.get_inputs()[0].shape
        self.is_nchw = (len(self.in_shape) == 4 and self.in_shape[1] == 3)

        if debug:
            print("[embed] model loaded")
            print("[embed] input:", self.sess.get_inputs()[0].shape)
            print("[embed] output:", self.sess.get_outputs()[0].shape)

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (self.in_h, self.in_w):
            aligned_bgr = cv2.resize(aligned_bgr, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        if self.is_nchw:
            x = np.transpose(rgb, (2, 0, 1))[None, ...]
        else:
            x = rgb[None, ...]
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12):
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32), n

    def embed(self, aligned_bgr: np.ndarray) -> EmbeddingResult:
        x = self._preprocess(aligned_bgr)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        v = y.reshape(-1).astype(np.float32)
        v_norm, n0 = self._l2_normalize(v)
        return EmbeddingResult(v_norm, n0, v_norm.size)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened.")

    time.sleep(0.5)
    for _ in range(10):
        cap.read()

    det = Haar5ptDetector(smooth_alpha=0.80, debug=False)
    emb_model = ArcFaceEmbedderONNX(debug=True)

    prev_emb: Optional[np.ndarray] = None
    print("Embedding Demo running. Press 'q' to quit, 'p' to print embedding.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        vis = frame.copy()
        faces = det.detect(frame, max_faces=1)

        if faces:
            f = faces[0]
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)

            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            res = emb_model.embed(aligned)

            lines = [f"embedding dim: {res.dim}", f"norm(before L2): {res.norm_before:.2f}"]
            if prev_emb is not None:
                sim = cosine_similarity(prev_emb, res.embedding)
                lines.append(f"cos(prev,this): {sim:.3f}")
            prev_emb = res.embedding

            y = 30
            for line in lines:
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 25

            h, w = vis.shape[:2]
            vis[10:170, w - 170:w - 10] = cv2.resize(aligned, (160, 160))
        else:
            cv2.putText(vis, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Face Embedding", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p") and prev_emb is not None:
            print("dim:", prev_emb.size, "min/max:", prev_emb.min(), prev_emb.max())
            print("first10:", prev_emb[:10])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()