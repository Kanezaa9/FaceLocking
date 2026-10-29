# src/recognize.py
"""
Live face recognition: detect faces, align, embed, compare against the
enrolled database, and label each face with a name or "Unknown".

Run:
    python -m src.recognize

Keys:
    q   : quit
    r   : reload database from disk
    +/- : loosen / tighten the acceptance threshold live
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX

DB_PATH = Path("data/db/face_db.npz")
DEFAULT_THRESHOLD = 0.27  # calibrated (lower = stricter)


def load_db(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(str(path), allow_pickle=True)
    return {k: np.asarray(data[k], dtype=np.float32).reshape(-1) for k in data.files}


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    accepted: bool


class Matcher:
    def __init__(self, db: Dict[str, np.ndarray], threshold: float):
        self.db = db
        self.threshold = threshold
        self._rebuild()

    def _rebuild(self):
        self.names: List[str] = sorted(self.db.keys())
        if self.names:
            self.mat = np.stack([self.db[n] for n in self.names], axis=0)
        else:
            self.mat = None

    def reload(self, path: Path):
        self.db = load_db(path)
        self._rebuild()

    def match(self, emb: np.ndarray) -> MatchResult:
        if self.mat is None:
            return MatchResult(None, 1.0, False)
        sims = self.mat @ emb.reshape(-1)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        dist = 1.0 - best_sim
        accepted = dist <= self.threshold
        return MatchResult(self.names[best_i] if accepted else None, dist, accepted)


def main():
    det = Haar5ptDetector(smooth_alpha=0.80, debug=False)
    embedder = ArcFaceEmbedderONNX(debug=False)

    db = load_db(DB_PATH)
    matcher = Matcher(db, DEFAULT_THRESHOLD)
    print(f"Loaded {len(matcher.names)} identities: {matcher.names}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened.")

    time.sleep(0.5)
    for _ in range(10):
        cap.read()

    print("q=quit, r=reload db, +/- adjust threshold")

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
            res = embedder.embed(aligned)
            m = matcher.match(res.embedding)

            label = m.name if m.name else "Unknown"
            color = (0, 255, 0) if m.accepted else (0, 0, 255)
            cv2.putText(vis, f"{label}  dist={m.distance:.3f}", (f.x1, max(0, f.y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        header = f"IDs={len(matcher.names)}  thr={matcher.threshold:.2f}"
        cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("recognize", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            matcher.reload(DB_PATH)
            print(f"Reloaded: {matcher.names}")
        elif key in (ord("+"), ord("=")):
            matcher.threshold = min(1.2, matcher.threshold + 0.01)
        elif key == ord("-"):
            matcher.threshold = max(0.05, matcher.threshold - 0.01)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()