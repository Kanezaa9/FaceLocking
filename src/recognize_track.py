# src/recognize_track.py
"""
Live recognition + face locking + horizontal servo tracking.
When no face is locked, the servo sweeps left-right searching until a
recognized face is found, then locks on and tracks it.

Run:
    python -m src.recognize_track

Keys:
    q   : quit
    l   : release the current lock (resumes sweeping)
    +/- : adjust recognition threshold
"""
from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .recognize import DB_PATH, DEFAULT_THRESHOLD, Matcher, load_db
from .servo_control import ServoController

SERVO_PORT = "COM5"

LOCK_LOSS_FRAMES = 15
LOCK_MATCH_RADIUS = 120

# only send a new servo command if the angle changed by at least this much
# (skips tiny frame-to-frame wobble, reduces serial write frequency)
MIN_ANGLE_DELTA = 3

# --- search/sweep settings ---
SWEEP_MIN_ANGLE = 30
SWEEP_MAX_ANGLE = 150
SWEEP_STEP_DEG = 2          # degrees to move per sweep tick
SWEEP_STEP_EVERY_S = 0.05   # how often to advance the sweep (seconds)



def face_center(f) -> np.ndarray:
    return np.array([(f.x1 + f.x2) / 2.0, (f.y1 + f.y2) / 2.0], dtype=np.float32)


class Sweeper:
    """Sweeps the servo back and forth between SWEEP_MIN_ANGLE and SWEEP_MAX_ANGLE."""

    def __init__(self):
        self.angle = 90.0
        self.direction = 1  # +1 = increasing angle, -1 = decreasing
        self.last_step_t = 0.0

    def reset_from(self, current_angle: float):
        """Start sweeping outward from wherever tracking last left off."""
        self.angle = current_angle
        self.direction = 1 if current_angle < 90 else -1
        self.last_step_t = 0.0

    def step(self) -> float:
        now = time.time()
        if now - self.last_step_t >= SWEEP_STEP_EVERY_S:
            self.last_step_t = now
            self.angle += self.direction * SWEEP_STEP_DEG
            if self.angle >= SWEEP_MAX_ANGLE:
                self.angle = SWEEP_MAX_ANGLE
                self.direction = -1
            elif self.angle <= SWEEP_MIN_ANGLE:
                self.angle = SWEEP_MIN_ANGLE
                self.direction = 1
        return self.angle


def main():
    det = Haar5ptDetector(smooth_alpha=0.80, debug=False)
    embedder = ArcFaceEmbedderONNX(debug=False)
    matcher = Matcher(load_db(DB_PATH), DEFAULT_THRESHOLD)
    servo = ServoController(port=SERVO_PORT)
    sweeper = Sweeper()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened.")
    time.sleep(0.5)
    for _ in range(10):
        cap.read()

    locked_name: Optional[str] = None
    locked_center: Optional[np.ndarray] = None
    missed_frames = 0
    last_sent_angle = 90.0

    print(f"Loaded identities: {matcher.names}")
    print("q=quit, l=release lock, +/- adjust threshold")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            H, W = frame.shape[:2]
            frame_center_x = W / 2.0
            vis = frame.copy()

            faces = det.detect(frame, max_faces=1)
            found_this_frame = False

            detected_name: Optional[str] = None
            detected_dist: float = 1.0
            detected_accepted: bool = False
            detected_center: Optional[np.ndarray] = None

            if faces:
                f = faces[0]
                detected_center = face_center(f)

                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                res = embedder.embed(aligned)
                m = matcher.match(res.embedding)
                detected_name, detected_dist, detected_accepted = m.name, m.distance, m.accepted

                if locked_name is None:
                    if detected_accepted:
                        locked_name = detected_name
                        locked_center = detected_center
                        missed_frames = 0
                        found_this_frame = True
                        print(f"[lock] acquired '{locked_name}' at {locked_center}")
                else:
                    close_enough = np.linalg.norm(detected_center - locked_center) < LOCK_MATCH_RADIUS
                    if detected_accepted and detected_name == locked_name and close_enough:
                        locked_center = detected_center
                        missed_frames = 0
                        found_this_frame = True

                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                label = detected_name if detected_name else "Unknown"
                color = (0, 255, 0) if detected_accepted else (0, 0, 255)
                cv2.putText(vis, f"{label} dist={detected_dist:.3f}", (f.x1, max(0, f.y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if locked_name is not None and not found_this_frame:
                missed_frames += 1
                if missed_frames > LOCK_LOSS_FRAMES:
                    print(f"[lock] lost '{locked_name}', releasing lock -> resuming sweep")
                    locked_name = None
                    locked_center = None
                    missed_frames = 0
                    sweeper.reset_from(last_sent_angle)

            if locked_name is not None and locked_center is not None:
                # --- TRACKING MODE ---
                offset_x = locked_center[0] - frame_center_x
                max_offset = W / 2.0
                norm = np.clip(offset_x / max_offset, -1.0, 1.0)
                angle = 90 - norm * 60

                if abs(angle - last_sent_angle) >= MIN_ANGLE_DELTA:
                    servo.set_angle(angle)
                    last_sent_angle = angle

                cv2.circle(vis, (int(locked_center[0]), int(locked_center[1])), 6, (255, 0, 255), -1)
                cv2.putText(vis, f"LOCKED: {locked_name}  offset={offset_x:.0f}px  angle={angle:.0f}",
                            (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            else:
                # --- SEARCH MODE: sweep until a recognized face is found ---
                angle = sweeper.step()
                servo.set_angle(angle)
                last_sent_angle = angle
                cv2.putText(vis, f"searching... angle={angle:.0f}", (10, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.line(vis, (W // 2, 0), (W // 2, H), (100, 100, 100), 1)
            cv2.imshow("recognize_track", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                locked_name = None
                locked_center = None
                missed_frames = 0
                sweeper.reset_from(last_sent_angle)
            if key in (ord("+"), ord("=")):
                matcher.threshold = min(1.2, matcher.threshold + 0.01)
            if key == ord("-"):
                matcher.threshold = max(0.05, matcher.threshold - 0.01)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        servo.close()


if __name__ == "__main__":
    main()