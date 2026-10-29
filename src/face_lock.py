# src/face_lock.py
"""
Face locking + position + expression + blink detection (no servo).

Every face in the frame is labelled (enrolled name, or "Unknown"). Once the
target person is recognized for LOCK_CONFIRM_FRAMES frames in a row, the
system locks onto them and reports:
  - where their nose tip is relative to the frame centre (LEFT/RIGHT/UP/DOWN,
    pixel offset and distance), printing a message when they change side
  - their expression: neutral, smiling, sad, frowning, surprised, grimacing
  - blinks, with a running count

Other faces never take over the lock: an unknown face is only labelled, and
the lock is released if the locked face turns out to be someone else for
LOCK_SWITCH_FRAMES frames. The lock survives brief recognition dropouts
(big expressions raise the distance) as long as the face stays in place.

Warnings: an ORANGE banner while the locked face is briefly missing, a RED
banner once it is lost / not in the frame.

Right after locking, hold a NEUTRAL face (eyes open) for about a second
while the expression and blink baselines are calibrated.

Run:
    python -m src.face_lock --name ATUMANYIRE            # built-in webcam
    python -m src.face_lock --name ATUMANYIRE --camera 1 # external HD camera

Keys:
    q   : quit
    l   : release the current lock
    c   : recalibrate neutral expression + reset blink count
    d   : toggle expression debug values
    r   : reload database from disk
    +/- : loosen / tighten the recognition threshold
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, FaceKpsBox, align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .expression import ExpressionDetector
from .recognize import DB_PATH, DEFAULT_THRESHOLD, Matcher, MatchResult, load_db

MAX_FACES = 4
LOCK_CONFIRM_FRAMES = 2
LOCK_LOSS_FRAMES = 15
LOCK_SWITCH_FRAMES = 8
LOCK_MATCH_RADIUS = 150    # px the nose may jump between frames and still count as the locked face
IMPOSTOR_DIST = 0.72       # locked face this far from the template = clearly a different person
CENTER_ZONE = 0.10         # |offset| below this fraction of width/height counts as CENTER
NOSE_EMA = 0.5
NEEDLE_DIAL_FRAC = 0.16    # tacho dial radius as a fraction of the shorter frame side
NEEDLE_DIAL_TICKS = 12     # tick marks around the dial
NEEDLE_HALF_WIDTH = 4      # needle half-width at the hub (px)
DEFAULT_HISTORY_PATH = Path("data/action_history.jsonl")


class ActionHistory:
    def __init__(self, path: Path):
        self.path = Path(path)

    def record(self, action_type: str, description: str, identity: Optional[str] = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "action_type": action_type,
            "description": description,
        }
        if identity:
            entry["identity"] = identity
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[history] could not write {self.path}: {exc}")


EXPR_COLORS = {
    "neutral": (200, 200, 200),
    "smiling": (0, 220, 0),
    "sad": (255, 140, 0),
    "frowning": (0, 0, 255),
    "surprised": (0, 220, 255),
    "grimacing": (180, 0, 255),
    "calibrating": (180, 180, 180),
}
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # DirectShow opens USB webcams faster on Windows and honours resolution requests
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {index} not opened.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(0.5)
    for _ in range(10):
        cap.read()
    print(f"Camera {index}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


def nose_of(f: FaceKpsBox) -> np.ndarray:
    return f.kps[2].astype(np.float32)


def position_of(nose: np.ndarray, W: int, H: int) -> Tuple[str, str, float, float, float]:
    """(horizontal side, vertical side, dx, dy, distance) of the nose relative to frame centre."""
    dx, dy = float(nose[0] - W / 2.0), float(nose[1] - H / 2.0)
    horiz = "LEFT" if dx < -CENTER_ZONE * W else ("RIGHT" if dx > CENTER_ZONE * W else "CENTER")
    vert = "UP" if dy < -CENTER_ZONE * H else ("DOWN" if dy > CENTER_ZONE * H else "CENTER")
    return horiz, vert, dx, dy, float(np.hypot(dx, dy))


def banner(img: np.ndarray, text: str, color) -> None:
    H, W = img.shape[:2]
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), color, 10)
    cv2.rectangle(img, (0, 40), (W, 90), color, -1)
    cv2.putText(img, text, (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def draw_tacho_needle(img: np.ndarray, pivot: Tuple[int, int], tip: Tuple[int, int],
                      color=(0, 255, 255)) -> Tuple[float, float]:
    """Tacho dial centred on pivot with a tapered needle whose tip tracks the nose.

    Returns (bearing_deg, length_px) where bearing is 0 when the tip is straight
    up and increases clockwise. The dial stays inside the full, uncropped frame.
    """
    cx, cy = pivot
    nx, ny = tip
    dx, dy = nx - cx, ny - cy
    length = math.hypot(dx, dy)

    H, W = img.shape[:2]
    r = int(min(W, H) * NEEDLE_DIAL_FRAC)
    cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 360, color, 1, cv2.LINE_AA)
    for k in range(NEEDLE_DIAL_TICKS):
        a = (2.0 * math.pi * k) / NEEDLE_DIAL_TICKS
        ca, sa = math.cos(a), math.sin(a)
        cv2.line(img, (int(cx + ca * r * 0.82), int(cy + sa * r * 0.82)),
                 (int(cx + ca * r), int(cy + sa * r)), color, 1, cv2.LINE_AA)

    if length >= 3.0:
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        poly = np.array([
            [int(cx + px * NEEDLE_HALF_WIDTH), int(cy + py * NEEDLE_HALF_WIDTH)],
            [int(cx - px * NEEDLE_HALF_WIDTH), int(cy - py * NEEDLE_HALF_WIDTH)],
            [int(nx), int(ny)],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, poly, color, cv2.LINE_AA)

    cv2.circle(img, (cx, cy), 6, color, -1, cv2.LINE_AA)
    cv2.circle(img, (int(nx), int(ny)), 5, color, -1, cv2.LINE_AA)

    bearing = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
    cv2.putText(img, f"{bearing:.0f} deg  {length:.0f}px", (cx + r + 8, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return bearing, length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="enrolled person to lock onto (default: first recognized)")
    ap.add_argument("--camera", type=int, default=0, help="camera index (external USB camera is usually 1)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH, help="JSONL action history path")
    args = ap.parse_args()

    det = Haar5ptDetector(debug=False, max_mesh_faces=MAX_FACES)
    embedder = ArcFaceEmbedderONNX(debug=False)
    matcher = Matcher(load_db(DB_PATH), DEFAULT_THRESHOLD)
    history = ActionHistory(args.history)
    expr = ExpressionDetector()
    if args.name and args.name not in matcher.names:
        raise SystemExit(f"'{args.name}' is not enrolled. Enrolled: {matcher.names}")
    target_text = args.name or "enrolled face"

    cap = open_camera(args.camera, args.width, args.height)

    locked_name: Optional[str] = None
    locked_nose: Optional[np.ndarray] = None
    missed_frames = 0
    candidate_name: Optional[str] = None
    candidate_n = 0
    switch_n = 0
    last_expr: Optional[str] = None
    last_side: Optional[str] = None
    blink_flash_until = 0.0
    show_debug = False
    fps, t_prev = 0.0, time.monotonic()

    def release(reason: str):
        nonlocal locked_name, locked_nose, missed_frames, last_expr, last_side, switch_n
        if locked_name is not None:
            print(f"[lock] {time.strftime('%H:%M:%S')} released '{locked_name}' ({reason}) - blinks this lock: {expr.blinks}")
            history.record("lock_released", f"{locked_name} lock released: {reason}", locked_name)
        locked_name, locked_nose, last_expr, last_side = None, None, None, None
        missed_frames = switch_n = 0
        expr.reset()

    print(f"Loaded {len(matcher.names)} identities: {matcher.names}  target: {target_text}")
    print(f"Action history: {args.history}")
    print("q=quit, l=release lock, c=recalibrate, d=debug, r=reload db, +/- threshold")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_prev)
            t_prev = now

            H, W = frame.shape[:2]
            vis = frame.copy()
            cx, cy = W // 2, H // 2
            cv2.line(vis, (cx - 15, cy), (cx + 15, cy), (160, 160, 160), 1)
            cv2.line(vis, (cx, cy - 15), (cx, cy + 15), (160, 160, 160), 1)

            faces: List[Tuple[FaceKpsBox, MatchResult]] = []
            for f in det.detect_all(frame, max_faces=MAX_FACES):
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                faces.append((f, matcher.match(embedder.embed(aligned).embedding)))

            locked_i: Optional[int] = None
            if locked_name is None:
                # --- acquire: target recognized for several frames in a row ---
                hits = [(m.distance, i) for i, (_, m) in enumerate(faces)
                        if m.accepted and (args.name is None or m.name == args.name)]
                if hits:
                    i = min(hits)[1]
                    name = faces[i][1].name
                    candidate_n = candidate_n + 1 if name == candidate_name else 1
                    candidate_name = name
                    if candidate_n >= LOCK_CONFIRM_FRAMES:
                        locked_name, locked_nose, locked_i = candidate_name, nose_of(faces[i][0]), i
                        candidate_name, candidate_n = None, 0
                        missed_frames = switch_n = 0
                        expr.reset()
                        print(f"[lock] {time.strftime('%H:%M:%S')} {locked_name} DETECTED - locked on. "
                              f"Hold a neutral face to calibrate")
                        history.record("lock_acquired", f"{locked_name} identity lock acquired", locked_name)
                else:
                    candidate_name, candidate_n = None, 0
            else:
                # --- keep: prefer the face recognized as the locked person, else the one nearest the old spot ---
                named = [(m.distance, i) for i, (_, m) in enumerate(faces) if m.accepted and m.name == locked_name]
                if named:
                    locked_i = min(named)[1]
                else:
                    near = [(float(np.linalg.norm(nose_of(f) - locked_nose)), i) for i, (f, _) in enumerate(faces)]
                    near = [n for n in near if n[0] < LOCK_MATCH_RADIUS]
                    if near:
                        locked_i = min(near)[1]
                if locked_i is not None:
                    m = faces[locked_i][1]
                    impostor = (m.accepted and m.name != locked_name) or m.distance > IMPOSTOR_DIST
                    switch_n = switch_n + 1 if impostor else 0
                    if switch_n >= LOCK_SWITCH_FRAMES:
                        release(f"face now looks like {m.name or 'someone unknown'}")
                        locked_i = None

            # --- draw every face ---
            for i, (f, m) in enumerate(faces):
                if i == locked_i:
                    continue
                color = (0, 255, 0) if m.accepted else RED
                label = m.name if m.accepted else "Unknown"
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                cv2.putText(vis, f"{label}  dist={m.distance:.2f}", (f.x1, max(0, f.y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- locked face: position, expression, blinks ---
            if locked_i is not None:
                f, m = faces[locked_i]
                missed_frames = 0
                raw_nose = nose_of(f)
                locked_nose = NOSE_EMA * locked_nose + (1 - NOSE_EMA) * raw_nose
                nx, ny = int(locked_nose[0]), int(locked_nose[1])

                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), MAGENTA, 3)
                cv2.putText(vis, f"{locked_name} (locked)  dist={m.distance:.2f}", (f.x1, max(0, f.y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, MAGENTA, 2)
                draw_tacho_needle(vis, (cx, cy), (nx, ny))

                horiz, vert, dx, dy, dist = position_of(locked_nose, W, H)
                side = horiz if vert == "CENTER" else (vert if horiz == "CENTER" else f"{vert}-{horiz}")
                if side != last_side:
                    print(f"[pos] {time.strftime('%H:%M:%S')} {locked_name} is now {side} "
                          f"(dx={dx:+.0f}px, dy={dy:+.0f}px)")
                    history.record("face_movement", f"{locked_name} moved {side} (dx={dx:+.0f}px, dy={dy:+.0f}px)", locked_name)
                    last_side = side

                lines = [
                    (f"{locked_name} DETECTED", MAGENTA),
                    (f"Position: {side}", (0, 255, 255)),
                    (f"Nose from centre: dx={dx:+.0f}px dy={dy:+.0f}px  dist={dist:.0f}px", (0, 255, 255)),
                ]

                if f.mesh is not None:
                    er = expr.update(f.mesh)
                    if er.label == "calibrating":
                        lines.append((f"Calibrating... keep neutral {int(er.calib_progress * 100)}%", (200, 200, 200)))
                    else:
                        if er.label != last_expr:
                            print(f"[expr] {time.strftime('%H:%M:%S')} {locked_name}: {er.label}")
                            history.record("expression", f"{locked_name} expression changed to {er.label}", locked_name)
                            last_expr = er.label
                        if er.just_blinked:
                            print(f"[blink] {time.strftime('%H:%M:%S')} {locked_name} blinked (total {er.blinks})")
                            history.record("blink", f"{locked_name} blink detected (total {er.blinks})", locked_name)
                            blink_flash_until = now + 0.4
                        lines.append((f"Expression: {er.label.upper()}", EXPR_COLORS[er.label]))
                        blink_text = f"Blinks: {er.blinks}"
                        if er.eyes_closed:
                            blink_text += "  (EYES CLOSED)"
                        elif now < blink_flash_until:
                            blink_text += "  BLINK!"
                        lines.append((blink_text, (255, 255, 0)))

                    if show_debug and er.deltas:
                        for k, (key, v) in enumerate(er.deltas.items()):
                            cv2.putText(vis, f"{key}: {v:+.3f}", (W - 180, 120 + 22 * k),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                        cv2.putText(vis, f"raw: {er.raw_label}", (W - 180, 120 + 22 * len(er.deltas)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

                for k, (text, color) in enumerate(lines):
                    cv2.putText(vis, text, (10, 125 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            # --- missing-face warnings ---
            if locked_name is not None and locked_i is None:
                missed_frames += 1
                if missed_frames > LOCK_LOSS_FRAMES:
                    release("face lost")
                else:
                    banner(vis, f"WARNING: {locked_name} NOT FOUND ({missed_frames}/{LOCK_LOSS_FRAMES})", ORANGE)
            if locked_name is None:
                msg = f"WARNING: {target_text} NOT FOUND"
                if candidate_n:
                    msg = f"Recognizing {candidate_name}... {candidate_n}/{LOCK_CONFIRM_FRAMES}"
                banner(vis, msg, ORANGE if candidate_n else RED)

            cv2.putText(vis, f"IDs={len(matcher.names)}  thr={matcher.threshold:.2f}  faces={len(faces)}  "
                             f"fps={fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("face_lock", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                release("manual")
            elif key == ord("c"):
                expr.reset()
                last_expr = None
                print("[expr] recalibrating - keep a neutral face, eyes open")
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord("r"):
                matcher.reload(DB_PATH)
                print(f"Reloaded: {matcher.names}")
            elif key in (ord("+"), ord("=")):
                matcher.threshold = min(1.2, matcher.threshold + 0.01)
            elif key == ord("-"):
                matcher.threshold = max(0.05, matcher.threshold - 0.01)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


#camera-motor-