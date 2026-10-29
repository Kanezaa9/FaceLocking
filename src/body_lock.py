from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyttsx3

# Initialize speech engine
engine = pyttsx3.init()

def say(text: str):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[speech error] {e}")

from .haar_5pt import Haar5ptDetector, FaceKpsBox, align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .recognize import DB_PATH, DEFAULT_THRESHOLD, Matcher, MatchResult, load_db
from .yolox import YoloXBodyDetector, BodyDetection
from .body_features import BodyFeatureExtractor, BodyFeatures, BodyLearner
from .tracker import SimpleBodyTracker, TrackedBody
from .face_lock import draw_tacho_needle, position_of

MAX_FACES = 4
LOCK_CONFIRM_FRAMES = 2
LOCK_LOSS_FRAMES = 90
DEFAULT_BODY_THRESHOLD = 0.35   # stricter body matching
BODY_REACQUIRE_THRESHOLD = 0.44   # minimum body similarity to re-lock after coming back
CENTER_ZONE = 0.10
DEFAULT_HISTORY_PATH = Path("data/action_history.jsonl")

MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)


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
            print(f"[history] error: {exc}")


def open_camera(index: Optional[int] = None, width: int = 640, height: int = 480) -> Tuple[cv2.VideoCapture, int]:
    """Open a webcam with a robust backend for Windows.
    Tries DirectShow (CAP_DSHOW) first, then falls back to the default backend.
    Returns the VideoCapture object and the index that succeeded.
    """
    candidates = [index] if index is not None else [0, 1, 2]
    if index is not None:
        for alt in [0, 1, 2]:
            if alt not in candidates:
                candidates.append(alt)
    for idx in candidates:
        # Try DirectShow backend which works better on many Windows setups
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # fallback to default backend
            cap.release()
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        time.sleep(0.2)
        ret, frame = cap.read()
        if ret and frame is not None and frame.mean() > 2.0:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Connected to Camera {idx} ({actual_w}x{actual_h}, mean={frame.mean():.1f})")
            return cap, idx
        cap.release()
    # Final fallback: try index 0 with default backend
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open any camera. Verify that a webcam is connected and not used by another application.")
    return cap, 0
    candidates = [index] if index is not None else [0, 1, 2]
    if index is not None:
        for alt in [0, 1, 2]:
            if alt not in candidates:
                candidates.append(alt)

    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        time.sleep(0.2)
        frame = None
        for _ in range(3):
            ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 2.0:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Connected to Camera {idx} ({actual_w}x{actual_h}, mean={frame.mean():.1f})")
            return cap, idx
        cap.release()

    idx = 1 if index is None else index
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {idx}")
    return cap, idx



class AsyncBodyDetector:
    def __init__(self, detector: YoloXBodyDetector):
        self.detector = detector
        self.in_queue: queue.Queue = queue.Queue(maxsize=1)
        self.out_queue: queue.Queue = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                frame = self.in_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame is None:
                break
            dets = self.detector.detect(frame)
            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            self.out_queue.put(dets)

    def submit(self, frame: np.ndarray):
        if self.in_queue.full():
            try:
                self.in_queue.get_nowait()
            except queue.Empty:
                pass
        self.in_queue.put(frame)

    def get_latest(self) -> Optional[List[BodyDetection]]:
        res = None
        while not self.out_queue.empty():
            try:
                res = self.out_queue.get_nowait()
            except queue.Empty:
                break
        return res

    def stop(self):
        self.running = False
        self.submit(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=1, help="Process every Nth frame for heavy detection (default 1 = every frame)")
    ap.add_argument("--name", default=None, help="Target identity to lock onto")
    ap.add_argument("--camera", type=int, default=None, help="Camera index (default: auto-detect)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--confidence", type=float, default=DEFAULT_BODY_THRESHOLD, help="Body match threshold")
    ap.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH, help="Action history jsonl")
    ap.add_argument("--sync", action="store_true", help="Run detection synchronously instead of worker thread")
    args = ap.parse_args()
    # Process every Nth frame for heavy detection to improve speed
    skip = max(1, args.skip)

    body_det = YoloXBodyDetector(score_thresh=0.35, nms_thresh=0.45)
    face_det = Haar5ptDetector(debug=False, max_mesh_faces=MAX_FACES)
    embedder = ArcFaceEmbedderONNX(debug=False)
    db = load_db(DB_PATH)
    matcher = Matcher(db, DEFAULT_THRESHOLD)
    tracker = SimpleBodyTracker()
    history = ActionHistory(args.history)

    if args.name and args.name not in matcher.names:
        raise SystemExit(f"'{args.name}' is not enrolled. Enrolled: {matcher.names}")

    target_name: Optional[str] = args.name
    learner = BodyLearner(target_name=target_name or "", match_threshold=args.confidence)
    async_det = None if args.sync else AsyncBodyDetector(body_det)

    cap, actual_cam = open_camera(args.camera, args.width, args.height)

    locked_track_id: Optional[int] = None
    lock_mode: Optional[str] = None
    candidate_track_id: Optional[int] = None
    candidate_name: Optional[str] = None
    candidate_n = 0
    missed_frames = 0
    last_side: Optional[str] = None
    show_debug = False
    fps, t_prev = 0.0, time.monotonic()
    last_body_dets: List[BodyDetection] = []

    def release_lock(reason: str):
        nonlocal locked_track_id, lock_mode, missed_frames, last_side
        if locked_track_id is not None:
            name_str = target_name or "Target"
            print(f"[lock] {time.strftime('%H:%M:%S')} released '{name_str}' ({reason})")
            history.record("lock_released", f"{name_str} lock released: {reason}", name_str)
        locked_track_id = None
        lock_mode = None
        missed_frames = 0
        last_side = None
        # deliberately keep learner.gallery intact so body re-acquisition works on re-entry

    print(f"Enrolled identities: {matcher.names}  target: {target_name or 'any enrolled'}")
    print("Keys: q=quit, l=release lock, c=clear learned features, r=reload db, +/- threshold, d=toggle debug")

    # Initialize frame skipping for performance
    frame_counter = 0
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Update FPS counter
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_prev)
            t_prev = now

            # Increment frame counter and decide whether to run heavy body detection this frame
            frame_counter += 1
            do_detection = (frame_counter % skip == 0)

            H, W = frame.shape[:2]
            vis = frame.copy()
            cx, cy = W // 2, H // 2
            cv2.line(vis, (cx - 15, cy), (cx + 15, cy), (160, 160, 160), 1)
            cv2.line(vis, (cx, cy - 15), (cx, cy + 15), (160, 160, 160), 1)

            if async_det is not None:
                if do_detection:
                    async_det.submit(frame)
                latest_dets = async_det.get_latest()
                if latest_dets is not None:
                    last_body_dets = latest_dets
            else:
                if do_detection:
                    last_body_dets = body_det.detect(frame)


            face_boxes = face_det.detect_all(frame, max_faces=MAX_FACES)
            faces: List[Tuple[FaceKpsBox, MatchResult]] = []
            for f in face_boxes:
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                faces.append((f, matcher.match(embedder.embed(aligned).embedding)))

            tracked_bodies = tracker.update(frame, last_body_dets, faces)

            if locked_track_id is None:
                # --- Try face-based acquisition first ---
                verified_cands = []
                for tb in tracked_bodies:
                    if tb.face_match and tb.face_match.accepted:
                        m_name = tb.face_match.name
                        if target_name is None or m_name == target_name:
                            verified_cands.append((tb.face_match.distance, tb, m_name))

                if verified_cands:
                    verified_cands.sort(key=lambda x: x[0])
                    best_cand = verified_cands[0]
                    cand_tb = best_cand[1]
                    cand_name = best_cand[2]

                    if cand_tb.track_id == candidate_track_id and cand_name == candidate_name:
                        candidate_n += 1
                    else:
                        candidate_track_id = cand_tb.track_id
                        candidate_name = cand_name
                        candidate_n = 1

                    if candidate_n >= LOCK_CONFIRM_FRAMES:
                        locked_track_id = candidate_track_id
                        target_name = candidate_name
                        learner.target_name = target_name
                        learner.reset()
                        if cand_tb.features:
                            cand_tb.features.from_rear = False
                            learner.add_sample(cand_tb.features)
                        lock_mode = "FACE"
                        candidate_track_id = None
                        candidate_n = 0
                        missed_frames = 0
                        print(f"[lock] {time.strftime('%H:%M:%S')} {target_name} DETECTED - full body locked. Learning body features...")
                        history.record("lock_acquired", f"{target_name} full body lock acquired", target_name)
                else:
                    candidate_track_id = None
                    candidate_n = 0

                # --- Body-only re-acquisition: person returned backwards after lock was lost ---
                if locked_track_id is None and learner.is_learned:
                    best_body_tb = None
                    best_body_sim = 0.0
                    for tb in tracked_bodies:
                        if tb.features:
                            sim, ok_body = learner.match(tb.features)
                            if ok_body and sim > best_body_sim:
                                best_body_sim = sim
                                best_body_tb = tb
                    if best_body_tb is not None:
                        locked_track_id = best_body_tb.track_id
                        lock_mode = "BODY_180"
                        candidate_track_id = None
                        candidate_n = 0
                        missed_frames = 0
                        print(f"[lock] {time.strftime('%H:%M:%S')} {target_name} re-acquired by body appearance (sim={best_body_sim:.2f}) - facing away")
            else:
                locked_tb = next((b for b in tracked_bodies if b.track_id == locked_track_id), None)
                if locked_tb is not None:
                    missed_frames = 0
                    if locked_tb.face_match and locked_tb.face_match.accepted and locked_tb.face_match.name == target_name:
                        lock_mode = "FACE"
                        if locked_tb.features:
                            locked_tb.features.from_rear = False
                            learner.add_sample(locked_tb.features)
                    else:
                        if locked_tb.features:
                            locked_tb.features.from_rear = True
                            sim, ok_body = learner.match(locked_tb.features)
                            if ok_body:
                                lock_mode = "BODY_180"
                                learner.add_sample(locked_tb.features)  # grow rear gallery
                            else:
                                if not learner.is_learned:
                                    lock_mode = "TRACK"
                                else:
                                    missed_frames += 1
                                    if missed_frames > 12:
                                        release_lock("body feature mismatch")
                                        locked_tb = None
                        else:
                            lock_mode = "TRACK"
                else:
                    best_match_tb = None
                    best_sim = 0.0
                    for tb in tracked_bodies:
                        if tb.features:
                            sim, ok_body = learner.match(tb.features)
                            if ok_body and sim > best_sim:
                                best_sim = sim
                                best_match_tb = tb

                    if best_match_tb is not None:
                        locked_track_id = best_match_tb.track_id
                        locked_tb = best_match_tb
                        lock_mode = "BODY_180"
                        missed_frames = 0
                        print(f"[lock] {time.strftime('%H:%M:%S')} Re-associated body for {target_name} via body features (sim={best_sim:.2f})")
                    else:
                        missed_frames += 1
                        if missed_frames > LOCK_LOSS_FRAMES:
                            release_lock("target body lost")

            for tb in tracked_bodies:
                bx1, by1, bx2, by2 = tb.box_int
                is_locked = (tb.track_id == locked_track_id)

                if is_locked:
                    color = MAGENTA
                    box_thick = 3
                elif tb.face_match and tb.face_match.accepted:
                    color = GREEN
                    box_thick = 2
                elif candidate_track_id == tb.track_id:
                    color = CYAN
                    box_thick = 2
                else:
                    color = (0, 0, 200)
                    box_thick = 1

                cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, box_thick)

                if tb.face is not None:
                    f = tb.face
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                    for pt in f.kps:
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1, cv2.LINE_AA)

                label_parts = []
                if is_locked:
                    if lock_mode == "FACE":
                        label_parts.append(f"{target_name} [FACE+BODY LOCKED]")
                    elif lock_mode == "BODY_180":
                        sim_val = 0.0
                        if tb.features:
                            sim_val, _ = learner.match(tb.features)
                        label_parts.append(f"{target_name} [BODY 180 RECOGNIZED] ({int(sim_val*100)}%)")
                    else:
                        label_parts.append(f"{target_name} [BODY TRACKED]")
                elif tb.face_match and tb.face_match.accepted:
                    # Use the existing acceptance check; if distance is above threshold, Matcher marks accepted=False
                    label_parts.append(f"{tb.face_match.name} (dist={tb.face_match.distance:.2f})")
                elif tb.face_match and not tb.face_match.accepted:
                    # Not accepted → treat as unknown
                    label_parts.append("Unknown")
                    say("Unknown person detected")
                    print("[unknown] Detected person not enrolled")
                elif tb.track_id == candidate_track_id:
                    label_parts.append(f"Verifying {candidate_name} ({candidate_n}/{LOCK_CONFIRM_FRAMES})")
                else:
                    if learner.is_learned and tb.features:
                        s, _ = learner.match(tb.features)
                        label_parts.append(f"Unknown (sim={int(s*100)}%)")
                        say("Unknown person detected")
                        print("[unknown] Detected person not enrolled")
                    else:
                        label_parts.append("Person")

                tag = " | ".join(label_parts)
                cv2.putText(vis, tag, (bx1, max(20, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            locked_tb_hud: Optional[TrackedBody] = None
            if locked_track_id is not None:
                locked_tb_hud = next((b for b in tracked_bodies if b.track_id == locked_track_id), None)
                locked_tb = locked_tb_hud
                if locked_tb is not None:
                    bcx, bcy = locked_tb.center
                    draw_tacho_needle(vis, (cx, cy), (int(bcx), int(bcy)))
                    horiz, vert, dx, dy, dist = position_of((bcx, bcy), W, H)
                    side = horiz if vert == "CENTER" else (vert if horiz == "CENTER" else f"{vert}-{horiz}")
                    if side != last_side:
                        print(f"[pos] {time.strftime('%H:%M:%S')} {target_name} is now {side} (dx={dx:+.0f}px, dy={dy:+.0f}px)")
                        history.record("body_movement", f"{target_name} moved {side} (dx={dx:+.0f}px, dy={dy:+.0f}px)", target_name)
                        last_side = side

                    hud_lines = [
                        (f"{target_name} BODY LOCKED ({lock_mode})", MAGENTA),
                        (f"Position: {side} (dx={dx:+.0f}px dy={dy:+.0f}px dist={dist:.0f}px)", YELLOW),
                    ]

                    if learner.is_learned:
                        hud_lines.append((
                            f"Body Model: ACTIVE  front={len(learner.front_gallery)}  rear={len(learner.rear_gallery)}",
                            (0, 255, 0),
                        ))
                    else:
                        pct = int(learner.progress * 100)
                        hud_lines.append((f"Learning Body Features: {pct}% ({learner.samples_collected}/{learner.min_samples})", CYAN))

                    for k, (text, col) in enumerate(hud_lines):
                        cv2.putText(vis, text, (15, 125 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

                    if learner.ema_feature is not None:
                        ur, ug, ub = learner.ema_feature.dominant_upper_rgb
                        lr, lg, lb = learner.ema_feature.dominant_lower_rgb
                        swatch_x = 15
                        swatch_y = 125 + 28 * len(hud_lines) + 6
                        cv2.putText(vis, "Learned Colors: Upper", (swatch_x, swatch_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                        cv2.rectangle(vis, (swatch_x + 160, swatch_y), (swatch_x + 190, swatch_y + 22), (ub, ug, ur), -1)
                        cv2.rectangle(vis, (swatch_x + 160, swatch_y), (swatch_x + 190, swatch_y + 22), (255, 255, 255), 1)
                        cv2.putText(vis, "Lower", (swatch_x + 205, swatch_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                        cv2.rectangle(vis, (swatch_x + 255, swatch_y), (swatch_x + 285, swatch_y + 22), (lb, lg, lr), -1)
                        cv2.rectangle(vis, (swatch_x + 255, swatch_y), (swatch_x + 285, swatch_y + 22), (255, 255, 255), 1)

            status_text = ""
            status_col = (0, 255, 0)
            if locked_track_id is not None and locked_tb_hud is None:
                missed_frames += 1
                if missed_frames > LOCK_LOSS_FRAMES:
                    release_lock("target body lost")
                else:
                    status_text = f"TARGET NOT IN FRAME ({missed_frames}/{LOCK_LOSS_FRAMES})"
                    status_col = ORANGE
            elif locked_track_id is None:
                if candidate_n:
                    status_text = f"CONFIRMING {candidate_name} ({candidate_n}/{LOCK_CONFIRM_FRAMES})"
                    status_col = CYAN
                elif learner.is_learned:
                    status_text = f"SEARCHING (body model READY - will re-lock on re-entry): {target_name or 'enrolled person'}"
                    status_col = (0, 200, 255)
                else:
                    status_text = f"SEARCHING: {target_name or 'enrolled person'}"
                    status_col = YELLOW
            else:
                status_text = f"LOCKED: {target_name} [{lock_mode}]"
                status_col = MAGENTA

            cv2.rectangle(vis, (0, 0), (W, 36), (20, 20, 20), -1)
            cv2.line(vis, (0, 36), (W, 36), (60, 60, 60), 1)
            header = f"IDs={len(matcher.names)}  Bodies={len(tracked_bodies)}  FPS={fps:.1f}  |  {status_text}"
            cv2.putText(vis, header, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2, cv2.LINE_AA)

            if show_debug:
                cv2.putText(vis, f"Learned samples: {len(learner.gallery)}", (W - 280, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1)
                for idx, tb in enumerate(tracked_bodies):
                    sim_txt = "N/A"
                    if tb.features and len(learner.gallery) > 0:
                        sim_s, _ = learner.match(tb.features)
                        sim_txt = f"{sim_s:.2f}"
                    cv2.putText(vis, f"T#{tb.track_id}: score={tb.score:.2f} sim={sim_txt}", (W - 280, 75 + 22 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Body & Face Tracking", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                release_lock("manual")
            elif key == ord("c"):
                learner.reset()
                release_lock("manual body model clear")
                print("[learner] Cleared learned body features - will re-learn on next face lock")
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord("r"):
                matcher.reload(DB_PATH)
                print(f"[db] Reloaded: {matcher.names}")
            elif key in (ord("+"), ord("=")):
                learner.match_threshold = min(0.95, learner.match_threshold + 0.02)
                print(f"[thr] Body match threshold: {learner.match_threshold:.2f}")
            elif key == ord("-"):
                learner.match_threshold = max(0.20, learner.match_threshold - 0.02)
                print(f"[thr] Body match threshold: {learner.match_threshold:.2f}")

    finally:
        if async_det is not None:
            async_det.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
