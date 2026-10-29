# mqtt_tracking/face_track_mqtt.py
"""
Live recognition + face locking + pan tracking over MQTT (ESP8266 + 28BYJ-48).

Same search/lock logic as src/recognize_track.py, but angles are published to
the MQTT broker instead of a serial servo.

Modes:
    fixed (default)   - camera is fixed (e.g. laptop webcam), the motor/pointer only
                        points at the face (absolute mapping, same formula as recognize_track.py).
    onboard           - camera is mounted ON the motor. Closed loop: each update
                        nudges the target angle toward the face until it is centred.

Run (from repo root, broker must be running):
    python -m mqtt_tracking.face_track_mqtt --host 127.0.0.1

Keys:
    q   : quit
    f   : toggle fullscreen
    i   : invert needle direction (if it turns opposite to the face on screen)
    l   : release the current lock (resumes sweeping)
    c   : re-centre (target 90 deg)
    h   : tell the ESP8266 that its CURRENT physical position is 90 deg (re-home)
    r   : release motor coils (then turn it to centre by hand and press h)
    +/- : adjust recognition threshold
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np

from src.haar_5pt import Haar5ptDetector, align_face_5pt
from src.embed import ArcFaceEmbedderONNX
from src.recognize_track import (
    DB_PATH, DEFAULT_THRESHOLD, LOCK_LOSS_FRAMES, LOCK_MATCH_RADIUS, MIN_ANGLE_DELTA,
    Matcher, Sweeper, face_center, load_db,
)
from .mqtt_publisher import MqttAngleController

# --- onboard (closed-loop) tracking ---
TRACK_DEADBAND = 0.08      # |normalized offset| below this = centred, don't move
TRACK_GAIN_DEG = 12.0      # degrees moved per update when the face is at the frame edge
TRACK_UPDATE_S = 0.20      # min time between corrections, lets the motor catch up (prevents overshoot)
ANGLE_MIN, ANGLE_MAX = 0.0, 180.0

# --- on-screen movement label ---
MOVE_CENTER_ZONE = 0.10    # |normalized offset| below this = face is at CENTER
MOVE_SPEED_PX = 3.0        # smoothed px/frame (at 640 px wide) above this = face is moving


def screen_size() -> tuple:
    """Primary screen size in real pixels (Windows); falls back to 1920x1080."""
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
        return ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="MQTT broker address")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="face_tracking", help="base topic")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280, help="requested camera width")
    ap.add_argument("--height", type=int, default=720, help="requested camera height")
    ap.add_argument("--mode", choices=["onboard", "fixed"], default="fixed")
    ap.add_argument("--invert", action="store_true", help="flip direction if the motor turns away from the face")
    ap.add_argument("--no-mirror", action="store_true", help="show the raw camera image instead of a selfie-style mirror")
    args = ap.parse_args()
    sign = -1.0 if args.invert else 1.0

    det = Haar5ptDetector(smooth_alpha=0.80, debug=False)
    embedder = ArcFaceEmbedderONNX(debug=False)
    matcher = Matcher(load_db(DB_PATH), DEFAULT_THRESHOLD)
    motor = MqttAngleController(args.host, args.port, args.topic)
    sweeper = Sweeper()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"[camera] {cam_w}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} (requested {args.width}x{args.height})")
    # pixel constants were tuned at 640 px wide; scale them so lock/motion behave the same at any resolution
    px_scale = cam_w / 640.0 if cam_w > 0 else 1.0
    lock_radius = LOCK_MATCH_RADIUS * px_scale
    move_speed_px = MOVE_SPEED_PX * px_scale
    time.sleep(0.5)
    for _ in range(10):
        cap.read()

    locked_name: Optional[str] = None
    locked_center: Optional[np.ndarray] = None
    missed_frames = 0
    target_angle = 90.0
    last_track_t = 0.0
    prev_x: Optional[float] = None
    vel_x = 0.0                 # EMA of the locked face's horizontal speed, px/frame
    last_motion = ""

    print(f"Loaded identities: {matcher.names}")
    print(f"mode={args.mode}  q=quit f=fullscreen i=invert needle l=release lock c=centre h=home r=release coils +/- threshold")

    # resizable window, opened as large as the screen allows while keeping the camera's aspect ratio
    win = "face_track_mqtt"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    scr_w, scr_h = screen_size()
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fit = min((scr_w * 0.95) / max(cam_w, 1), (scr_h * 0.88) / cam_h)   # leave room for title bar + taskbar
    cv2.resizeWindow(win, int(cam_w * fit), int(cam_h * fit))
    fullscreen = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            H, W = frame.shape[:2]
            # Detection/recognition run on the raw frame (enrolled crops are unmirrored); only the
            # display, LEFT/RIGHT labels and needle direction use the mirrored view, so the person,
            # their face on screen and the needle all move the same way.
            mirror = not args.no_mirror
            sx = (lambda x: W - x) if mirror else (lambda x: x)
            vis = cv2.flip(frame, 1) if mirror else frame.copy()
            found_this_frame = False

            faces = det.detect(frame, max_faces=1)
            if faces:
                f = faces[0]
                center = face_center(f)
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                m = matcher.match(embedder.embed(aligned).embedding)
                name, dist, accepted = m.name, m.distance, m.accepted

                if locked_name is None:
                    if accepted:
                        locked_name, locked_center = name, center
                        missed_frames = 0
                        found_this_frame = True
                        print(f"[lock] acquired '{locked_name}'")
                elif accepted and name == locked_name and \
                        np.linalg.norm(center - locked_center) < lock_radius:
                    locked_center = center
                    missed_frames = 0
                    found_this_frame = True

                color = (0, 255, 0) if accepted else (0, 0, 255)
                bx1, bx2 = sorted((int(sx(f.x1)), int(sx(f.x2))))
                cv2.rectangle(vis, (bx1, f.y1), (bx2, f.y2), color, 2)
                cv2.putText(vis, f"{name or 'Unknown'} dist={dist:.3f}", (bx1, max(0, f.y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if locked_name is not None and not found_this_frame:
                missed_frames += 1
                if missed_frames > LOCK_LOSS_FRAMES:
                    print(f"[lock] lost '{locked_name}' -> resuming sweep")
                    locked_name, locked_center, missed_frames = None, None, 0
                    sweeper.reset_from(target_angle)

            if locked_name is not None and locked_center is not None:
                x = float(sx(locked_center[0]))          # position as shown on screen
                norm = float(np.clip((x - W / 2.0) / (W / 2.0), -1.0, 1.0))
                if args.mode == "fixed":
                    angle = 90 - sign * norm * 60
                    if abs(angle - target_angle) >= MIN_ANGLE_DELTA:
                        target_angle = angle
                else:
                    now = time.time()
                    if abs(norm) > TRACK_DEADBAND and now - last_track_t >= TRACK_UPDATE_S and found_this_frame:
                        target_angle = float(np.clip(target_angle - sign * norm * TRACK_GAIN_DEG,
                                                     ANGLE_MIN, ANGLE_MAX))
                        last_track_t = now
                motor.set_angle(target_angle)

                # face position + movement, as shown on screen
                if prev_x is not None and found_this_frame:
                    vel_x = 0.6 * vel_x + 0.4 * (x - prev_x)
                prev_x = x
                side = "CENTER" if abs(norm) < MOVE_CENTER_ZONE else ("LEFT" if norm < 0 else "RIGHT")
                motion = "still" if abs(vel_x) < move_speed_px else ("moving LEFT" if vel_x < 0 else "moving RIGHT")
                if motion != last_motion:
                    print(f"[face] {motion}  (at {side}, motor -> {target_angle:.0f} deg)")
                    last_motion = motion

                cv2.circle(vis, (int(x), int(locked_center[1])), 6, (255, 0, 255), -1)
                cv2.putText(vis, f"FACE: {side} | {motion}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if motion != "still" else (255, 255, 255), 2)
                cv2.putText(vis, f"LOCKED: {locked_name}  offset={norm:+.2f}  angle={target_angle:.0f}",
                            (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            else:
                prev_x, vel_x, last_motion = None, 0.0, ""
                target_angle = sweeper.step()
                motor.set_angle(target_angle)
                cv2.putText(vis, f"searching... angle={target_angle:.0f}", (10, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            net = f"MQTT {'OK' if motor.connected else 'DOWN'} | ESP8266: {motor.device_status}"
            if motor.device_position is not None:
                net += f" @ {motor.device_position} deg"
            cv2.putText(vis, net, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if motor.connected and motor.device_status == "online" else (0, 0, 255), 2)
            cv2.line(vis, (W // 2, 0), (W // 2, H), (100, 100, 100), 1)
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("i"):
                sign = -sign
                print(f"[needle] direction {'inverted' if sign < 0 else 'normal'}")
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if key == ord("l"):
                locked_name, locked_center, missed_frames = None, None, 0
                sweeper.reset_from(target_angle)
            if key == ord("c"):
                target_angle = 90.0
                sweeper.reset_from(target_angle)
            if key == ord("h"):
                motor.send_cmd("home")
                target_angle = 90.0
                motor.last_angle = 90
                sweeper.reset_from(target_angle)
            if key == ord("r"):
                motor.send_cmd("release")
            if key in (ord("+"), ord("=")):
                matcher.threshold = min(1.2, matcher.threshold + 0.01)
            if key == ord("-"):
                matcher.threshold = max(0.05, matcher.threshold - 0.01)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        motor.close()


if __name__ == "__main__":
    main()
