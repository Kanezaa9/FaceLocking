
# mqtt_tracking/face_track_mqtt_vertical.py
"""
Vertical (tilt) face tracking over MQTT for a second 28BYJ-48 on the same ESP8266.

Same detection / recognition / search-and-lock logic as face_track_mqtt.py, but
the locked face's VERTICAL position drives the tilt motor instead of the pan
motor. Angles are published to <base>/angle (base defaults to "face_tracking/tilt");
the ESP8266 running esp8266_stepper_mqtt_dual.ino subscribes and turns the second
ULN2003/28BYJ-48.

Flash the dual firmware first (it still drives the pan motor on the original
"face_tracking" topics), then run, from the repo root, with the broker up:

    python -m mqtt_tracking.face_track_mqtt_vertical --host 127.0.0.1
    python -m mqtt_tracking.face_track_mqtt_vertical --invert        # if the tilt turns the wrong way

Keys (same as face_track_mqtt.py):
    q : quit            f : fullscreen      i : invert direction
    l : release lock    c : centre (90)     h : home (here = 90)   r : release coils
    +/- : adjust recognition threshold
"""
from __future__ import annotations

import argparse
import math
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
from .face_track_mqtt import (
    ANGLE_MAX, ANGLE_MIN, MOVE_CENTER_ZONE, MOVE_SPEED_PX,
    TRACK_DEADBAND, TRACK_GAIN_DEG, TRACK_UPDATE_S, screen_size,
)
from .mqtt_publisher import MqttAngleController

NEEDLE_DIAL_FRAC = 0.16
NEEDLE_DIAL_TICKS = 12
NEEDLE_HALF_WIDTH = 4


def draw_tacho_needle(img, pivot, tip, color=(0, 255, 255)):
    """Tacho dial at pivot with a tapered needle whose tip is the target's nose.

    Returns (bearing_deg, length_px); bearing is 0 straight up, clockwise positive.
    Only the overlay is drawn - the full, uncropped camera frame stays visible.
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
    ap.add_argument("--host", default="127.0.0.1", help="MQTT broker address")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="face_tracking/tilt", help="base topic for the tilt axis")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280, help="requested camera width")
    ap.add_argument("--height", type=int, default=720, help="requested camera height")
    ap.add_argument("--mode", choices=["onboard", "fixed"], default="fixed")
    ap.add_argument("--invert", action="store_true", help="flip direction if the tilt turns away from the face")
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
    px_scale = cam_w / 640.0 if cam_w > 0 else 1.0
    lock_radius = LOCK_MATCH_RADIUS * px_scale
    move_speed_px = MOVE_SPEED_PX * px_scale
    time.sleep(0.5)
    for _ in range(10):
        cap.read()

    locked_name: Optional[str] = None
    locked_center: Optional[np.ndarray] = None
    locked_nose: Optional[np.ndarray] = None
    missed_frames = 0
    target_angle = 90.0
    last_track_t = 0.0
    prev_y: Optional[float] = None
    vel_y = 0.0
    last_motion = ""

    print(f"Loaded identities: {matcher.names}")
    print(f"mode={args.mode} tilt  q=quit f=fullscreen i=invert l=release c=centre h=home r=release coils +/- threshold")

    win = "face_track_mqtt_vertical"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    scr_w, scr_h = screen_size()
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fit = min((scr_w * 0.95) / max(cam_w, 1), (scr_h * 0.88) / cam_h)
    cv2.resizeWindow(win, int(cam_w * fit), int(cam_h * fit))
    fullscreen = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            H, W = frame.shape[:2]
            mirror = not args.no_mirror
            sx = (lambda x: W - x) if mirror else (lambda x: x)
            vis = cv2.flip(frame, 1) if mirror else frame.copy()
            found_this_frame = False

            faces = det.detect(frame, max_faces=1)
            if faces:
                f = faces[0]
                center = face_center(f)
                nose = f.kps[2].astype(np.float32)
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                m = matcher.match(embedder.embed(aligned).embedding)
                name, dist, accepted = m.name, m.distance, m.accepted

                if locked_name is None:
                    if accepted:
                        locked_name, locked_center, locked_nose = name, center, nose
                        missed_frames = 0
                        found_this_frame = True
                        print(f"[lock] acquired '{locked_name}'")
                elif accepted and name == locked_name and \
                        np.linalg.norm(center - locked_center) < lock_radius:
                    locked_center, locked_nose = center, nose
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
                    locked_name, locked_center, locked_nose, missed_frames = None, None, None, 0
                    sweeper.reset_from(target_angle)

            if locked_name is not None and locked_nose is not None:
                y = float(locked_nose[1])
                norm = float(np.clip((y - H / 2.0) / (H / 2.0), -1.0, 1.0))
                if args.mode == "fixed":
                    angle = 90 + sign * norm * 60
                    if abs(angle - target_angle) >= MIN_ANGLE_DELTA:
                        target_angle = angle
                else:
                    now = time.time()
                    if abs(norm) > TRACK_DEADBAND and now - last_track_t >= TRACK_UPDATE_S and found_this_frame:
                        target_angle = float(np.clip(target_angle - sign * norm * TRACK_GAIN_DEG,
                                                     ANGLE_MIN, ANGLE_MAX))
                        last_track_t = now
                motor.set_angle(target_angle)

                if prev_y is not None and found_this_frame:
                    vel_y = 0.6 * vel_y + 0.4 * (y - prev_y)
                prev_y = y
                side = "CENTER" if abs(norm) < MOVE_CENTER_ZONE else ("UP" if norm < 0 else "DOWN")
                motion = "still" if abs(vel_y) < move_speed_px else ("moving UP" if vel_y < 0 else "moving DOWN")
                if motion != last_motion:
                    print(f"[face] {motion}  (at {side}, tilt -> {target_angle:.0f} deg)")
                    last_motion = motion

                cx, cy = W // 2, H // 2
                r = int(min(W, H) * NEEDLE_DIAL_FRAC)
                needle_rad = math.radians(target_angle - 90.0)
                tip_x = int(cx + (r * 0.90) * math.sin(needle_rad))
                tip_y = int(cy - (r * 0.90) * math.cos(needle_rad))
                draw_tacho_needle(vis, (cx, cy), (tip_x, tip_y))
                cv2.circle(vis, (int(locked_nose[0]), int(y)), 5, (0, 255, 255), -1)
                cv2.putText(vis, f"FACE: {side} | {motion}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if motion != "still" else (255, 255, 255), 2)
                cv2.putText(vis, f"LOCKED: {locked_name}  offset={norm:+.2f}  tilt={target_angle:.0f}",
                            (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            else:
                prev_y, vel_y, last_motion = None, 0.0, ""
                target_angle = sweeper.step()
                motor.set_angle(target_angle)
                cx, cy = W // 2, H // 2
                r = int(min(W, H) * NEEDLE_DIAL_FRAC)
                needle_rad = math.radians(target_angle - 90.0)
                tip_x = int(cx + (r * 0.90) * math.sin(needle_rad))
                tip_y = int(cy - (r * 0.90) * math.cos(needle_rad))
                draw_tacho_needle(vis, (cx, cy), (tip_x, tip_y), color=(0, 165, 255))
                cv2.putText(vis, f"searching... tilt={target_angle:.0f}", (10, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            net = f"MQTT {'OK' if motor.connected else 'DOWN'} | ESP8266: {motor.device_status}"
            if motor.device_position is not None:
                net += f" @ {motor.device_position} deg"
            cv2.putText(vis, net, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if motor.connected and motor.device_status == "online" else (0, 0, 255), 2)
            cv2.line(vis, (0, H // 2), (W, H // 2), (100, 100, 100), 1)
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("i"):
                sign = -sign
                print(f"[tilt] direction {'inverted' if sign < 0 else 'normal'}")
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if key == ord("l"):
                locked_name, locked_center, locked_nose, missed_frames = None, None, None, 0
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
