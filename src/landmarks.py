import time

import cv2
import numpy as np
import mediapipe as mp

IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291


def main():
    cascade_path = "models/haarcascade_frontalface_default.xml"
    face = cv2.CascadeClassifier(cascade_path)
    if face.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = None
    else:
        from pathlib import Path
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        base_options = mp_python.BaseOptions(model_asset_path="models/face_landmarker.task")
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        fm = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")

    time.sleep(0.5)
    for i in range(10):
        ok, f = cap.read()
        if ok:
            print(f"warmup read {i}: ok={ok} shape={f.shape} mean_pixel={f.mean():.1f}")
        else:
            print(f"warmup read {i}: ok={ok}")
        time.sleep(0.1)

    print("Haar + FaceMesh 5pt (minimal). Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm_pts = None
        if fm is not None:
            res = fm.process(rgb)
            if res.multi_face_landmarks:
                lm_pts = [[p.x * W, p.y * H] for p in res.multi_face_landmarks[0].landmark]
        elif landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)
            if res.face_landmarks:
                lm_pts = [[p.x * W, p.y * H] for p in res.face_landmarks[0]]

        if lm_pts:
            idxs = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]
            pts = [lm_pts[i] for i in idxs]
            kps = np.array(pts, dtype=np.float32)  # (5,2)

            # enforce left/right ordering
            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]
            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 3]]

            for (px, py) in kps.astype(int):
                cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)

            cv2.putText(frame, "5pt", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("5pt Landmarks", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()