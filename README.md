# Face Recognition 5-Point Landmark System

A real-time face recognition pipeline built with OpenCV, MediaPipe, and ArcFace. Designed for embedded use with servo-based face tracking via ESP8266.

## How It Works (Pipeline)

The system goes through **6 stages** to recognize a face:

```
Camera → Detection → 5-Point Landmarks → Alignment → Embedding → Matching → Output
```

### Stage 1: Face Detection (`src/detect.py`)
Uses OpenCV's **Haar Cascade** classifier (`haarcascade_frontalface_default.xml`) to detect face bounding boxes in each camera frame. Runs on grayscale with `scaleFactor=1.1`, `minNeighbors=5`, and minimum face size of 60×60 pixels.

### Stage 2: 5-Point Landmark Extraction (`src/landmarks.py`, `src/haar_5pt.py`)
For each detected face, **MediaPipe FaceMesh** extracts 5 key facial landmarks:
- **Left eye** (index 33)
- **Right eye** (index 263)
- **Nose tip** (index 1)
- **Mouth left** (index 61)
- **Mouth right** (index 291)

Points are normalized to pixel coordinates and enforce left/right ordering. The `Haar5ptDetector` class cross-validates landmarks against the Haar bounding box and rejects inconsistent detections.

### Stage 3: Face Alignment (`src/align.py`, `src/haar_5pt.py`)
The 5 landmarks are mapped to a **standard ArcFace template** (canonical 112×112 positions). An affine transform is computed via `cv2.estimateAffinePartial2D` and the face is warped into a normalized 112×112 crop. This ensures the face is always upright, centered, and scale-invariant.

Reference template positions (from InsightFace/ArcFace):
```
Left eye:      (38.29, 51.70)
Right eye:     (73.53, 51.50)
Nose:          (56.03, 71.74)
Left mouth:    (41.55, 92.37)
Right mouth:   (70.73, 92.20)
```

### Stage 4: Embedding (`src/embed.py`)
The aligned 112×112 face is fed through an **ArcFace ONNX model** (`models/embedder_arcface.onnx`) running on CPU via ONNX Runtime. Preprocessing normalizes pixel values to `[-1, 1]`. The model outputs a **512-dimensional L2-normalized embedding vector** that uniquely represents the face.

### Stage 5: Enrollment (`src/enroll.py`)
To register a new person:
1. Enter a name when prompted
2. Capture face samples (manually with SPACE, or auto-capture every ~0.25s with `a`)
3. Each frame: detect → align → embed
4. Save at least 15 samples, then press `s` to save
5. All sample embeddings are **averaged into a single template** and L2-normalized
6. Saved to `data/db/face_db.npz` (embeddings) and `data/db/face_db.json` (metadata)
7. Aligned crops saved to `data/enroll/<NAME>/`

### Stage 6: Recognition (`src/recognize.py`, `src/recognize_track.py`)
Live recognition pipeline:
1. Camera frame → detect face → align → embed
2. Compute **cosine similarity** against all enrolled templates
3. If best match distance ≤ threshold (default **0.34**), label the face; otherwise "Unknown"
4. Distance = `1 - cosine_similarity`

### Evaluation (`src/evaluate.py`)
Computes genuine (same person) vs impostor (different person) distance distributions from enrolled crops, then sweeps thresholds to find the optimal operating point at 1% false-accept rate (FAR).

### Servo Tracking (`src/recognize_track.py`, `src/servo_control.py`)
When enabled, the system can:
- **Search mode**: Sweep a pan servo (connected to ESP8266 via serial) left-right to find faces
- **Lock mode**: When a recognized face is found, lock onto it and track by adjusting servo angle based on horizontal offset from frame center
- **Loss recovery**: If the locked face is lost for 15+ frames, release lock and resume sweeping

Servo commands are sent over serial (default COM5, 9600 baud) with ESP8266 boot settle handling.

## Project Structure

```
face-recognition-5pt/
├── models/
│   ├── haarcascade_frontalface_default.xml   # Haar cascade for face detection
│   └── embedder_arcface.onnx                 # ArcFace ONNX embedding model
├── data/
│   ├── db/
│   │   ├── face_db.npz                       # Enrolled face embeddings
│   │   └── face_db.json                      # Enrollment metadata
│   └── enroll/<NAME>/*.jpg                   # Aligned face crops per person
└── src/
    ├── camera.py          # Camera test utility
    ├── detect.py          # Step 1: Haar face detection
    ├── landmarks.py       # Step 2: 5-point landmark extraction
    ├── haar_5pt.py        # Core: Haar + FaceMesh detector + alignment math
    ├── align.py           # Step 3: Face alignment demo
    ├── embed.py           # Step 4: ArcFace embedding
    ├── enroll.py          # Step 5: Enrollment (capture + save template)
    ├── recognize.py       # Step 6: Live recognition
    ├── recognize_track.py # Recognition + servo tracking
    ├── evaluate.py        # Threshold evaluation (genuine/impostor)
    ├── servo_control.py   # ESP8266 serial servo controller
    ├── yolox.py           # CPU YOLOX ONNX pedestrian/body detector
    ├── body_features.py   # Multi-zone body feature extraction & online learning
    ├── tracker.py         # Body tracking & face-to-body containment association
    └── body_lock.py       # Full-body tracking + 180-deg rotation re-identification
```

## Setup — Files You Need to Download

Some files are **not** committed to this repo (too large for GitHub, or generated on your machine). You must set these up before running anything:

### 1. ArcFace ONNX model (REQUIRED — not in the repo)

`models/embedder_arcface.onnx` (~166 MB) exceeds GitHub's 100 MB per-file limit, so it is **not included**. The code expects it at:

```
models/embedder_arcface.onnx
```

Download it (or export it from PyTorch/InsightFace) and place it there. It must accept a 3×112×112 RGB input and output a 512-D embedding vector. Good sources:

- **InsightFace Model Zoo** — https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch (export the MXNet/PyTorch backbone to ONNX)
- Existing public ArcFace ONNX exports from sources such as `face.evoLVe.PyTorch` or `InsightFace` repos

Verify it works by running `python -m src.embed` — you should see `[embed] model loaded` and input/output shapes `[1,3,112,112]` → `[1,512]`.

### 2. Haar cascade (included)

`models/haarcascade_frontalface_default.xml` **is** committed (it's small), so nothing to do.

### 3. Enrollment database (generated)

`data/db/face_db.npz` + `data/db/face_db.json` and the `data/enroll/*.jpg` crops are created by you when you run `python -m src.enroll`. They are git-ignored.

## Requirements

- Python 3.11+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- ONNX Runtime (`onnxruntime`)
- NumPy (`numpy`)
- pyserial (`pyserial`) — for servo tracking only

```bash
pip install opencv-python mediapipe onnxruntime numpy pyserial
```

## Usage

Each module runs as a standalone script:

```bash
# Camera test
python -m src.camera

# Face detection only (Haar)
python -m src.detect

# 5-point landmarks (Haar + MediaPipe)
python -m src.landmarks

# Alignment demo
python -m src.align

# Embedding demo
python -m src.embed

# Enroll a new person
python -m src.enroll

# Live recognition
python -m src.recognize

# Recognition + servo tracking
python -m src.recognize_track

# Face locking + position + expressions + blinks (no servo)
python -m src.face_lock --name ATUMANYIRE --camera 1

# Full body tracking + face association + 180-deg rotation recognition (CPU-optimized YOLOX)
python -m src.body_lock --name Tehila --camera 0

# Evaluate threshold
python -m src.evaluate
```

## Workflow

1. **Test camera** — `python -m src.camera`
2. **Verify detection** — `python -m src.detect`
3. **Check landmarks** — `python -m src.landmarks`
4. **Enroll person A** — `python -m src.enroll`
5. **Enroll person B** — `python -m src.enroll`
6. **Evaluate threshold** — `python -m src.evaluate`
7. **Run recognition** — `python -m src.recognize`
8. **(Optional) Servo tracking** — `python -m src.recognize_track`

## Controls (recognize.py / recognize_track.py)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reload database from disk |
| `+` / `-` | Loosen / tighten recognition threshold |
| `l` | Release servo lock (tracking mode only) |

## Controls (face_lock.py)

```bash
python -m src.face_lock --name ATUMANYIRE --camera 0  # built-in webcam
python -m src.face_lock --name ATUMANYIRE --camera 1  # external camera
python -m src.face_lock --name ATUMANYIRE --camera 1 --history data/action_history.jsonl
```

Labels every face in the frame (enrolled name or **Unknown**) and locks onto the target person (no servo). While locked it shows:
- **Position** of the nose tip relative to the frame centre: LEFT / RIGHT / UP / DOWN / CENTER (as seen on screen), plus dx, dy and distance in pixels. Side changes are printed to the console.
- **Expression**: neutral, smiling, sad, frowning, surprised, grimacing.
- **Blinks** with a running count.
- An **orange** warning while the locked face is briefly missing and a **red** warning when the target is not in the frame.

Each session appends JSON Lines events to `data/action_history.jsonl`. Every entry includes `timestamp`, `action_type`, `description`, and the locked `identity` when available. Movement, expression changes, blinks, and lock state changes are recorded as they occur.

Faces that are not enrolled are boxed and labelled `Unknown`; an enrolled person other than `--name` is labelled with their own name. Neither can acquire or steal the selected identity lock. When the selected identity is absent, the banner reads `WARNING: <name> NOT FOUND`.

Hold a neutral face with eyes open for ~1 s right after locking while expression/blink baselines calibrate.

| Key | Action |
|-----|--------|
| `q` | Quit |
| `l` | Release lock |
| `c` | Recalibrate neutral expression and reset blink count |
| `d` | Toggle expression debug values (for tuning thresholds in `src/expression.py`) |
| `r` | Reload database from disk |
| `+` / `-` | Loosen / tighten recognition threshold |

## Controls (enroll.py)

| Key | Action |
|-----|--------|
| `SPACE` | Capture one sample |
| `a` | Toggle auto-capture |
| `s` | Save enrollment (needs 5+ samples) |
| `r` | Reset captured samples |
| `q` | Quit |

## Hardware (Servo Tracking)

- **MCU**: ESP8266 (or compatible) on serial port
- **Servo**: Standard pan servo (0-180 degrees)
- **Connection**: Serial at 9600 baud (default COM5)
- The system handles ESP8266 DTR-reset boot timing automatically

## License

Group / Academic project.
