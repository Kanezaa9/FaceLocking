from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class BodyDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    track_id: Optional[int] = None

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


class YoloXBodyDetector:
    def __init__(
        self,
        model_path: str | Path = "models/yolox_nano.onnx",
        input_size: Tuple[int, int] = (416, 416),
        score_thresh: float = 0.35,
        nms_thresh: float = 0.45,
        num_threads: int = 4,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLOX model not found at {self.model_path}")

        self.input_size = input_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self._grids, self._strides = self._build_grids_and_strides(self.input_size)

    @staticmethod
    def _build_grids_and_strides(img_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        grids = []
        expanded_strides = []
        strides = [8, 16, 32]
        hsizes = [img_size[0] // s for s in strides]
        wsizes = [img_size[1] // s for s in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((grid.shape[0], 1), stride))

        grids_arr = np.concatenate(grids, axis=0)
        strides_arr = np.concatenate(expanded_strides, axis=0)
        return grids_arr, strides_arr

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        ih, iw = self.input_size
        h, w = img.shape[:2]
        r = min(ih / h, iw / w)
        rw, rh = int(w * r), int(h * r)

        padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
        resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        padded[:rh, :rw] = resized

        tensor = padded.transpose((2, 0, 1)).astype(np.float32)
        tensor = np.ascontiguousarray(tensor)
        return tensor, r

    def _decode_and_filter(
        self, outputs: np.ndarray, ratio: float, orig_w: int, orig_h: int
    ) -> List[BodyDetection]:
        pred = outputs[0]
        boxes_raw = pred[:, :4]
        obj_conf = pred[:, 4]
        cls_conf = pred[:, 5]

        scores = obj_conf * cls_conf
        mask = scores > self.score_thresh
        if not np.any(mask):
            return []

        sub_boxes = boxes_raw[mask].copy()
        sub_scores = scores[mask]
        sub_grids = self._grids[mask]
        sub_strides = self._strides[mask]

        sub_boxes[:, :2] = (sub_boxes[:, :2] + sub_grids) * sub_strides
        sub_boxes[:, 2:4] = np.exp(sub_boxes[:, 2:4]) * sub_strides

        cx = sub_boxes[:, 0]
        cy = sub_boxes[:, 1]
        bw = sub_boxes[:, 2]
        bh = sub_boxes[:, 3]

        x1 = np.clip((cx - bw / 2.0) / ratio, 0, orig_w - 1)
        y1 = np.clip((cy - bh / 2.0) / ratio, 0, orig_h - 1)
        x2 = np.clip((cx + bw / 2.0) / ratio, 0, orig_w - 1)
        y2 = np.clip((cy + bh / 2.0) / ratio, 0, orig_h - 1)

        nms_boxes = []
        for a, b, c, d in zip(x1, y1, x2, y2):
            nms_boxes.append([int(a), int(b), int(c - a), int(d - b)])

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            sub_scores.tolist(),
            score_threshold=self.score_thresh,
            nms_threshold=self.nms_thresh,
        )

        detections: List[BodyDetection] = []
        if len(indices) > 0:
            for idx in np.asarray(indices).reshape(-1):
                detections.append(
                    BodyDetection(
                        x1=int(x1[idx]),
                        y1=int(y1[idx]),
                        x2=int(x2[idx]),
                        y2=int(y2[idx]),
                        score=float(sub_scores[idx]),
                    )
                )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def detect(self, frame: np.ndarray) -> List[BodyDetection]:
        h, w = frame.shape[:2]
        tensor, ratio = self._preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: tensor[None, ...]})[0]
        return self._decode_and_filter(outputs, ratio, w, h)
