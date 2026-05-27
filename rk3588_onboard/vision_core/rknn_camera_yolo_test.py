from pathlib import Path
import time
import json
import signal
import sys

import cv2
import numpy as np
from rknnlite.api import RKNNLite


# =========================
# Basic config
# =========================

MODEL_PATH = Path("/home/marvsmart/animal_patrol/models/best_fp.rknn")
CAMERA_DEVICE = "/dev/video12"

INPUT_SIZE = 640
CONF_THRES = 0.30
IOU_THRES = 0.45

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Log once per second
LOG_INTERVAL_SEC = 1.0

# Model class order:
# 0: peacock
# 1: tiger
# 2: elephant
# 3: wolf
# 4: monkey
CLASS_NAMES = {
    0: "peacock",
    1: "tiger",
    2: "elephant",
    3: "wolf",
    4: "monkey",
}

NUM_CLASSES = len(CLASS_NAMES)

running = True


# =========================
# Signal handling
# =========================

def signal_handler(sig, frame):
    global running
    print("\nReceived stop signal, exiting...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =========================
# Image preprocessing
# =========================

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize image while keeping aspect ratio, then pad to target size.
    Returns:
        padded image
        resize ratio
        padding offset: (dw, dh)
    """
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    r = min(new_w / w, new_h / h)

    resized_w = int(round(w * r))
    resized_h = int(round(h * r))

    dw = new_w - resized_w
    dh = new_h - resized_h
    dw /= 2
    dh /= 2

    if (w, h) != (resized_w, resized_h):
        img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return img, r, (dw, dh)


def preprocess(frame):
    """
    Convert OpenCV BGR frame to RKNN input.
    Current input format: NHWC uint8 RGB.
    """
    img, ratio, pad = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0).astype(np.uint8)
    return input_data, ratio, pad


# =========================
# YOLOv8 postprocess
# =========================

def xywh_to_xyxy(boxes):
    xyxy = np.zeros_like(boxes)

    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    return xyxy


def postprocess(outputs, ratio, pad, original_shape):
    """
    Supports common YOLOv8 output:
        [1, 9, 8400]
        [1, 8400, 9]

    where:
        9 = 4 box values + 5 class scores
    """
    if outputs is None or len(outputs) == 0:
        return []

    out = np.array(outputs[0])
    out = np.squeeze(out)

    if out.ndim != 2:
        print("Unsupported output dimension:", out.shape)
        return []

    expected_dim = 4 + NUM_CLASSES

    if out.shape[0] == expected_dim:
        pred = out.T
    elif out.shape[1] == expected_dim:
        pred = out
    else:
        print("Unsupported YOLO output shape:", out.shape)
        print("Expected one dimension to be:", expected_dim)
        return []

    boxes_xywh = pred[:, 0:4]
    class_scores = pred[:, 4:4 + NUM_CLASSES]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences >= CONF_THRES
    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes_xywh) == 0:
        return []

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    dw, dh = pad

    # Restore box coordinates from letterbox image to original image
    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy /= ratio

    orig_h, orig_w = original_shape[:2]

    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w - 1)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h - 1)

    nms_boxes = []

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        nms_boxes.append([
            int(x1),
            int(y1),
            int(max(0, x2 - x1)),
            int(max(0, y2 - y1)),
        ])

    indices = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes,
        scores=confidences.astype(float).tolist(),
        score_threshold=CONF_THRES,
        nms_threshold=IOU_THRES,
    )

    detections = []

    if len(indices) == 0:
        return detections

    indices = np.array(indices).reshape(-1)

    for i in indices:
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        cls_id = int(class_ids[i])
        conf = float(confidences[i])
        name = CLASS_NAMES.get(cls_id, str(cls_id))

        detections.append({
            "class_id": cls_id,
            "name": name,
            "conf": round(conf, 4),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    return detections


def build_summary(detections):
    summary = {}

    for det in detections:
        name = det["name"]
        summary[name] = summary.get(name, 0) + 1

    return summary


# =========================
# Main
# =========================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"RKNN model not found: {MODEL_PATH}")

    rknn = RKNNLite()

    print("--> load rknn model")
    ret = rknn.load_rknn(str(MODEL_PATH))
    print("load_rknn ret =", ret)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed, ret={ret}")

    print("--> init runtime")
    try:
        core_mask = RKNNLite.NPU_CORE_0_1_2
        print("Using NPU core mask: NPU_CORE_0_1_2")
    except AttributeError:
        core_mask = RKNNLite.NPU_CORE_0
        print("NPU_CORE_0_1_2 not available, fallback to NPU_CORE_0")

    ret = rknn.init_runtime(core_mask=core_mask)
    print("init_runtime ret =", ret)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed, ret={ret}")

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed: {CAMERA_DEVICE}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("Camera opened")
    print("Actual width:", actual_width)
    print("Actual height:", actual_height)
    print("Actual FPS setting:", actual_fps)
    print("Headless recognition started. Press Ctrl+C to stop.")

    frame_count_total = 0
    infer_count_total = 0
    fail_count_total = 0

    frame_count_window = 0
    infer_count_window = 0

    first_output_printed = False

    start_time = time.time()
    last_log_time = start_time
    last_detections = []
    last_summary = {}

    try:
        while running:
            ret, frame = cap.read()

            if not ret or frame is None:
                fail_count_total += 1
                print(json.dumps({
                    "level": "warning",
                    "event": "camera_read_failed",
                    "fail_count_total": fail_count_total,
                    "timestamp": round(time.time(), 3),
                }, ensure_ascii=False))

                if fail_count_total >= 30:
                    raise RuntimeError("Camera read failed too many times")

                continue

            frame_count_total += 1
            frame_count_window += 1

            input_data, ratio, pad = preprocess(frame)

            t0 = time.time()
            outputs = rknn.inference(inputs=[input_data])
            infer_ms = (time.time() - t0) * 1000.0

            infer_count_total += 1
            infer_count_window += 1

            if not first_output_printed:
                print("First output shape:")
                for i, out in enumerate(outputs):
                    arr = np.array(out)
                    print(
                        f"output[{i}] shape={arr.shape}, "
                        f"dtype={arr.dtype}, "
                        f"min={arr.min()}, "
                        f"max={arr.max()}"
                    )
                first_output_printed = True

            detections = postprocess(outputs, ratio, pad, frame.shape)
            summary = build_summary(detections)

            last_detections = detections
            last_summary = summary

            now = time.time()
            log_dt = now - last_log_time

            if log_dt >= LOG_INTERVAL_SEC:
                elapsed = now - start_time

                # Because every loop does one inference, fps and infer_fps will usually be close.
                fps = frame_count_window / log_dt
                infer_fps = infer_count_window / log_dt

                log_item = {
                    "timestamp": round(now, 3),
                    "elapsed_s": round(elapsed, 1),
                    "fps": round(fps, 2),
                    "infer_fps": round(infer_fps, 2),
                    "last_infer_ms": round(infer_ms, 2),
                    "frame_count_total": frame_count_total,
                    "infer_count_total": infer_count_total,
                    "fail_count_total": fail_count_total,
                    "summary": last_summary,
                    "detections": last_detections,
                }

                print(json.dumps(log_item, ensure_ascii=False))

                frame_count_window = 0
                infer_count_window = 0
                last_log_time = now

    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping...")

    finally:
        cap.release()
        rknn.release()

        total_elapsed = time.time() - start_time

        final_log = {
            "event": "finished",
            "elapsed_s": round(total_elapsed, 1),
            "frame_count_total": frame_count_total,
            "infer_count_total": infer_count_total,
            "fail_count_total": fail_count_total,
        }

        print(json.dumps(final_log, ensure_ascii=False))
        print("Test finished")


if __name__ == "__main__":
    main()