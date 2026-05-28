from pathlib import Path
import time
import json
import signal
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from rknnlite.api import RKNNLite


# =========================
# Basic config
# =========================

MODEL_PATH = Path("/home/marvsmart/animal_patrol/models/best_416_fp.rknn")
CAMERA_DEVICE = "/dev/video12"

INPUT_SIZE = 416
CONF_THRES = 0.30
IOU_THRES = 0.45

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

LOG_INTERVAL_SEC = 1.0

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
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def safe_print(obj):
    try:
        print(json.dumps(obj, ensure_ascii=False), flush=True)
    except BrokenPipeError:
        sys.exit(0)


# =========================
# Image preprocessing
# =========================

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
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
    if outputs is None or len(outputs) == 0:
        return []

    out = np.array(outputs[0])
    out = np.squeeze(out)

    if out.ndim != 2:
        return []

    expected_dim = 4 + NUM_CLASSES

    if out.shape[0] == expected_dim:
        pred = out.T
    elif out.shape[1] == expected_dim:
        pred = out
    else:
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
            "conf": conf,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    return detections


# =========================
# Single-species rule
# =========================

def apply_single_species_rule(detections):
    """
    Business rule:
    In one recognition area, only one animal species is allowed.
    Multiple animals of the same species are allowed.
    If multiple classes are detected in one frame, keep only the dominant class.

    Dominant class priority:
    1. larger detection count
    2. larger confidence sum
    3. larger max confidence
    """
    if not detections:
        return "none", 0, []

    class_stats = {}

    for det in detections:
        cls_id = int(det["class_id"])
        conf = float(det["conf"])

        if cls_id not in class_stats:
            class_stats[cls_id] = {
                "count": 0,
                "conf_sum": 0.0,
                "max_conf": 0.0,
            }

        class_stats[cls_id]["count"] += 1
        class_stats[cls_id]["conf_sum"] += conf
        class_stats[cls_id]["max_conf"] = max(class_stats[cls_id]["max_conf"], conf)

    dominant_cls = max(
        class_stats.keys(),
        key=lambda cls_id: (
            class_stats[cls_id]["count"],
            class_stats[cls_id]["conf_sum"],
            class_stats[cls_id]["max_conf"],
        )
    )

    filtered = [
        det for det in detections
        if int(det["class_id"]) == int(dominant_cls)
    ]

    animal = CLASS_NAMES.get(int(dominant_cls), str(dominant_cls))
    count = len(filtered)

    return animal, count, filtered


def update_window_stats(window_stats, animal, count):
    """
    Accumulate recognition result in one logging window.
    """
    window_stats["total_frames"] += 1

    if animal == "none" or count <= 0:
        window_stats["none_frames"] += 1
        return

    window_stats["animal_hit_frames"][animal] += 1
    window_stats["animal_count_hist"][animal][count] += 1


def get_window_result(window_stats):
    """
    Select dominant animal in current logging window.

    Rule:
    1. Choose animal with most hit frames.
    2. If tie, choose animal with larger total count evidence.
    3. Count uses the most frequent count value of that animal.
    """
    animal_hit_frames = window_stats["animal_hit_frames"]

    if not animal_hit_frames:
        return "none", 0

    def animal_score(animal):
        hit_frames = animal_hit_frames[animal]
        count_evidence = sum(
            count * freq
            for count, freq in window_stats["animal_count_hist"][animal].items()
        )
        return hit_frames, count_evidence

    dominant_animal = max(animal_hit_frames.keys(), key=animal_score)

    count_hist = window_stats["animal_count_hist"][dominant_animal]

    dominant_count = max(
        count_hist.keys(),
        key=lambda c: (count_hist[c], c)
    )

    return dominant_animal, int(dominant_count)


def reset_window_stats():
    return {
        "total_frames": 0,
        "none_frames": 0,
        "animal_hit_frames": Counter(),
        "animal_count_hist": defaultdict(Counter),
    }
# =========================
# Main
# =========================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"RKNN model not found: {MODEL_PATH}")

    rknn = RKNNLite()

    ret = rknn.load_rknn(str(MODEL_PATH))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed, ret={ret}")

    try:
        core_mask = RKNNLite.NPU_CORE_0_1_2
    except AttributeError:
        core_mask = RKNNLite.NPU_CORE_0

    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed, ret={ret}")

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed: {CAMERA_DEVICE}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    frame_count_window = 0
    fail_count_total = 0

    last_log_time = time.time()
    window_stats = reset_window_stats()

    try:
        while running:
            ret, frame = cap.read()

            if not ret or frame is None:
                fail_count_total += 1

                if fail_count_total >= 30:
                    raise RuntimeError("Camera read failed too many times")

                continue

            frame_count_window += 1

            input_data, ratio, pad = preprocess(frame)

            outputs = rknn.inference(inputs=[input_data])
            detections = postprocess(outputs, ratio, pad, frame.shape)

            animal, count, filtered_detections = apply_single_species_rule(detections)
            update_window_stats(window_stats, animal, count)

            now = time.time()
            dt = now - last_log_time

            if dt >= LOG_INTERVAL_SEC:
                fps = frame_count_window / dt

                window_animal, window_count = get_window_result(window_stats)

                log_item = {
                    "timestamp": round(now, 3),
                    "animal": window_animal,
                    "count": window_count,
                    "fps": round(fps, 2),
                }

                safe_print(log_item)

                frame_count_window = 0
                window_stats = reset_window_stats()
                last_log_time = now

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        rknn.release()


if __name__ == "__main__":
    main()