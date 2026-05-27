from pathlib import Path
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite


MODEL_PATH = Path("/home/marvsmart/animal_patrol/models/best_fp.rknn")
CAMERA_DEVICE = "/dev/video12"

INPUT_SIZE = 640
CONF_THRES = 0.30
IOU_THRES = 0.45

CLASS_NAMES = {
    0: "peacock",
    1: "tiger",
    2: "elephant",
    3: "wolf",
    4: "monkey",
}

# OpenCV 使用 BGR 颜色，不是 RGB
CLASS_COLORS = {
    0: (255, 0, 255),    # peacock: purple
    1: (0, 165, 255),    # tiger: orange
    2: (255, 180, 0),    # elephant: blue
    3: (0, 0, 255),      # wolf: red
    4: (0, 200, 0),      # monkey: green
}

NUM_CLASSES = len(CLASS_NAMES)


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

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def preprocess(frame):
    img, ratio, pad = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # RKNN 常见输入：NHWC uint8
    input_data = np.expand_dims(img_rgb, axis=0).astype(np.uint8)

    return input_data, ratio, pad


def xywh_to_xyxy(boxes):
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def postprocess(outputs, ratio, pad, original_shape):
    """
    支持常见 YOLOv8 输出：
    [1, 9, 8400] 或 [1, 8400, 9]
    其中 9 = 4 个 bbox + 5 个类别分数。
    """
    if outputs is None or len(outputs) == 0:
        return []

    out = np.array(outputs[0])
    out = np.squeeze(out)

    # 打印一次，方便确认实际输出
    # 常见：out shape = (9, 8400) 或 (8400, 9)
    if out.ndim != 2:
        print("暂不支持的输出维度:", out.shape)
        return []

    expected_dim = 4 + NUM_CLASSES

    if out.shape[0] == expected_dim:
        pred = out.T
    elif out.shape[1] == expected_dim:
        pred = out
    else:
        print("暂不支持的 YOLO 输出 shape:", out.shape)
        print("需要的最后维度应为:", expected_dim)
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

    # OpenCV NMSBoxes 需要 x,y,w,h
    nms_boxes = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        nms_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

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
            "box": [x1, y1, x2, y2],
            "class_id": cls_id,
            "name": name,
            "conf": conf,
        })

    return detections


def draw_detections(frame, detections):
    """
    使用 OpenCV 绘制英文标签和不同颜色的识别框。
    英文标签不会出现中文“??”问题。
    """
    if not detections:
        return frame

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = int(det["class_id"])
        name = det["name"]
        conf = float(det["conf"])

        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        label = f"{name} {conf:.2f}"

        # 画识别框
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # 计算文字大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        text_size, baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        text_w, text_h = text_size

        # 标签背景位置
        label_x1 = x1
        label_y1 = max(0, y1 - text_h - baseline - 8)
        label_x2 = x1 + text_w + 10
        label_y2 = label_y1 + text_h + baseline + 8

        # 画标签背景
        cv2.rectangle(
            frame,
            (label_x1, label_y1),
            (label_x2, label_y2),
            color,
            -1
        )

        # 写英文类别名
        cv2.putText(
            frame,
            label,
            (label_x1 + 5, label_y2 - baseline - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return frame
def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"RKNN 模型不存在: {MODEL_PATH}")

    rknn = RKNNLite()

    print("--> load rknn model")
    ret = rknn.load_rknn(str(MODEL_PATH))
    print("load_rknn ret =", ret)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed, ret={ret}")

    print("--> init runtime")
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    print("init_runtime ret =", ret)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed, ret={ret}")

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"摄像头打开失败: {CAMERA_DEVICE}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("摄像头打开成功")
    print("实际宽度:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("实际高度:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("实际 FPS:", cap.get(cv2.CAP_PROP_FPS))
    print("按 q 退出")

    frame_count = 0
    infer_count = 0
    first_output_printed = False
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("读取摄像头失败")
            break

        frame_count += 1

        input_data, ratio, pad = preprocess(frame)

        t0 = time.time()
        outputs = rknn.inference(inputs=[input_data])
        infer_ms = (time.time() - t0) * 1000.0

        infer_count += 1

        if not first_output_printed:
            print("第一次输出 shape：")
            for i, out in enumerate(outputs):
                arr = np.array(out)
                print(f"output[{i}] shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
            first_output_printed = True

        detections = postprocess(outputs, ratio, pad, frame.shape)

        summary = {}
        for det in detections:
            summary[det["name"]] = summary.get(det["name"], 0) + 1

        if infer_count % 10 == 0:
            elapsed = time.time() - start_time
            print({
                "elapsed_s": round(elapsed, 1),
                "frame_count": frame_count,
                "infer_count": infer_count,
                "infer_ms": round(infer_ms, 2),
                "summary": summary,
            })

        vis = draw_detections(frame.copy(), detections)
        cv2.imshow("RK3588 RKNN YOLO Camera", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    rknn.release()
    cv2.destroyAllWindows()
    print("测试结束")


if __name__ == "__main__":
    main()
