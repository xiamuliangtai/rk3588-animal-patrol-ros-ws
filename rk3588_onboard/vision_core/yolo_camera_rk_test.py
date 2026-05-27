from pathlib import Path
import time

import cv2
from ultralytics import YOLO


MODEL_PATH = Path("/home/marvsmart/animal_patrol/models/best.pt")
CAMERA_DEVICE = "/dev/video12"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
print("模型加载成功")
print("model.names =", model.names)
print("model.task =", model.task)

# RK3588 / Linux 使用 V4L2，不使用 Windows 的 CAP_DSHOW
cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError(f"摄像头打开失败: {CAMERA_DEVICE}")

# 推荐先用 MJPG + 1280x720
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("摄像头打开成功")
print("实际宽度:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("实际高度:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("实际 FPS 设置值:", cap.get(cv2.CAP_PROP_FPS))

# 你的动物模型是 5 类
TARGET_CLASS_IDS = [0, 1, 2, 3, 4]

skip_frame = 1
frame_count = 0
infer_count = 0
fail_count = 0
start_time = time.time()

print("开始识别，按 q 退出。")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret or frame is None:
        fail_count += 1
        print("读取摄像头失败 fail_count =", fail_count)
        if fail_count > 30:
            break
        continue

    frame_count += 1

    # 跳帧：降低 .pt CPU 推理压力
    if frame_count % (skip_frame + 1) != 0:
        cv2.imshow("RK3588 YOLO Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    t0 = time.time()

    results = model(
        frame,
        imgsz=640,
        conf=0.30,
        iou=0.45,
        classes=TARGET_CLASS_IDS,
        device="cpu",
        verbose=False
    )[0]

    infer_ms = (time.time() - t0) * 1000.0
    infer_count += 1

    # 统计检测数量
    summary = {}
    if results.boxes is not None and len(results.boxes) > 0:
        for cls_id in results.boxes.cls.cpu().numpy().astype(int):
            name = model.names.get(int(cls_id), str(cls_id))
            summary[name] = summary.get(name, 0) + 1

    if infer_count % 5 == 0:
        elapsed = time.time() - start_time
        print({
            "elapsed_s": round(elapsed, 1),
            "frame_count": frame_count,
            "infer_count": infer_count,
            "infer_ms": round(infer_ms, 1),
            "summary": summary,
            "fail_count": fail_count,
        })

    annotated_frame = results.plot(labels=True, conf=True)
    cv2.imshow("RK3588 YOLO Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("测试结束")
