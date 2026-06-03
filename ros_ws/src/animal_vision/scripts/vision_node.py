#!/usr/bin/env python3
# ---- logging compatibility patch for rospy ----
# Some RK3588 Python/ROS environments may fail to parse logging levels
# such as "DEBUG", "INFO", or even numeric strings like "20" during
# rospy.init_node(). Patch logging before importing rospy.
import logging

logging._nameToLevel.update({
    "CRITICAL": 50,
    "FATAL": 50,
    "ERROR": 40,
    "WARN": 30,
    "WARNING": 30,
    "INFO": 20,
    "DEBUG": 10,
    "NOTSET": 0,
})

logging._levelToName.update({
    50: "CRITICAL",
    40: "ERROR",
    30: "WARNING",
    20: "INFO",
    10: "DEBUG",
    0: "NOTSET",
})

_original_check_level = logging._checkLevel

def _safe_check_level(level):
    if isinstance(level, str):
        s = level.strip().upper()
        if s.isdigit():
            return int(s)
        if s in logging._nameToLevel:
            return logging._nameToLevel[s]
    return _original_check_level(level)

logging._checkLevel = _safe_check_level
# ---- end logging compatibility patch for rospy ----

import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import rospy
import rosgraph.roslogging

# ---- disable rospy fileConfig logging on RK3588 ----
# Some RK3588 virtualenv + ROS Noetic environments fail at
# logging.config.fileConfig() during rospy.init_node().
# This bypasses only ROS Python log-file configuration.
# It does not affect ROS topic publishing.
def _rk3588_safe_configure_logging(*args, **kwargs):
    return None

rosgraph.roslogging.configure_logging = _rk3588_safe_configure_logging
# ---- end disable rospy fileConfig logging on RK3588 ----

from std_msgs.msg import String
from rknnlite.api import RKNNLite


VISION_CORE_PATH = "/home/marvsmart/animal_patrol/rk3588_onboard/vision_core"

if VISION_CORE_PATH not in sys.path:
    sys.path.insert(0, VISION_CORE_PATH)

import rknn_camera_yolo_test as core


class StableResultFilter:
    """
    Multi-second stable result filter.

    Input:
        raw result per logging window.

    Output:
        stable animal result.

    Rule:
        1. Ignore none.
        2. In recent N windows, dominant animal must appear enough times.
        3. Dominant animal ratio must reach threshold.
        4. Count uses the most frequent count value.
    """

    def __init__(self, window_size=5, min_ratio=0.60, min_hit_windows=3):
        self.history = deque(maxlen=int(window_size))
        self.window_size = int(window_size)
        self.min_ratio = float(min_ratio)
        self.min_hit_windows = int(min_hit_windows)

    def update(self, animal, count):
        self.history.append({
            "animal": str(animal),
            "count": int(count),
        })
        return self.get_result()

    def get_result(self):
        if not self.history:
            return "none", 0, 0.0

        animal_counter = Counter()
        count_hist = defaultdict(Counter)

        for item in self.history:
            animal = item["animal"]
            count = int(item["count"])

            if animal == "none" or count <= 0:
                continue

            animal_counter[animal] += 1
            count_hist[animal][count] += 1

        if not animal_counter:
            return "none", 0, 0.0

        dominant_animal, hit_windows = animal_counter.most_common(1)[0]
        ratio = hit_windows / len(self.history)

        if hit_windows < self.min_hit_windows:
            return "none", 0, ratio

        if ratio < self.min_ratio:
            return "none", 0, ratio

        dominant_count = max(
            count_hist[dominant_animal].keys(),
            key=lambda c: (count_hist[dominant_animal][c], c)
        )

        return dominant_animal, int(dominant_count), ratio


class VisionNode:
    def __init__(self):
        rospy.init_node("vision_node", anonymous=False)

        self.model_path = Path(rospy.get_param(
            "~model_path",
            str(core.MODEL_PATH)
        ))

        self.camera_device = rospy.get_param(
            "~camera_device",
            core.CAMERA_DEVICE
        )

        core.INPUT_SIZE = int(rospy.get_param(
            "~input_size",
            core.INPUT_SIZE
        ))

        core.CONF_THRES = float(rospy.get_param(
            "~conf_thres",
            core.CONF_THRES
        ))

        core.IOU_THRES = float(rospy.get_param(
            "~iou_thres",
            core.IOU_THRES
        ))

        self.camera_width = int(rospy.get_param(
            "~camera_width",
            core.CAMERA_WIDTH
        ))

        self.camera_height = int(rospy.get_param(
            "~camera_height",
            core.CAMERA_HEIGHT
        ))

        self.camera_fps = int(rospy.get_param(
            "~camera_fps",
            core.CAMERA_FPS
        ))

        self.publish_interval_sec = float(rospy.get_param(
            "~publish_interval_sec",
            1.0
        ))

        stable_window_size = int(rospy.get_param(
            "~stable_window_size",
            5
        ))

        stable_min_ratio = float(rospy.get_param(
            "~stable_min_ratio",
            0.60
        ))

        stable_min_hit_windows = int(rospy.get_param(
            "~stable_min_hit_windows",
            3
        ))

        self.raw_pub = rospy.Publisher(
            "/vision/result_raw",
            String,
            queue_size=10
        )

        self.stable_pub = rospy.Publisher(
            "/vision/result_stable",
            String,
            queue_size=10
        )

        self.stable_filter = StableResultFilter(
            window_size=stable_window_size,
            min_ratio=stable_min_ratio,
            min_hit_windows=stable_min_hit_windows,
        )

        self.rknn = None
        self.cap = None

    def setup_rknn(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN model not found: {self.model_path}")

        self.rknn = RKNNLite()

        rospy.loginfo(f"Loading RKNN model: {self.model_path}")
        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed, ret={ret}")

        try:
            core_mask = RKNNLite.NPU_CORE_0_1_2
            rospy.loginfo("Using NPU core mask: NPU_CORE_0_1_2")
        except AttributeError:
            core_mask = RKNNLite.NPU_CORE_0
            rospy.logwarn("NPU_CORE_0_1_2 unavailable, fallback to NPU_CORE_0")

        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed, ret={ret}")

        rospy.loginfo("RKNN runtime initialized")

    def setup_camera(self):
        self.cap = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera open failed: {self.camera_device}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

        rospy.loginfo(f"Camera opened: {self.camera_device}")
        rospy.loginfo(f"Actual width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        rospy.loginfo(f"Actual height: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        rospy.loginfo(f"Actual FPS setting: {self.cap.get(cv2.CAP_PROP_FPS)}")

    @staticmethod
    def publish_json(pub, data):
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        pub.publish(msg)

    def run(self):
        self.setup_rknn()
        self.setup_camera()

        frame_count_window = 0
        fail_count_total = 0

        last_publish_time = time.time()
        window_stats = core.reset_window_stats()

        rospy.loginfo("vision_node started")

        while not rospy.is_shutdown():
            ret, frame = self.cap.read()

            if not ret or frame is None:
                fail_count_total += 1
                rospy.logwarn(f"Camera read failed, fail_count_total={fail_count_total}")

                if fail_count_total >= 30:
                    raise RuntimeError("Camera read failed too many times")

                continue

            frame_count_window += 1

            input_data, ratio, pad = core.preprocess(frame)

            outputs = self.rknn.inference(inputs=[input_data])
            detections = core.postprocess(outputs, ratio, pad, frame.shape)

            raw_animal, raw_count, _ = core.apply_single_species_rule(detections)
            core.update_window_stats(window_stats, raw_animal, raw_count)

            now = time.time()
            dt = now - last_publish_time

            if dt >= self.publish_interval_sec:
                fps = frame_count_window / dt

                window_animal, window_count = core.get_window_result(window_stats)

                stable_animal, stable_count, stable_ratio = self.stable_filter.update(
                    window_animal,
                    window_count
                )

                raw_result = {
                    "timestamp": round(now, 3),
                    "animal": window_animal,
                    "count": int(window_count),
                    "fps": round(fps, 2),
                }

                stable_result = {
                    "timestamp": round(now, 3),
                    "animal": stable_animal,
                    "count": int(stable_count),
                    "fps": round(fps, 2),
                }

                self.publish_json(self.raw_pub, raw_result)
                self.publish_json(self.stable_pub, stable_result)

                rospy.loginfo(json.dumps(stable_result, ensure_ascii=False))

                frame_count_window = 0
                window_stats = core.reset_window_stats()
                last_publish_time = now

        self.cleanup()

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


if __name__ == "__main__":
    node = VisionNode()

    try:
        node.run()
    finally:
        node.cleanup()
