#!/usr/bin/env python3

import logging

# Keep rospy usable in the RK3588 virtualenv where logging names may be absent.
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
        value = level.strip().upper()
        if value.isdigit():
            return int(value)
        if value in logging._nameToLevel:
            return logging._nameToLevel[value]
    return _original_check_level(level)


logging._checkLevel = _safe_check_level

import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import rosgraph.roslogging
import rospy
from animal_vision.msg import DetectionEvent
from rknnlite.api import RKNNLite
from std_msgs.msg import String


def _rk3588_safe_configure_logging(*args, **kwargs):
    return None


rosgraph.roslogging.configure_logging = _rk3588_safe_configure_logging

VISION_CORE_PATH = "/home/marvsmart/animal_patrol/rk3588_onboard/vision_core"

if VISION_CORE_PATH not in sys.path:
    sys.path.insert(0, VISION_CORE_PATH)

import rknn_camera_yolo_test as core


class StableResultFilter:
    """Filter window-level detections and preserve their earliest frame time."""

    def __init__(self, window_size=5, min_ratio=0.60, min_hit_windows=3):
        self.history = deque(maxlen=int(window_size))
        self.min_ratio = float(min_ratio)
        self.min_hit_windows = int(min_hit_windows)

    def update(self, animal, count, first_seen_ns):
        self.history.append({
            "animal": str(animal),
            "count": int(count),
            "first_seen_ns": int(first_seen_ns),
        })
        return self.get_result()

    def get_result(self):
        if not self.history:
            return "none", 0, 0.0, 0

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
            return "none", 0, 0.0, 0

        dominant_animal, hit_windows = animal_counter.most_common(1)[0]
        ratio = hit_windows / len(self.history)
        if hit_windows < self.min_hit_windows or ratio < self.min_ratio:
            return "none", 0, ratio, 0

        dominant_count = max(
            count_hist[dominant_animal].keys(),
            key=lambda value: (count_hist[dominant_animal][value], value),
        )

        first_seen_values = [
            int(item["first_seen_ns"])
            for item in self.history
            if item["animal"] == dominant_animal
            and int(item["count"]) > 0
            and int(item["first_seen_ns"]) > 0
        ]
        first_seen_ns = min(first_seen_values) if first_seen_values else 0
        return dominant_animal, int(dominant_count), ratio, first_seen_ns


class VisionNode:
    def __init__(self):
        rospy.init_node("vision_node", anonymous=False)

        self.model_path = Path(
            rospy.get_param("~model_path", str(core.MODEL_PATH))
        )
        self.camera_device = rospy.get_param(
            "~camera_device", core.CAMERA_DEVICE
        )

        core.INPUT_SIZE = int(rospy.get_param("~input_size", core.INPUT_SIZE))
        core.CONF_THRES = float(
            rospy.get_param("~conf_thres", core.CONF_THRES)
        )
        core.IOU_THRES = float(rospy.get_param("~iou_thres", core.IOU_THRES))

        self.camera_width = int(
            rospy.get_param("~camera_width", core.CAMERA_WIDTH)
        )
        self.camera_height = int(
            rospy.get_param("~camera_height", core.CAMERA_HEIGHT)
        )
        self.camera_fps = int(
            rospy.get_param("~camera_fps", core.CAMERA_FPS)
        )
        self.publish_interval_sec = float(
            rospy.get_param("~publish_interval_sec", 0.2)
        )

        stable_window_size = int(
            rospy.get_param("~stable_window_size", 5)
        )
        stable_min_ratio = float(
            rospy.get_param("~stable_min_ratio", 0.60)
        )
        stable_min_hit_windows = int(
            rospy.get_param("~stable_min_hit_windows", 3)
        )

        self.raw_pub = rospy.Publisher(
            "/vision/result_raw", String, queue_size=10
        )
        self.stable_pub = rospy.Publisher(
            "/vision/result_stable", String, queue_size=10
        )
        self.event_pub = rospy.Publisher(
            "/vision/detection_event", DetectionEvent, queue_size=10
        )

        self.stable_filter = StableResultFilter(
            window_size=stable_window_size,
            min_ratio=stable_min_ratio,
            min_hit_windows=stable_min_hit_windows,
        )

        self.rknn = None
        self.cap = None
        self.last_stable_found = False
        self.event_id = 0
        self.latched_first_seen_ns = 0

    def setup_rknn(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                "RKNN model not found: {}".format(self.model_path)
            )

        self.rknn = RKNNLite()
        rospy.loginfo("Loading RKNN model: %s", self.model_path)

        result = self.rknn.load_rknn(str(self.model_path))
        if result != 0:
            raise RuntimeError("load_rknn failed, ret={}".format(result))

        try:
            core_mask = RKNNLite.NPU_CORE_0_1_2
            rospy.loginfo("Using NPU core mask: NPU_CORE_0_1_2")
        except AttributeError:
            core_mask = RKNNLite.NPU_CORE_0
            rospy.logwarn("NPU_CORE_0_1_2 unavailable; using NPU_CORE_0")

        result = self.rknn.init_runtime(core_mask=core_mask)
        if result != 0:
            raise RuntimeError("init_runtime failed, ret={}".format(result))

        rospy.loginfo("RKNN runtime initialized")

    def setup_camera(self):
        self.cap = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Camera open failed: {}".format(self.camera_device)
            )

        self.cap.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
        )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

        rospy.loginfo("Camera opened: %s", self.camera_device)
        rospy.loginfo(
            "Actual camera mode: %.0fx%.0f @ %.1f",
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            self.cap.get(cv2.CAP_PROP_FPS),
        )

    @staticmethod
    def publish_json(publisher, data):
        message = String()
        message.data = json.dumps(data, ensure_ascii=False)
        publisher.publish(message)

    @staticmethod
    def time_from_ns(timestamp_ns):
        if timestamp_ns <= 0:
            return rospy.Time(0, 0)
        seconds = timestamp_ns // 1000000000
        nanoseconds = timestamp_ns % 1000000000
        return rospy.Time(seconds, nanoseconds)

    def publish_detection_event(
        self, publish_stamp, target_found, first_seen_ns
    ):
        message = DetectionEvent()
        message.header.stamp = publish_stamp
        message.header.frame_id = "camera"
        message.target_found = bool(target_found)
        message.first_seen = self.time_from_ns(first_seen_ns)
        message.event_id = int(self.event_id)
        self.event_pub.publish(message)

    def update_event_latch(self, stable_found, stable_first_seen_ns, now_ns):
        if stable_found and not self.last_stable_found:
            self.event_id = (self.event_id + 1) & 0xFFFFFFFF
            if self.event_id == 0:
                self.event_id = 1
            self.latched_first_seen_ns = (
                int(stable_first_seen_ns)
                if int(stable_first_seen_ns) > 0
                else int(now_ns)
            )
        elif not stable_found:
            self.latched_first_seen_ns = 0

        self.last_stable_found = bool(stable_found)

    def run(self):
        self.setup_rknn()
        self.setup_camera()

        frame_count_window = 0
        consecutive_camera_failures = 0
        window_stats = core.reset_window_stats()
        window_first_seen_by_animal = {}
        last_publish_monotonic = time.monotonic()

        rospy.loginfo("vision_node started")

        while not rospy.is_shutdown():
            success, frame = self.cap.read()
            frame_stamp = rospy.Time.now()

            if not success or frame is None:
                consecutive_camera_failures += 1
                rospy.logwarn_throttle(
                    1.0,
                    "Camera read failed; consecutive_failures={}".format(
                        consecutive_camera_failures
                    ),
                )
                if consecutive_camera_failures >= 30:
                    raise RuntimeError("Camera read failed too many times")
                rospy.sleep(0.02)
                continue

            consecutive_camera_failures = 0
            frame_count_window += 1

            input_data, ratio, pad = core.preprocess(frame)
            outputs = self.rknn.inference(inputs=[input_data])
            detections = core.postprocess(outputs, ratio, pad, frame.shape)
            raw_animal, raw_count, _ = core.apply_single_species_rule(
                detections
            )
            core.update_window_stats(window_stats, raw_animal, raw_count)

            if raw_animal != "none" and int(raw_count) > 0:
                window_first_seen_by_animal.setdefault(
                    str(raw_animal), int(frame_stamp.to_nsec())
                )

            now_monotonic = time.monotonic()
            interval = now_monotonic - last_publish_monotonic
            if interval < self.publish_interval_sec:
                continue

            publish_stamp = rospy.Time.now()
            publish_stamp_ns = int(publish_stamp.to_nsec())
            fps = frame_count_window / max(interval, 1.0e-6)

            window_animal, window_count = core.get_window_result(window_stats)
            window_first_seen_ns = int(
                window_first_seen_by_animal.get(str(window_animal), 0)
            )

            (
                stable_animal,
                stable_count,
                stable_ratio,
                stable_first_seen_ns,
            ) = self.stable_filter.update(
                window_animal,
                window_count,
                window_first_seen_ns,
            )

            stable_found = (
                stable_animal != "none" and int(stable_count) > 0
            )
            self.update_event_latch(
                stable_found,
                stable_first_seen_ns,
                publish_stamp_ns,
            )

            raw_result = {
                "timestamp_ns": publish_stamp_ns,
                "target_found": int(
                    window_animal != "none" and int(window_count) > 0
                ),
                "first_seen_ns": window_first_seen_ns,
                "animal": window_animal,
                "count": int(window_count),
                "fps": round(fps, 2),
            }
            stable_result = {
                "timestamp_ns": publish_stamp_ns,
                "target_found": int(stable_found),
                "first_seen_ns": int(self.latched_first_seen_ns),
                "event_id": int(self.event_id),
                "animal": stable_animal,
                "count": int(stable_count),
                "stable_ratio": round(float(stable_ratio), 3),
                "fps": round(fps, 2),
            }

            self.publish_json(self.raw_pub, raw_result)
            self.publish_json(self.stable_pub, stable_result)
            self.publish_detection_event(
                publish_stamp,
                stable_found,
                self.latched_first_seen_ns,
            )
            rospy.loginfo(json.dumps(stable_result, ensure_ascii=False))

            frame_count_window = 0
            window_stats = core.reset_window_stats()
            window_first_seen_by_animal = {}
            last_publish_monotonic = now_monotonic

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
