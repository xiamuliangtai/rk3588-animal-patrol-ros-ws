#!/usr/bin/env python3
import json
import rospy
from std_msgs.msg import String


def callback(msg):
    try:
        data = json.loads(msg.data)
    except json.JSONDecodeError:
        rospy.logwarn("Invalid JSON: %s", msg.data)
        return

    timestamp = data.get("timestamp")
    animal = data.get("animal")
    count = data.get("count")
    fps = data.get("fps")

    rospy.loginfo(
        "stable result | timestamp=%s animal=%s count=%s fps=%s",
        timestamp,
        animal,
        count,
        fps
    )


def main():
    rospy.init_node("vision_result_listener", anonymous=False)
    rospy.Subscriber("/vision/result_stable", String, callback, queue_size=10)
    rospy.loginfo("vision_result_listener started")
    rospy.spin()


if __name__ == "__main__":
    main()
