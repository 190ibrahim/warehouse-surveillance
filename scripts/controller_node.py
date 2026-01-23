#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import PointStamped, Twist
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray, Int32MultiArray


class ControllerNode:
    def __init__(self) -> None:
        self.angular_kp = float(rospy.get_param("~angular_kp", 1.2))
        self.max_angular = float(rospy.get_param("~max_angular", 0.3))
        self.linear_speed = float(rospy.get_param("~linear_speed", 0.1))
        self.max_linear = float(rospy.get_param("~max_linear", 0.3))
        self.deadband = float(rospy.get_param("~deadband", 0.05))
        self.smooth_alpha = float(rospy.get_param("~smooth_alpha", 0.6))
        self.control_rate = float(rospy.get_param("~control_rate", 10.0))

        self.prefer_intruder = bool(rospy.get_param("~prefer_intruder", True))
        # If true, only move forward when an intruder is present.
        self.move_on_intruder_only = bool(rospy.get_param("~move_on_intruder_only", True))
        self.tracks_timeout = float(rospy.get_param("~tracks_timeout", 0.5))
        self.status_timeout = float(rospy.get_param("~status_timeout", 0.5))
        self.target_timeout = float(rospy.get_param("~target_timeout", 0.5))
        self.stop_on_no_target = bool(rospy.get_param("~stop_on_no_target", True))
        self.approach_area_ratio = float(rospy.get_param("~approach_area_ratio", 0.06))

        self.image_w: Optional[int] = None
        self.image_h: Optional[int] = None

        self.last_target: Optional[PointStamped] = None
        self.last_target_stamp = rospy.Time(0)

        self.last_tracks: List[Tuple[int, float, float, float, float, float]] = []
        self.last_tracks_stamp = rospy.Time(0)

        self.last_status: Dict[int, Tuple[int, int]] = {}
        self.last_status_stamp = rospy.Time(0)

        self.smoothed_x = 0.0

        self.target_sub = rospy.Subscriber("target", PointStamped, self.target_callback, queue_size=1)
        self.tracks_sub = rospy.Subscriber("tracks", Float32MultiArray, self.tracks_callback, queue_size=1)
        self.status_sub = rospy.Subscriber(
            "person_status", Int32MultiArray, self.status_callback, queue_size=1
        )
        self.camera_info_sub = rospy.Subscriber(
            "camera_info", CameraInfo, self.camera_info_callback, queue_size=1
        )

        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)

        rospy.loginfo("Controller node ready.")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.image_w = msg.width
        self.image_h = msg.height

    def target_callback(self, msg: PointStamped) -> None:
        self.last_target = msg
        self.last_target_stamp = rospy.Time.now()

    def tracks_callback(self, msg: Float32MultiArray) -> None:
        data = msg.data
        tracks = []
        for i in range(0, len(data), 6):
            if i + 5 >= len(data):
                break
            track_id = int(data[i])
            x1, y1, x2, y2, conf = data[i + 1 : i + 6]
            tracks.append((track_id, x1, y1, x2, y2, conf))
        self.last_tracks = tracks
        self.last_tracks_stamp = rospy.Time.now()

    def status_callback(self, msg: Int32MultiArray) -> None:
        data = msg.data
        status = {}
        for i in range(0, len(data), 3):
            if i + 2 >= len(data):
                break
            track_id = int(data[i])
            authorized = int(data[i + 1])
            marker_id = int(data[i + 2])
            status[track_id] = (authorized, marker_id)
        self.last_status = status
        self.last_status_stamp = rospy.Time.now()

    def _select_intruder_target(self, now: rospy.Time) -> Optional[Tuple[float, float, float]]:
        if self.image_w is None or self.image_h is None:
            return None
        if (now - self.last_tracks_stamp).to_sec() > self.tracks_timeout:
            return None
        if (now - self.last_status_stamp).to_sec() > self.status_timeout:
            return None

        intruders = []
        for track_id, x1, y1, x2, y2, _ in self.last_tracks:
            status = self.last_status.get(track_id)
            if status is None:
                continue
            authorized, _ = status
            if authorized == 0:
                area = max(0.0, (x2 - x1) * (y2 - y1))
                intruders.append((area, x1, y1, x2, y2))

        if not intruders:
            return None

        # Pick the largest intruder (closest in view).
        _, x1, y1, x2, y2 = max(intruders, key=lambda x: x[0])
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        nx = (cx - self.image_w * 0.5) / (self.image_w * 0.5)
        ny = (cy - self.image_h * 0.5) / (self.image_h * 0.5)
        # Area ratio is a simple distance proxy.
        area = max(0.0, (x2 - x1) * (y2 - y1))
        img_area = float(self.image_w * self.image_h)
        area_ratio = area / img_area if img_area > 0.0 else 0.0
        return nx, ny, area_ratio

    def _select_fallback_target(self, now: rospy.Time) -> Optional[Tuple[float, float, Optional[float]]]:
        if self.last_target is None:
            return None
        if (now - self.last_target_stamp).to_sec() > self.target_timeout:
            return None
        return self.last_target.point.x, self.last_target.point.y, None

    def _compute_cmd(
        self, target_x: float, area_ratio: Optional[float], allow_linear: bool
    ) -> Twist:
        if abs(target_x) < self.deadband:
            target_x = 0.0

        # Smooth the horizontal error to reduce jitter.
        self.smoothed_x = self.smooth_alpha * target_x + (1.0 - self.smooth_alpha) * self.smoothed_x

        angular = -self.angular_kp * self.smoothed_x
        angular = max(-self.max_angular, min(self.max_angular, angular))

        linear = 0.0
        if allow_linear and area_ratio is not None and self.linear_speed > 0.0:
            # Slow down as the target fills more of the image.
            if self.approach_area_ratio > 0.0 and area_ratio < self.approach_area_ratio:
                scale = 1.0 - (area_ratio / self.approach_area_ratio)
                scale = max(0.0, min(1.0, scale))
            else:
                scale = 0.0
            linear = self.linear_speed * scale * max(0.0, 1.0 - abs(self.smoothed_x))
            linear = max(-self.max_linear, min(self.max_linear, linear))

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        return cmd

    def control_loop(self, _event) -> None:
        now = rospy.Time.now()
        target = None
        allow_linear = False

        if self.prefer_intruder:
            target = self._select_intruder_target(now)
            if target is not None:
                allow_linear = True

        if target is None:
            target = self._select_fallback_target(now)
            if target is not None and not self.move_on_intruder_only:
                allow_linear = True

        if target is None:
            if self.stop_on_no_target:
                self.cmd_pub.publish(Twist())
            return

        cmd = self._compute_cmd(target[0], target[2], allow_linear)
        self.cmd_pub.publish(cmd)


def main() -> None:
    rospy.init_node("controller_node")
    ControllerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
