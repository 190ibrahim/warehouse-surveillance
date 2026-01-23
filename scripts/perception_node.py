#!/usr/bin/env python3

import os
from typing import Optional

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float32MultiArray, Int32MultiArray

try:
    from ultralytics import YOLO
except ImportError:  # Provide a clear error in ROS logs
    YOLO = None


class PerceptionNode:
    def __init__(self) -> None:
        if YOLO is None:
            rospy.logerr("ultralytics is not installed. Install it to run YOLOv8.")
            rospy.signal_shutdown("Missing ultralytics dependency")
            return

        default_weights = "/home/ibrahim/catkin_ws/src/warehouse_surveillance/yolo/yolov8n.pt"
        self.weights = rospy.get_param("~weights", default_weights)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")
        self.conf = float(rospy.get_param("~conf", 0.4))
        self.iou = float(rospy.get_param("~iou", 0.5))
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.device = rospy.get_param("~device", "")
        self.max_fps = float(rospy.get_param("~max_fps", 10.0))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))
        self.aruco_enabled = bool(rospy.get_param("~aruco_enabled", True))
        self.aruco_dict_name = rospy.get_param("~aruco_dictionary", "DICT_4X4_50")
        self.authorized_ids = rospy.get_param("~authorized_ids", [])
        self.aruco_use_roi = bool(rospy.get_param("~aruco_use_roi", True))
        self.roi_y_min_ratio = float(rospy.get_param("~roi_y_min_ratio", 0.25))
        self.roi_y_max_ratio = float(rospy.get_param("~roi_y_max_ratio", 0.6))
        self.roi_x_margin = float(rospy.get_param("~roi_x_margin", 0.15))
        self.aruco_roi_min_dim = int(rospy.get_param("~aruco_roi_min_dim", 160))
        self.aruco_roi_max_scale = float(rospy.get_param("~aruco_roi_max_scale", 3.0))
        self.aruco_max_tracks = int(rospy.get_param("~aruco_max_tracks", 8))
        self.aruco_min_bbox_px = int(rospy.get_param("~aruco_min_bbox_px", 30))
        self.aruco_roi_fallback_full = bool(rospy.get_param("~aruco_roi_fallback_full", True))
        self.aruco_hold_time = float(rospy.get_param("~aruco_hold_time", 0.4))
        self.aruco_blur_ksize = int(rospy.get_param("~aruco_blur_ksize", 0))
        self.aruco_adaptive_thresh_constant = float(
            rospy.get_param("~aruco_adaptive_thresh_constant", 7.0)
        )
        self.aruco_min_marker_perimeter_rate = float(
            rospy.get_param("~aruco_min_marker_perimeter_rate", 0.02)
        )
        self.aruco_max_marker_perimeter_rate = float(
            rospy.get_param("~aruco_max_marker_perimeter_rate", 4.0)
        )
        self.aruco_polygonal_approx_accuracy_rate = float(
            rospy.get_param("~aruco_polygonal_approx_accuracy_rate", 0.05)
        )
        self.aruco_corner_refine = bool(rospy.get_param("~aruco_corner_refine", True))
        self.aruco_corner_refine_win_size = int(
            rospy.get_param("~aruco_corner_refine_win_size", 5)
        )
        self.aruco_corner_refine_max_iters = int(
            rospy.get_param("~aruco_corner_refine_max_iters", 30)
        )
        self.aruco_corner_refine_min_accuracy = float(
            rospy.get_param("~aruco_corner_refine_min_accuracy", 0.1)
        )

        if not os.path.isfile(self.weights):
            rospy.logerr("YOLO weights not found: %s", self.weights)
            rospy.signal_shutdown("Missing YOLO weights")
            return

        self.bridge = CvBridge()
        self.model = YOLO(self.weights)

        self.aruco_detector = None
        if self.aruco_enabled:
            try:
                aruco = cv2.aruco
                if hasattr(aruco, self.aruco_dict_name):
                    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, self.aruco_dict_name))
                    params = aruco.DetectorParameters()
                    params.adaptiveThreshConstant = self.aruco_adaptive_thresh_constant
                    params.minMarkerPerimeterRate = self.aruco_min_marker_perimeter_rate
                    params.maxMarkerPerimeterRate = self.aruco_max_marker_perimeter_rate
                    params.polygonalApproxAccuracyRate = self.aruco_polygonal_approx_accuracy_rate
                    if self.aruco_corner_refine and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
                        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
                        params.cornerRefinementWinSize = self.aruco_corner_refine_win_size
                        params.cornerRefinementMaxIterations = self.aruco_corner_refine_max_iters
                        params.cornerRefinementMinAccuracy = self.aruco_corner_refine_min_accuracy
                    self.aruco_detector = aruco.ArucoDetector(aruco_dict, params)
                else:
                    rospy.logwarn("Unknown ArUco dictionary: %s", self.aruco_dict_name)
                    self.aruco_enabled = False
            except AttributeError:
                rospy.logwarn("OpenCV ArUco module not available; install opencv-contrib-python.")
                self.aruco_enabled = False

        # Normalize authorized_ids to a list of ints.
        if isinstance(self.authorized_ids, (int, float)):
            self.authorized_ids = [int(self.authorized_ids)]
        elif isinstance(self.authorized_ids, (list, tuple)):
            self.authorized_ids = [int(x) for x in self.authorized_ids]
        else:
            self.authorized_ids = []

        self.processing = False
        self.last_process_time = rospy.Time(0)
        self.last_marker_by_track = {}

        self.image_sub = rospy.Subscriber(
            "image", Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        self.target_pub = rospy.Publisher("target", PointStamped, queue_size=1)
        self.tracks_pub = rospy.Publisher("tracks", Float32MultiArray, queue_size=1)
        self.authorized_pub = rospy.Publisher("target_authorized", Bool, queue_size=1)
        self.person_status_pub = rospy.Publisher("person_status", Int32MultiArray, queue_size=1)
        self.aruco_ids_pub = rospy.Publisher("aruco_ids", Int32MultiArray, queue_size=1)
        self.debug_pub = rospy.Publisher("debug_image", Image, queue_size=1)

        rospy.loginfo("Perception node ready. Weights: %s", self.weights)

    def _detect_marker_in_rect(self, frame, rx1, ry1, rx2, ry2):
        if self.aruco_detector is None:
            return -1
        if rx2 <= rx1 or ry2 <= ry1:
            return -1

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return -1

        scale = 1.0
        max_dim = max(roi.shape[0], roi.shape[1])
        if max_dim > 0 and max_dim < self.aruco_roi_min_dim:
            scale = min(self.aruco_roi_min_dim / float(max_dim), self.aruco_roi_max_scale)
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.aruco_blur_ksize and self.aruco_blur_ksize > 0:
            k = self.aruco_blur_ksize
            if k % 2 == 0:
                k += 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        corners, ids_found, _ = self.aruco_detector.detectMarkers(gray)
        if ids_found is None or len(ids_found) == 0:
            return -1

        best_idx = 0
        best_area = -1.0
        for j, corner in enumerate(corners):
            pts = corner.reshape(-1, 2)
            area = float(cv2.contourArea(pts.astype(np.float32)))
            if area > best_area:
                best_area = area
                best_idx = j

        return int(ids_found[best_idx][0])

    def image_callback(self, msg: Image) -> None:
        if self.processing:
            return

        now = rospy.Time.now()
        if self.max_fps > 0:
            min_dt = 1.0 / self.max_fps
            if (now - self.last_process_time).to_sec() < min_dt:
                return

        self.processing = True
        self.last_process_time = now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                classes=[0],  # person class
                verbose=False,
            )

            if not results:
                return

            result = results[0]
            tracks_msg = Float32MultiArray()
            target_msg: Optional[PointStamped] = None
            best_score: Optional[float] = None
            target_authorized = False
            person_status_msg = Int32MultiArray()
            aruco_ids = []
            track_info = []
            now = rospy.Time.now()

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = (
                    result.boxes.conf.cpu().numpy()
                    if result.boxes.conf is not None
                    else np.zeros(len(xyxy), dtype=np.float32)
                )
                ids = result.boxes.id
                if ids is not None:
                    ids = ids.cpu().numpy().astype(np.int32)
                else:
                    ids = np.full((len(xyxy),), -1, dtype=np.int32)

                img_h, img_w = frame.shape[:2]

                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = box.tolist()
                    conf = float(confs[i]) if i < len(confs) else 0.0
                    track_id = int(ids[i]) if i < len(ids) else -1

                    # tracks format: [id, x1, y1, x2, y2, conf] repeated
                    tracks_msg.data.extend([float(track_id), x1, y1, x2, y2, conf])

                    # Pick the largest person as the tracking target.
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    area = w * h
                    score = area
                    if best_score is None or score > best_score:
                        best_score = score
                        cx = 0.5 * (x1 + x2)
                        cy = 0.5 * (y1 + y2)
                        target_msg = PointStamped()
                        target_msg.header = msg.header
                        target_msg.header.frame_id = "image"
                        # Normalize pixel center to (-1..1) for controller use.
                        target_msg.point.x = (cx - img_w * 0.5) / (img_w * 0.5)
                        target_msg.point.y = (cy - img_h * 0.5) / (img_h * 0.5)
                        target_msg.point.z = conf

                    track_info.append(
                        {
                            "id": track_id,
                            "bbox": (x1, y1, x2, y2),
                            "w": w,
                            "h": h,
                            "area": area,
                            "marker_id": -1,
                            "authorized": False,
                        }
                    )

                if self.aruco_enabled and self.aruco_detector is not None and track_info:
                    # Limit ArUco checks to the largest tracks if configured.
                    if self.aruco_max_tracks > 0:
                        idxs = sorted(
                            range(len(track_info)),
                            key=lambda i: track_info[i]["area"],
                            reverse=True,
                        )[: self.aruco_max_tracks]
                    else:
                        idxs = list(range(len(track_info)))

                    for i in idxs:
                        info = track_info[i]
                        x1, y1, x2, y2 = info["bbox"]
                        w = info["w"]
                        h = info["h"]
                        if w < self.aruco_min_bbox_px or h < self.aruco_min_bbox_px:
                            continue

                        if self.aruco_use_roi:
                            roi_x1 = x1 + self.roi_x_margin * w
                            roi_x2 = x2 - self.roi_x_margin * w
                            roi_y1 = y1 + self.roi_y_min_ratio * h
                            roi_y2 = y1 + self.roi_y_max_ratio * h
                        else:
                            roi_x1, roi_y1, roi_x2, roi_y2 = x1, y1, x2, y2

                        rx1 = max(0, int(roi_x1))
                        ry1 = max(0, int(roi_y1))
                        rx2 = min(img_w, int(roi_x2))
                        ry2 = min(img_h, int(roi_y2))
                        if rx2 <= rx1 or ry2 <= ry1:
                            continue

                        marker_id = self._detect_marker_in_rect(frame, rx1, ry1, rx2, ry2)
                        if marker_id == -1 and self.aruco_use_roi and self.aruco_roi_fallback_full:
                            rx1 = max(0, int(x1))
                            ry1 = max(0, int(y1))
                            rx2 = min(img_w, int(x2))
                            ry2 = min(img_h, int(y2))
                            marker_id = self._detect_marker_in_rect(frame, rx1, ry1, rx2, ry2)

                        if marker_id == -1 and self.aruco_hold_time > 0.0:
                            last = self.last_marker_by_track.get(info["id"])
                            if last is not None:
                                last_id, last_time = last
                                if (now - last_time).to_sec() <= self.aruco_hold_time:
                                    marker_id = last_id

                        info["marker_id"] = marker_id
                        if marker_id != -1:
                            if not self.authorized_ids:
                                info["authorized"] = True
                            else:
                                info["authorized"] = marker_id in self.authorized_ids
                            aruco_ids.append(marker_id)
                            self.last_marker_by_track[info["id"]] = (marker_id, now)

            for info in track_info:
                person_status_msg.data.extend(
                    [int(info["id"]), 1 if info["authorized"] else 0, int(info["marker_id"])]
                )

            if self.aruco_hold_time > 0.0 and self.last_marker_by_track:
                cutoff = now - rospy.Duration(self.aruco_hold_time)
                self.last_marker_by_track = {
                    key: value
                    for key, value in self.last_marker_by_track.items()
                    if value[1] >= cutoff
                }

            if target_msg is not None and track_info:
                # Set target authorization based on the largest tracked person.
                best_idx = max(range(len(track_info)), key=lambda i: track_info[i]["area"])
                target_authorized = track_info[best_idx]["authorized"]

            if tracks_msg.data:
                self.tracks_pub.publish(tracks_msg)
            if target_msg is not None:
                self.target_pub.publish(target_msg)
            if person_status_msg.data:
                self.person_status_pub.publish(person_status_msg)
            self.authorized_pub.publish(Bool(data=target_authorized))
            if aruco_ids:
                self.aruco_ids_pub.publish(Int32MultiArray(data=aruco_ids))

            if self.publish_debug and self.debug_pub.get_num_connections() > 0:
                annotated = result.plot()
                for info in track_info:
                    x1, y1, x2, y2 = info["bbox"]
                    label = "authorized" if info["authorized"] else "intruder"
                    color = (0, 200, 0) if info["authorized"] else (0, 0, 255)
                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )
                debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)


        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, "CvBridge error: %s", exc)
        except Exception as exc:
            rospy.logerr_throttle(5.0, "Perception error: %s", exc)
        finally:
            self.processing = False


def main() -> None:
    rospy.init_node("perception_node")
    PerceptionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
