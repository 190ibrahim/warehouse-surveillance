import cv2
import time
import numpy as np
from ultralytics import YOLO

def _setup_aruco(dict_name: str):
    try:
        aruco = cv2.aruco
    except AttributeError:
        print("OpenCV ArUco module not available; install opencv-contrib-python.")
        return None

    if not hasattr(aruco, dict_name):
        print(f"Unknown ArUco dictionary: {dict_name}")
        return None

    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    params = aruco.DetectorParameters()
    return aruco.ArucoDetector(aruco_dict, params)

def _detect_marker_in_rect(detector, frame, rx1, ry1, rx2, ry2):
    if detector is None or rx2 <= rx1 or ry2 <= ry1:
        return -1

    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return -1

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    corners, ids_found, _ = detector.detectMarkers(gray)
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

def main():
    # setup camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("no camera")
        return

    # Optional: ask camera for a higher capture resolution (if supported)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # load YOLO model
    model = YOLO('yolo/yolov8s.pt')
    prev_time = time.time()

    # ArUco auth setup
    aruco_dict_name = "DICT_ARUCO_ORIGINAL"
    authorized_ids = []
    aruco_detector = _setup_aruco(aruco_dict_name)
    aruco_use_roi = True
    roi_y_min_ratio = 0.25
    roi_y_max_ratio = 0.60
    roi_x_margin = 0.15

    # ---- Recording + "make it big" settings ----
    scale = 1.5  # 1.0 = same size, 1.5 = bigger, 2.0 = much bigger
    out_path = "tracking_output.mp4"
    out_fps = 30.0  # will be set more accurately after first frame if possible

    writer = None  # init after we know frame size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Window settings (resizable / fullscreen)
    win_name = "Tracking"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # Option A: make window fullscreen (comment out if you don't want)
    # cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("press q to quit")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # track people in frame
        results = model.track(
            frame,
            classes=[0],
            conf=0.5,
            tracker="bytetrack.yaml",
            verbose=False
        )

        tracks_count = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for bbox, track_id, conf in zip(boxes.xyxy, boxes.id, boxes.conf):
                    x1, y1, x2, y2 = map(int, bbox.cpu().numpy())
                    track_id = int(track_id.cpu().numpy())
                    conf = float(conf.cpu().numpy())
                    tracks_count += 1

                    marker_id = -1
                    if aruco_detector is not None:
                        img_h, img_w = frame.shape[:2]
                        if aruco_use_roi:
                            w = max(0, x2 - x1)
                            h = max(0, y2 - y1)
                            roi_x1 = x1 + roi_x_margin * w
                            roi_x2 = x2 - roi_x_margin * w
                            roi_y1 = y1 + roi_y_min_ratio * h
                            roi_y2 = y1 + roi_y_max_ratio * h
                        else:
                            roi_x1, roi_y1, roi_x2, roi_y2 = x1, y1, x2, y2

                        rx1 = max(0, int(roi_x1))
                        ry1 = max(0, int(roi_y1))
                        rx2 = min(img_w, int(roi_x2))
                        ry2 = min(img_h, int(roi_y2))
                        marker_id = _detect_marker_in_rect(aruco_detector, frame, rx1, ry1, rx2, ry2)

                    if marker_id != -1:
                        authorized = True if not authorized_ids else (marker_id in authorized_ids)
                    else:
                        authorized = False

                    color = (0, 200, 0) if authorized else (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    status = "AUTH" if authorized else "UNAUTH"
                    marker_text = f" | M:{marker_id}" if marker_id != -1 else ""
                    cv2.putText(
                        frame,
                        f'ID: {track_id} ({conf:.2f}) {status}{marker_text}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

        # FPS calc
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()

        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Tracked: {tracks_count} persons', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ---- Resize to make it big (this affects BOTH display and recording) ----
        if scale != 1.0:
            frame_out = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            frame_out = frame

        # ---- Init writer once we know output frame size ----
        if writer is None:
            h, w = frame_out.shape[:2]
            # try to use camera FPS if available
            cam_fps = cam.get(cv2.CAP_PROP_FPS)
            if cam_fps and cam_fps > 1:
                out_fps = float(cam_fps)
            writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))
            if not writer.isOpened():
                print("Could not open VideoWriter. Try a different codec/path.")
                writer = None

        # ---- Write frame ----
        if writer is not None:
            writer.write(frame_out)

        # Show
        cv2.imshow(win_name, frame_out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cam.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Saved video to: {out_path}")

if __name__ == "__main__":
    main()
