# Warehouse Surveillance Simulation (ROS Noetic)

Gazebo + RViz simulation with TurtleBot2, Pedsim pedestrians, and the
`warehouse_surveillance` perception/control nodes (YOLOv8 + ArUco).

## What this repo does
- Spawns a simple warehouse world with straight walls.
- Spawns pedestrians from Pedsim.
- Runs a perception node that detects people, tracks them, and checks ArUco IDs.
- Runs a controller node that turns toward the target and moves only for intruders.

## System loop (short version)
1. Camera image comes in.
2. YOLOv8 detects people.
3. ByteTrack assigns track IDs.
4. ArUco is searched in a chest ROI.
5. Each person is marked as authorized or intruder.
6. Controller picks a target and publishes `cmd_vel`.

## Implementation details (what we built)

### Perception node (YOLO + ByteTrack + ArUco)
- Subscribes to `image` (RGB).
- Runs `model.track()` with `persist=True`, so ByteTrack keeps state.
- Only uses class `0` (person).
- Creates a list of tracks with `id, x1, y1, x2, y2, conf`.
- Picks the largest person as the default target.
- Normalizes target center to `[-1..1]` for controller input.

ArUco logic:
- Uses a chest ROI inside the person box. This is `roi_y_min_ratio` to
  `roi_y_max_ratio`, plus `roi_x_margin`.
- Skips small boxes with `aruco_min_bbox_px`.
- Optional full-body fallback if ROI fails
  (`aruco_roi_fallback_full`).
- Optional blur and corner refine to improve detection.
- Limits how many tracks to scan with `aruco_max_tracks`.
- `aruco_hold_time` reuses last marker for a short gap.
- `sticky_marker_auth` keeps marker IDs authorized for the whole session.
- `track_auth_hold_time` keeps a track authorized for N seconds after last marker.
- Association is per track per frame. If the marker is seen in the ROI,
  we mark that track as authorized right away.

Published data:
- `tracks` (`Float32MultiArray`):
  `[id, x1, y1, x2, y2, conf]` repeated.
- `target` (`PointStamped`):
  `point.x` and `point.y` are normalized center, `point.z` is confidence.
- `person_status` (`Int32MultiArray`):
  `[track_id, authorized(1/0), marker_id]` repeated.
- `target_authorized` (`Bool`).
- `aruco_ids` (`Int32MultiArray`) list of marker IDs.
- `debug_image` with red/green boxes and labels.

### Controller node (target and motion)
- Subscribes to `tracks`, `person_status`, `target`, and `camera_info`.
- Picks the largest intruder (authorized == 0). This is the main target.
- If no intruder exists, it can still rotate to the current target.
- It only moves forward when an intruder is present
  (`move_on_intruder_only`).
- Angular velocity is proportional to horizontal error, with smoothing.
- Linear velocity is scaled by bbox area ratio:
  bigger box means closer, so it slows down.
- `approach_area_ratio` sets the stop distance.
- This is reactive control. There is no global path planning.

### World and scenario
- World is a simple 4-wall rectangle.
- Pedestrians are defined in the Pedsim scenario XML.
- The ArUco model is in `models/person_standing_aruco`.

## Key files
- `launch/warehouse_sim.launch` main launch file.
- `config/perception.yaml` YOLO + ArUco settings.
- `config/controller.yaml` motion settings.
- `scenarios/eng_hall.xml` pedestrian scenario.
- `worlds/eng_hall.world` Gazebo world with 4 walls.
- `models/actor_model_aruco.sdf` + `models/person_standing_aruco` ArUco human.
- `rviz/navigation.rviz` RViz config.

## Setup (fresh workspace)
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# Main project
git clone https://github.com/190ibrahim/warehouse-surveillance.git

# Simulation dependencies
git clone https://github.com/TempleRAIL/pedsim_ros_with_gazebo.git
git clone https://github.com/TempleRAIL/robot_gazebo.git

# TurtleBot2 packages (installs into ./turtlebot2_noetic_packages)
wget https://raw.githubusercontent.com/zzuxzt/turtlebot2_noetic_packages/master/turtlebot2_noetic_install.sh
chmod +x turtlebot2_noetic_install.sh
./turtlebot2_noetic_install.sh

# Python deps for perception
python3 -m pip install --user ultralytics opencv-contrib-python

cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Run
```bash
source ~/catkin_ws/devel/setup.bash
roslaunch warehouse_surveillance warehouse_sim.launch
```

Optional mapping (new terminal):
```bash
source ~/catkin_ws/devel/setup.bash
roslaunch robot_gazebo gmapping_demo.launch
```

## Topics and remaps
- Camera image default: `/zed/zed_node/rgb_raw/image_raw_color`
- Camera info default: `/zed/zed_node/rgb_raw/camera_info`
- Velocity command: `/cmd_vel_mux/input/navi`

If you use a different camera, edit the remaps in
`launch/warehouse_sim.launch`.

## How authorization works
- `authorized_ids` in `config/perception.yaml` is the allow list (default `[7]`).
- If a marker matches, the person is marked authorized.
- `sticky_marker_auth` keeps that marker ID authorized for the session.
- `track_auth_hold_time` keeps a track authorized for N seconds after last seen.

## Key parameters (short list)
Perception (`config/perception.yaml`):
- `conf`, `iou`, `imgsz`: YOLO detection settings.
- `max_fps`: limits how often we process frames.
- `aruco_use_roi`, `roi_*`: chest ROI settings.
- `aruco_max_tracks`: max people to scan for ArUco each frame.
- `aruco_hold_time`: short marker reuse window.
- `sticky_marker_auth`, `track_auth_hold_time`: authorization memory.

Controller (`config/controller.yaml`):
- `angular_kp`, `max_angular`: turning response.
- `linear_speed`, `max_linear`: forward speed.
- `approach_area_ratio`: stop distance based on bbox size.

## Control logic
- Prefer intruders. If none, it can still rotate to the current target.
- Move forward only when an intruder is present (`move_on_intruder_only`).
- Slow down when the person is close using bbox area ratio
  (`approach_area_ratio` in `config/controller.yaml`).
- No obstacle avoidance. Keep speed low.

## Constraints
- Track IDs are not stable. They can change when the detector misses a frame.
- ArUco detection depends on marker size, view angle, and ROI settings.
- No depth or obstacle avoidance. This is vision only.

## Common issues and fixes

**No camera image**
- Check the topic with `rostopic list | grep image_raw`.
- Update the remap in `launch/warehouse_sim.launch`.

**`import cv2.aruco` fails**
- `python3 -m pip install --user opencv-contrib-python`

**`/odom` missing**
- Build the plugin and re-source:
  `catkin_make --pkg kobuki_gazebo_plugins`
  then `source devel/setup.bash`

**TF_REPEATED_DATA warnings**
- You likely started extra static TF publishers.
- Kill duplicates and relaunch.

**Track IDs jump to high numbers**
- This is normal for ByteTrack when detections drop for a frame.
- Reduce `max_fps` skipping, lower `conf`, or use a local ByteTrack config.

**ArUco not detected or appears on the wall**
- Make sure the actor SDF points to `person_standing_aruco`.
- Check the texture path in the model folder.
- Increase ROI size or allow full-body fallback.

**Pedsim crash (exit code -11)**
- Validate scenario XML:
  `xmllint --noout ~/catkin_ws/src/warehouse_surveillance/scenarios/eng_hall.xml`

**No `/clock` in sim time**
- Make sure Gazebo is running and `use_sim_time` is true.

## Notes
- YOLO weights are at `yolo/yolov8n.pt`.
- RViz config lives at `rviz/navigation.rviz`.
