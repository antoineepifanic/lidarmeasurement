# README — OAK-D Lite Package Detection & Measurement

## Overview
This script detects packages moving on a conveyor using an OAK-D Lite camera (Canny edge detection) and measures their width, length, height, orientation (yaw), and lateral position (Y) along the conveyor. Height is obtained from a 360° LiDAR (`ldlidar_ros2` / ROS2), and the horizontal dimensions are corrected using that height to compensate for camera perspective.

## How it works
- **Detection**: Canny edge detection on the camera crop (no color segmentation), followed by morphological closing, temporal accumulation over several frames, and hole filling to get solid package blobs.
- **Border removal**: pressing `C` while the conveyor is empty captures the static edges of the conveyor's physical borders and erases them from the edge map, so they are never mistaken for a package.
- **Height**: the LiDAR node listens to a narrow angular window centered on the conveyor and averages the closest points (the top of the package), converting distance to height.
- **Stability**: a package is only reported once its contour has been detected consistently for several frames, with a short tolerance for missed frames (finger passing by, brief occlusion, etc.).
- **Final size**: for each package, the few frames closest to the image center are used to compute a median (most reliable) size, saved to a small history log.

## Requirements
- ROS2 installed and sourced
- `ldlidar_ros2` package and driver for the LiDAR
- Python packages: `opencv-python`, `depthai` (v3 API), `numpy`, `rclpy`

## Before running: calibration required
1. Launch the LiDAR: `ros2 launch ldlidar_ros2 ld14.launch.py`
2. Identify the LiDAR angle that corresponds to your conveyor (RViz2 or `ros2 topic echo /scan --field ranges`).
3. Update `ANGLE_MIN_DEG` / `ANGLE_MAX_DEG` in the script accordingly.
4. Adjust `MAX_PACKAGE_AREA` and `CM_PER_PX_RATIO` if the camera setup (distance, crop) changes.

## Running
```bash
source /opt/ros/<your_distro>/setup.bash
source ~/ldlidar_ros2_ws/install/setup.bash
python3 test_camera_en.py
```
Make sure the `ldlidar_ros2` node is already running in a separate terminal (or launched by the script if enabled).

## Controls
- `Q`: quit
- `C`: capture the background — **run this once with an empty conveyor** so the borders are learned and ignored

## Windows displayed
- **OAK-D Lite - Main Feed**: live camera feed with the locked package, its bounding box, and measurements
- **Debug - Canny (indicative only)**: raw edge map used for tuning (Sigma trackbar)
- **Package history (debug)**: log of the final, cleaned-up size of each package that has passed