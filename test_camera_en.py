"""
OAK-D Lite — Canny detection (outline only) + Height via 360° LiDAR (ldlidar_ros2 / ROS2)
v6.4 update:
  - Detection using Canny only (no color mask)
  - Fixed LiDAR height calculation (top of package instead of median)
  - Y coordinate (conveyor width) in cm from the package center
  - Yaw referenced to the LEFT EDGE (physical conveyor border)
  - Y stored and displayed in the "History" window
  - Conveyor border removal by ERASING their edges from the contour map
    (learned on an EMPTY conveyor, key C). Valid because the package never
    touches the border -> only the border is erased, never the package.

CHANGES FROM v6.3:
  In v6.3, the border mask was only used to reject a contour whose CENTER
  fell on a border. It did not touch the edges, so:
    - the "Debug - Canny" window did not change when pressing C;
    - most importantly, it did NOT fix the real problem (a border edge too
      close to the package that degrades its detection), since the border
      edge remained present in the contour map.
  v6.4: since the package never TOUCHES the physical border (there is
  always a strip of belt between the two), the border edges can safely be
  erased from the contour map:
        edges = edges AND NOT border_mask
  Immediate and VISIBLE effect in "Debug - Canny": the striped bands
  disappear, only the package remains. No package edge is affected, since
  none lies under the mask (no contact = no hole).

CHANGES FROM v6.2:
  1) History + Y: the package's Y coordinate is now accumulated together
     with the dimensions and saved into recorded_packages, then displayed
     in the "Package history" window. This is valid: the package only
     translates along x (conveyor direction), so physically its width (y)
     position is CONSTANT throughout its pass; taking the median near the
     center only smooths detection noise, it does not mix genuinely
     different positions.

  2) Borders: see above (method finalized in v6.4).

CHANGES FROM v6.1:
  1) Package Y coordinate: the position of the CENTER of the package's
     surface along the conveyor width is now displayed, in cm. Convention:
     Y = 0 at the LEFT edge of the crop, Y = max at the RIGHT edge. The
     conversion uses the existing CM_PER_PX_RATIO (crop's cx × CM_PER_PX_RATIO).
     The length coordinate (x along the conveyor) is intentionally not
     computed.

  2) Orientation (yaw): previously, an angle of 0° meant the package's long
     side was PARALLEL to the image's horizontal axis (top/bottom edges). It
     is now referenced to the LEFT EDGE (image's vertical axis), which is
     the conveyor's physical border: this is much more interpretable. In
     practice, this is a simple 90° offset followed by renormalization into
     [-90, 90]. So 0° = package aligned with the conveyor's left edge.

CHANGES FROM v5:
  Before (v5): the MEDIAN of all valid points in the conveyor's 30° angular
  window was used. This works well for a tall box that occupies a large
  part of that window, but for a low box (< ~10 cm), it only intercepts a
  small handful of the ~30 scan beams: most of the remaining points still
  see the empty belt around the box. The median is then pulled toward the
  empty-conveyor distance, the computed height becomes ≈ 0 or negative, and
  the safety filter (height > 0) rejects the measurement -> H stays "--".

  Now (v6): since "closer to the LiDAR = higher" (confirmed: the LiDAR and
  camera are mounted vertically above the conveyor), only the points
  CLOSEST to the window are of interest, regardless of how many beams
  actually hit the box. The average of the N closest points is taken (N=3
  by default) rather than a raw minimum, so as not to be fooled by a single
  isolated noise point, while remaining sensitive even if the box only
  intercepts 2 or 3 beams out of ~30.

The rest (temporal stability, geometric filters, dimensions, interface) is
strictly identical to v5. No influence from the color mask: Canny only.

⚠️ CALIBRATION REQUIRED before use:
  You must determine ANGLE_CENTER_DEG, the angle (in the LiDAR's frame, as
  published on /scan) that physically corresponds to your conveyor's
  location. Method: run `ros2 launch ldlidar_ros2 ld14.launch.py`, open
  RViz2 (or run `ros2 topic echo /scan --field ranges`), place an object at
  the exact conveyor location and identify the matching angle (0° = LiDAR
  reference, direction given by the laser_scan_dir parameter of its launch
  file).

Prerequisites: ROS2 environment sourced BEFORE launching this script
  source /opt/ros/<your_distro>/setup.bash
  source ~/ldlidar_ros2_ws/install/setup.bash
  and the ldlidar_ros2 node must already be running (ros2 launch ldlidar_ros2
  ld14.launch.py) in a separate terminal, OR be launched by this script (see
  below, disabled by default to keep things simple).

Quit: press Q
"""

import math
import threading
from collections import deque

import cv2
import depthai as dai
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

# ─── 360° LiDAR parameters (ldlidar_ros2) ────────────────────────────────────
SCAN_TOPIC             = "/scan"
ANGLE_MIN_DEG          = 70.0    # <-- TO CALIBRATE: lower bound of your conveyor
ANGLE_MAX_DEG          = 100.0   # <-- TO CALIBRATE: upper bound of your conveyor
ANGLE_CENTER_DEG       = (ANGLE_MIN_DEG + ANGLE_MAX_DEG) / 2.0
ANGLE_WIDTH_DEG        = ANGLE_MAX_DEG - ANGLE_MIN_DEG
CONVEYOR_HEIGHT_CM     = 110.5
MIN_VALID_POINTS       = 3       # noise safety: too few valid points -> ignore the measurement

# ─── NEW (v6): package top = closest points, not the median ─────────────────
# The N closest points in the angular window are averaged to estimate the
# top of the package. N=1 would be a raw minimum (sensitive to noise from a
# single point); too large a value becomes a disguised median again and
# falls back into the same trap as before for small boxes. 3 is a good
# compromise with ~30 points in the window.
NB_TOP_POINTS          = 3

# ─── Physical parameters (unchanged) ─────────────────────────────────────────
MARGIN_X = int((490 - 150) * 0.05)
MARGIN_Y = int((350 - 50)  * 0.05)

X_MIN = (150 + MARGIN_X) + 51
X_MAX = 490 - MARGIN_X
Y_MIN, Y_MAX = 50 + MARGIN_Y, 350 - MARGIN_Y

CROP_CENTER_X = (X_MAX - X_MIN) // 2
CROP_CENTER_Y = (Y_MAX - Y_MIN) // 2

MIN_PACKAGE_AREA = 5000
MAX_PACKAGE_AREA = 60000  # <-- TO CALIBRATE: max area (px²) of A SINGLE package seen from above.
                           # Acts as a safeguard against merging: a blob bigger than this is
                           # almost certainly 2 nearby packages fused together, or a
                           # package+finger/stray object fused by the morphological closing.
                           # It is rejected rather than validated (see also CANNY_CLOSING_SIZE
                           # below, reduced to limit these merges at the source).
CM_PER_PX_RATIO = 26.5 / 114.0

# ─── Automatic Canny parameters (kept for debug display only) ───────────────
CANNY_SIGMA = 0.33

# ─── Temporal stability parameters ───────────────────────────────────────────
MIN_STABLE_FRAMES = 5
stability_counter = 0
last_valid_contour        = None  # last valid contour found (used during the grace period)
frames_since_last_hit     = 0     # number of consecutive frames without a valid detection

# ─── Saving the "real" size of a package (median near the center) ───────────
# The package's most reliable measurement is when it is close to the center
# of the image (minimal parallax/tilt effect there, see discussion). So, for
# EACH package currently being tracked, its measurements (center_dist, W, L,
# H, yaw) are accumulated on every locked frame, then at the end of its pass
# (when it truly leaves the field of view, see MISS_TOLERANCE) only the few
# frames where it was closest to the center are kept, and the median is
# taken -> a single "real" size per package, insensitive to a single frame's
# noise.
NB_BEST_CENTER_FRAMES  = 5  # number of frames (closest to center) used for the median
package_measurement_history = []  # measurements of the package currently tracked: list of (center_dist, width_cm, length_cm, height_cm, yaw_deg, y_coord_cm)
recorded_packages            = []  # final saved sizes, one per complete package: list of (width_cm, length_cm, height_cm, yaw_deg, y_coord_cm)

# ─── Geometric filter parameters ─────────────────────────────────────────────
MAX_ASPECT_RATIO     = 4.0
MIN_CONVEXITY_INDEX  = 0.80

# ─── Canny contour closing parameters ────────────────────────────────────────
# REDUCED (15->7, 2it->1it): a 15px kernel on a crop barely ~290px wide also
# fills in REAL gaps between two distinct objects (2 nearby packages, a
# package+finger), fusing their contours even before fill_holes. A smaller
# kernel still closes the micro-breaks of a single frame (the 4th side that
# flickers), but stops fusing genuinely separate objects. Closing the 4th
# side across several frames is now mainly handled by the temporal
# accumulation below (NB_ACCUMULATION_FRAMES increased) rather than by the
# kernel size.
CANNY_CLOSING_SIZE = 7   # morphological closing kernel (odd number recommended)
CLOSING_ITERATIONS = 1

# ─── Temporal accumulation: smooths flickering from frame-to-frame noise ────
# Increased (3->5) to compensate for the reduced CANNY_CLOSING_SIZE: it is
# now the union over several frames that closes the 4th side when it
# flickers, rather than an oversized spatial kernel that also fused distinct
# objects.
NB_ACCUMULATION_FRAMES = 5
edges_buffer = deque(maxlen=NB_ACCUMULATION_FRAMES)

# ─── Tolerance to frames without detection (hysteresis) ──────────────────────
# Before: any single frame without a valid contour reset stability_counter to
# 0, which restarted the whole stabilization cycle -> the lock appeared to
# flicker even when the package was actually there 9 frames out of 10. Now,
# up to MISS_TOLERANCE consecutive frames without detection are tolerated by
# continuing to display/consider the last valid contour, without losing the
# lock. This also absorbs a brief finger pass or a neighboring package
# causing one or two missed frames.
MISS_TOLERANCE = 4

# ─── Illumination normalization parameters ───────────────────────────────────
ILLUM_BLUR_SIZE = 51  # must be odd; increase if lighting gradients are wider

# ─── Conveyor border mask (learned on an empty background) ──────────────────
# Filled in by capture_border_background() when C is pressed with an EMPTY
# conveyor. Stays None until a capture has been made -> v6.2 behavior.
# Usage: everything falling under this mask is erased from the edge map
# (edges AND NOT border_mask). Since the package never touches the border,
# only the border edge is removed, never the package's.
border_mask = None
# Mask dilation: slightly widens the learned borders to absorb noise and a
# slight jitter of the border from one frame to another. Larger = a wider
# band around the border is erased (more tolerant, but eats more into the
# usable area closest to the edge).
BORDER_MASK_DILATION = 7   # elliptical kernel (odd number recommended)
BORDER_DILATION_ITER  = 2


def normalize_illumination(gray_channel, blur_size=ILLUM_BLUR_SIZE):
    """
    Flattens slow lighting variations (shadows, halos, light gradients on
    the conveyor background) by dividing the image by a blurred (low
    frequency) estimate of the background, then rescaling around an average
    value of 128. High-frequency details (package edges, texture) are
    preserved; slow background variations disappear.
    """
    if blur_size % 2 == 0:
        blur_size += 1
    estimated_background = cv2.GaussianBlur(gray_channel, (blur_size, blur_size), 0)
    estimated_background = np.where(estimated_background == 0, 1, estimated_background)
    normalized = (gray_channel.astype(np.float32) / estimated_background.astype(np.float32)) * 128.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def capture_border_background(empty_crop, sigma):
    """
    Learns, from an EMPTY conveyor frame, where the static edges are
    (striped bands from borders, rails, etc.) and turns them into a mask
    that will be ERASED from the edge map on every frame.

    The exact same preprocessing as the main loop is redone (illumination
    normalization -> blur -> Canny) so the learned edges match those that
    will appear in real time. Then a slight DILATION is applied: the learned
    border covers a band slightly wider than the bare edge, which tolerates
    noise and a small shift. Since the package never touches the border,
    widening the mask a bit does not erase any package edge.
    """
    gray = cv2.cvtColor(empty_crop, cv2.COLOR_BGR2GRAY)
    gray_norm = normalize_illumination(gray)
    blurred = cv2.GaussianBlur(gray_norm, (7, 7), 0)
    min_thresh, max_thresh = canny_auto(blurred, sigma=sigma)
    background_edges = cv2.Canny(blurred, min_thresh, max_thresh)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (BORDER_MASK_DILATION, BORDER_MASK_DILATION)
    )
    return cv2.dilate(background_edges, kernel, iterations=BORDER_DILATION_ITER)


def fill_holes(mask):
    """
    Fills internal holes of a binary mask (255/0): the background is filled
    from a corner via flood fill, inverted, then combined with the original
    mask. Any pixel unreachable from the outside (i.e. a hole surrounded by
    white) becomes white.

    IMPORTANT: an artificial 1-pixel black border is added first. Without
    it, if mask noise touches the edge of the crop, the starting pixel
    (0,0) is already white: the flood fill then fills nothing new, its
    inverse becomes the exact inverse of the mask, and the final OR always
    produces a fully white image. The artificial border guarantees the
    starting point is always background, regardless of the mask's actual
    content up to its edges.
    """
    padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = padded_mask.shape
    flood_mask = padded_mask.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_mask, mask_ff, (0, 0), 255)
    flood_mask_inv = cv2.bitwise_not(flood_mask)
    filled_pad = padded_mask | flood_mask_inv
    return filled_pad[1:-1, 1:-1]


# ─── ROS2 node: 360° LiDAR -> conveyor distance bridge ───────────────────────
class LidarConveyorBridge(Node):
    """
    Subscribes to the full 360° LiDAR scan and only extracts the angular
    slice physically corresponding to the conveyor. Exposes a single
    distance in centimeters, protected by a lock for thread-safe access
    from the OpenCV loop.

    v6: the returned distance is now the AVERAGE OF THE N CLOSEST POINTS in
    the angular window (top of the package), rather than the median of all
    points in the window. See the explanation at the top of the file.
    """

    def __init__(self, angle_center_deg, angle_width_deg):
        super().__init__('lidar_conveyor_bridge')
        self.angle_center = math.radians(angle_center_deg)
        self.angle_half_width = math.radians(angle_width_deg) / 2.0

        self._dist_cm = None
        self._lock = threading.Lock()

        self.sub = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Listening on {SCAN_TOPIC}, angular window "
            f"[{angle_center_deg - angle_width_deg/2:.1f}°, "
            f"{angle_center_deg + angle_width_deg/2:.1f}°], "
            f"top = average of the {NB_TOP_POINTS} closest points"
        )

    @staticmethod
    def _diff_angle(a, b):
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    def _scan_callback(self, msg: LaserScan):
        valid_values = []
        angle = msg.angle_min
        for r in msg.ranges:
            if abs(self._diff_angle(angle, self.angle_center)) <= self.angle_half_width:
                if msg.range_min <= r <= msg.range_max and not math.isinf(r) and not math.isnan(r):
                    valid_values.append(r)
            angle += msg.angle_increment

        if len(valid_values) >= MIN_VALID_POINTS:
            # v6: ascending sort -> the first elements are the points
            # CLOSEST to the LiDAR, hence physically the highest (the top of
            # the package if there is one, otherwise the belt itself).
            # The NB_TOP_POINTS closest are averaged: small enough to stay
            # sensitive to a box that only intercepts a few beams, large
            # enough to ignore an isolated, abnormally close noise point.
            valid_values.sort()
            k = min(NB_TOP_POINTS, len(valid_values))
            closest_values = valid_values[:k]
            dist_m = sum(closest_values) / k
            with self._lock:
                self._dist_cm = dist_m * 100.0
        else:
            with self._lock:
                self._dist_cm = None

    def get_distance_cm(self):
        with self._lock:
            return self._dist_cm


def get_package_height(lidar_node: LidarConveyorBridge):
    dist_cm = lidar_node.get_distance_cm()
    if dist_cm is None:
        return None
    height = CONVEYOR_HEIGHT_CM - dist_cm
    return round(height, 1) if height > 0 else None


def canny_auto(blurred_gray_image, sigma=CANNY_SIGMA):
    """Kept only for the visual debug window."""
    median      = np.median(blurred_gray_image)
    min_thresh  = int(max(0,   (1.0 - sigma) * median))
    max_thresh  = int(min(255, (1.0 + sigma) * median))
    return min_thresh, max_thresh


def clean_contour(cnt):
    """
    Repairs the geometry of a contour deformed by the texture/pattern of a
    "busy" package (logos, labels, gradients), WITHOUT touching the Canny
    detection itself:
      1) Convex hull (cv2.convexHull): a package is convex by nature, so
         this directly fills in the small notches created by the pattern
         -- this is what was dragging MIN_CONVEXITY_INDEX down on small
         busy-patterned packages.
      2) approxPolyDP on this hull: smooths the last small zigzags/bumps
         (caused by the pattern) while hugging the contour closely enough
         not to deform the true rectangular shape (epsilon proportional to
         the perimeter, so it adapts to the package's size).
    The returned contour is used for ALL of the following steps (geometric
    validation, measurement, display, accumulation for the history) so as
    to stay consistent throughout the pipeline.
    """
    hull = cv2.convexHull(cnt)
    perimeter = cv2.arcLength(hull, True)
    if perimeter == 0:
        return hull
    epsilon = 0.02 * perimeter  # <-- TUNE IF NEEDED: smaller = hugs closer, larger = smooths more
    return cv2.approxPolyDP(hull, epsilon, True)


def contour_geometry_valid(cnt):
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    if w == 0 or h == 0:
        return False

    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    convexity_index = cv2.contourArea(cnt) / hull_area
    if convexity_index < MIN_CONVEXITY_INDEX:
        return False

    return True


def compute_yaw(rect):
    """
    Returns the package's yaw angle in degrees within [-90, 90], from the
    minAreaRect, REFERENCED TO THE LEFT EDGE of the conveyor (image's
    vertical axis).

    OpenCV returns an angle in ]-90, 0] (or [0, 90) depending on version)
    tied to the 'width' side; it is first normalized so it relates to the
    LONGEST SIDE (the package's length). At this stage, 0° would mean "long
    side parallel to the HORIZONTAL axis of the image" (top/bottom edge).

    v6.2: a 90° offset is then applied to reference the angle to the LEFT
    EDGE (vertical axis), which is the conveyor's physical border. So 0° =
    the package's long side aligned with the conveyor's left edge.
    Re-normalized into [-90, 90].
    """
    (w, h) = rect[1]
    angle = rect[2]
    if w < h:
        angle += 90.0

    # v6.2: 90° offset to switch from the "top edge" reference (horizontal
    # axis) to the "left edge" reference (vertical axis = conveyor border).
    angle += 90.0

    # bring back into [-90, 90]
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return round(angle, 1)


def finalize_current_package():
    """
    Called when a tracked package TRULY leaves the field of view (end of
    the grace period, not just a flicker). Among all measurements
    accumulated during its pass, takes the NB_BEST_CENTER_FRAMES frames
    where it was closest to the center of the image, and computes their
    median: this value is treated as the package's "real" size. Then clears
    the accumulator for the next package.
    """
    global package_measurement_history, recorded_packages

    if package_measurement_history:
        sorted_measurements = sorted(package_measurement_history, key=lambda m: m[0])
        best_measurements = sorted_measurements[:min(NB_BEST_CENTER_FRAMES, len(sorted_measurements))]

        final_width = float(np.median([m[1] for m in best_measurements]))
        final_length = float(np.median([m[2] for m in best_measurements]))
        valid_heights = [m[3] for m in best_measurements if m[3] is not None]
        final_height = float(np.median(valid_heights)) if valid_heights else None

        # yaw: circular mean over 2*angle (orientation ambiguous at 180°)
        yaws = [math.radians(2.0 * m[4]) for m in best_measurements]
        final_yaw = math.degrees(0.5 * math.atan2(
            np.mean([math.sin(a) for a in yaws]),
            np.mean([math.cos(a) for a in yaws]),
        ))
        final_yaw = round(final_yaw, 1)

        # Y: the package only moves along x, so y is physically constant ->
        # the median near the center only smooths detection noise.
        final_y = float(np.median([m[5] for m in best_measurements]))

        recorded_packages.append((final_width, final_length, final_height, final_yaw, final_y))

    package_measurement_history = []


def draw_package_history(recorded_packages, max_lines=15):
    """
    Separate debug window: lists the "real" sizes already saved, one line
    per complete package. Has no impact on the main window.
    """
    n_lines = min(len(recorded_packages), max_lines)
    img = np.zeros((70 + 25 * max(n_lines, 1), 500, 3), dtype=np.uint8)

    cv2.putText(img, "Package history (real size - median near center)",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(img, f"Total recorded packages: {len(recorded_packages)}",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    start = max(0, len(recorded_packages) - max_lines)
    y = 65
    for i, (width, length, height, yaw, y_coord) in enumerate(recorded_packages[start:], start=start + 1):
        if height is not None:
            text = f"Package #{i}: W={width:.1f} L={length:.1f} H={height:.1f} Y={y_coord:.1f} Yaw={yaw:.1f}"
        else:
            text = f"Package #{i}: W={width:.1f} L={length:.1f} H=-- Y={y_coord:.1f} Yaw={yaw:.1f}"
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1)
        y += 25

    return img


def main():
    global stability_counter, edges_buffer
    global last_valid_contour, frames_since_last_hit
    global package_measurement_history, recorded_packages
    global border_mask

    # ─── ROS2 init + LiDAR bridge node ────────────────────────────────────────
    rclpy.init()
    lidar_node = LidarConveyorBridge(ANGLE_CENTER_DEG, ANGLE_WIDTH_DEG)
    spin_thread = threading.Thread(target=rclpy.spin, args=(lidar_node,), daemon=True)
    spin_thread.start()

    # ─── OAK-D pipeline (RGB only) — depthai v3 API ──────────────────────────
    pipeline = dai.Pipeline()
    color_cam = pipeline.create(dai.node.Camera).build()
    color_queue = color_cam.requestOutput(
        (640, 400), type=dai.ImgFrame.Type.BGR888p, fps=30
    ).createOutputQueue()

    # ─── Interface ─────────────────────────────────────────────────────────────
    cv2.namedWindow("OAK-D Lite - Main Feed")
    cv2.namedWindow("Debug - Canny (indicative only)")
    cv2.namedWindow("Package history (debug)")

    def nothing(x):
        pass

    cv2.createTrackbar("Sigma x100", "Debug - Canny (indicative only)", 33, 100, nothing)

    print("Starting OAK-D Lite v6.4 (Canny-only detection + 360° LiDAR for height)...")
    print("  Q: quit")
    print("  C: capture the background (EMPTY conveyor) -> learns the borders to ignore")
    print(f"\n  Detection: Canny (outline only)")
    print(f"  LiDAR: angular window centered on {ANGLE_CENTER_DEG}° "
          f"(width {ANGLE_WIDTH_DEG}°) on topic {SCAN_TOPIC}")
    print(f"  Height = average of the {NB_TOP_POINTS} closest LiDAR points (top of the package)")
    print(f"  Y coordinate = position of the package center along the width "
          f"(0 = left edge, max = right edge), in cm")
    print(f"  Yaw = referenced to the left edge (conveyor's physical border)")
    print(f"  Borders: press C with an EMPTY conveyor to have them ignored\n")

    # Last crop/sigma seen, so that the C key (background capture) can work
    # on the current frame even when handled outside the "if color_packet is
    # not None" block.
    last_crop  = None
    last_sigma = CANNY_SIGMA

    try:
        pipeline.start()
        while pipeline.isRunning():
                color_packet = color_queue.tryGet()

                if color_packet is not None:
                    vis_color = color_packet.getCvFrame()
                    crop = vis_color[Y_MIN:Y_MAX, X_MIN:X_MAX]
                    last_crop = crop

                    # ─── Main detection: Canny, with reinforced closing ──────────
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    # Illumination normalization BEFORE Canny: reduces false
                    # edges caused by slow lighting variations of the
                    # conveyor background, without erasing the package's
                    # real edges.
                    gray_norm = normalize_illumination(gray)
                    blurred = cv2.GaussianBlur(gray_norm, (7, 7), 0)

                    sigma_val = cv2.getTrackbarPos("Sigma x100", "Debug - Canny (indicative only)") / 100.0
                    last_sigma = sigma_val
                    min_thresh, max_thresh = canny_auto(blurred, sigma=sigma_val)
                    edges = cv2.Canny(blurred, min_thresh, max_thresh)

                    # ─── Conveyor border removal ──────────────────────────────────
                    # If a background has been captured (key C), erase from
                    # the edge map anything that falls under a learned
                    # border:
                    #     edges = edges AND NOT border_mask
                    # This acts here, on the raw map -> visible right away in
                    # the "Debug - Canny" window. Since the package never
                    # TOUCHES the border (there is always a strip of belt
                    # between the two), no package edge lies under the mask:
                    # only the border is erased, never the package.
                    if border_mask is not None:
                        edges = cv2.bitwise_and(edges, cv2.bitwise_not(border_mask))

                    # Slight dilation to thicken edges before closing
                    kernel_dilate = np.ones((3, 3), np.uint8)
                    edges = cv2.dilate(edges, kernel_dilate, iterations=1)

                    # Strong closing: fills small gaps in the contour (this
                    # was the original problem).
                    kernel_close = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (CANNY_CLOSING_SIZE, CANNY_CLOSING_SIZE)
                    )
                    closed_edges = cv2.morphologyEx(
                        edges, cv2.MORPH_CLOSE, kernel_close, iterations=CLOSING_ITERATIONS
                    )

                    # Temporal accumulation (OR over the last N frames): if
                    # the contour's opening varies slightly from one frame
                    # to another due to noise, it's enough for one of the
                    # last N frames to have closed it for the union to be
                    # closed. This removes the flicker where the fill
                    # succeeds only every other frame.
                    edges_buffer.append(closed_edges)
                    accumulated_edges = edges_buffer[0]
                    for e in list(edges_buffer)[1:]:
                        accumulated_edges = cv2.bitwise_or(accumulated_edges, e)

                    # Filling internal holes: turns the closed contour (just
                    # the edges) into a solid blob usable by findContours +
                    # the geometric filters.
                    final_mask = fill_holes(accumulated_edges)

                    # Light cleanup of residual noise
                    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)

                    edges_debug = edges  # for the debug window

                    # ─── Detection: on the closed + filled Canny mask ────────────
                    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)

                    best_contour = None
                    min_center_dist = float('inf')

                    for cnt in contours:
                        raw_area = cv2.contourArea(cnt)
                        if raw_area < MIN_PACKAGE_AREA:
                            continue
                        if raw_area > MAX_PACKAGE_AREA:
                            # Likely merge: 2 nearby packages or a
                            # package+finger fused by the morphological
                            # closing. Rejected rather than validating a
                            # blob that contains more than one object.
                            continue

                        # Geometric cleanup: repairs contours dented by a
                        # busy pattern/logo (see discussion). Everything
                        # else in the loop now works on the cleaned contour.
                        cnt = clean_contour(cnt)

                        cnt_area = cv2.contourArea(cnt)
                        if cnt_area < MIN_PACKAGE_AREA or cnt_area > MAX_PACKAGE_AREA:
                            # The convex hull can slightly increase the area
                            # (filling notches) -> the same bounds are
                            # re-checked on the cleaned contour to stay
                            # consistent.
                            continue
                        if not contour_geometry_valid(cnt):
                            continue

                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            center_dist = (cx - CROP_CENTER_X)**2 + (cy - CROP_CENTER_Y)**2
                            if center_dist < min_center_dist:
                                min_center_dist  = center_dist
                                best_contour = cnt

                    real_detection_this_frame = False

                    if best_contour is not None:
                        # Detection succeeded this frame: this contour is
                        # stored and the missed-frame counter is reset to
                        # zero.
                        real_detection_this_frame = True
                        last_valid_contour    = best_contour
                        frames_since_last_hit = 0
                        stability_counter += 1
                    else:
                        frames_since_last_hit += 1
                        if (last_valid_contour is not None
                                and frames_since_last_hit <= MISS_TOLERANCE):
                            # Short gap (4th side flickering, a finger
                            # briefly passing near the package, a
                            # neighboring package interfering for one
                            # frame): continue with the last known contour
                            # instead of losing everything.
                            # stability_counter is neither incremented nor
                            # reset during this grace period.
                            best_contour = last_valid_contour
                        else:
                            # Too many consecutive missed frames: the
                            # package is truly gone (or was never there)
                            # -> finalize its "real" size (median near the
                            # center) before resetting, ready for the next
                            # package.
                            finalize_current_package()
                            stability_counter    = 0
                            last_valid_contour = None

                    contour_stable = (best_contour is not None
                                      and stability_counter >= MIN_STABLE_FRAMES)

                    height_cm = get_package_height(lidar_node)

                    if contour_stable:
                        rect = cv2.minAreaRect(best_contour)
                        width_px, height_px = rect[1]
                        yaw_deg = compute_yaw(rect)

                        if height_cm is not None:
                            lidar_dist_cm = CONVEYOR_HEIGHT_CM - height_cm
                            factor = lidar_dist_cm / CONVEYOR_HEIGHT_CM
                        else:
                            factor = 1.0

                        width_cm = width_px * CM_PER_PX_RATIO * factor
                        length_cm = height_px * CM_PER_PX_RATIO * factor

                        M  = cv2.moments(best_contour)
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(vis_color, (cx + X_MIN, cy + Y_MIN), 4, (0, 0, 255), -1)

                        # ─── Package Y coordinate along the conveyor width ──────
                        # cx is relative to the crop: 0 = left edge, (X_MAX -
                        # X_MIN) = right edge. Converted to cm using the same
                        # pixel->cm ratio already used for dimensions. Y = 0
                        # at the left edge, Y = max at the right edge.
                        y_coord_cm = cx * CM_PER_PX_RATIO

                        # --- Accumulation for the package's "real" size ---
                        # Only on a real detection (not a grace-period frame
                        # that just redisplays the old contour): otherwise
                        # the same measurement at the same distance from the
                        # center would be duplicated multiple times.
                        if real_detection_this_frame:
                            current_center_dist = (cx - CROP_CENTER_X)**2 + (cy - CROP_CENTER_Y)**2
                            package_measurement_history.append(
                                (current_center_dist, width_cm, length_cm, height_cm, yaw_deg, y_coord_cm)
                            )

                        box        = cv2.boxPoints(rect)
                        box        = np.int32(box)
                        box_global = box + [X_MIN, Y_MIN]

                        cv2.drawContours(vis_color, [box_global], 0, (0, 255, 80), 2)

                        cv2.putText(vis_color, "Package locked",
                                    (X_MIN, Y_MIN - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

                        if height_cm is not None:
                            dims_text = f"W:{width_cm:.1f}  L:{length_cm:.1f}  H:{height_cm:.1f} cm  Y:{y_coord_cm:.1f} cm  Yaw:{yaw_deg:.1f}"
                        else:
                            dims_text = f"W:{width_cm:.1f}  L:{length_cm:.1f}  H:-- cm  Y:{y_coord_cm:.1f} cm  Yaw:{yaw_deg:.1f}"

                    elif best_contour is not None:
                        cv2.putText(vis_color, f"Stabilizing... ({stability_counter}/{MIN_STABLE_FRAMES})",
                                    (X_MIN, Y_MIN - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                        dims_text = "W:--  L:--  H:-- cm  Y:-- cm  Yaw:--"

                    else:
                        cv2.putText(vis_color, "Waiting...",
                                    (X_MIN, Y_MIN - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                        if height_cm is not None:
                            dims_text = f"W:--  L:--  H:{height_cm:.1f} cm  Y:-- cm  Yaw:--"
                        else:
                            dims_text = "W:--  L:--  H:-- cm  Y:-- cm  Yaw:--"

                    cv2.rectangle(vis_color, (10, 10), (560, 50), (0, 0, 0), -1)
                    cv2.putText(vis_color, dims_text, (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.putText(vis_color, f"Canny(debug):{min_thresh}/{max_thresh}",
                                (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

                    lidar_color = (0, 255, 0) if height_cm is not None else (0, 0, 255)
                    cv2.circle(vis_color, (620, 20), 8, lidar_color, -1)
                    cv2.putText(vis_color, "LiDAR", (590, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, lidar_color, 1)

                    cv2.rectangle(vis_color, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 0, 255), 1)

                    cv2.imshow("OAK-D Lite - Main Feed", vis_color)
                    cv2.imshow("Debug - Canny (indicative only)", edges_debug)
                    cv2.imshow("Package history (debug)", draw_package_history(recorded_packages))

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Background capture: the conveyor must be EMPTY at this
                    # moment. This learns the static edges (borders) ->
                    # forbidden-zone mask. Recapture if the lighting changes
                    # significantly.
                    if last_crop is not None:
                        border_mask = capture_border_background(last_crop, last_sigma)
                        print("[C] Background captured: borders learned and erased from "
                              "the edge map (visible in Debug - Canny).")
                    else:
                        print("[C] No frame available for background capture.")

    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        lidar_node.destroy_node()
        rclpy.shutdown()
        print("\nProgram terminated.")


if __name__ == "__main__":
    main()
