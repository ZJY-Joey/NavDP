#!/usr/bin/env python3
"""ROS2 runner that subscribes RealSense bag topics and sends no-goal
requests to NavDP server.

Usage:
1) Start NavDP server separately.
2) Run this node.
3) Play bag in another terminal:
   ros2 bag play <bag_path>
"""

import argparse
import math
import json
import threading
import time
from collections import deque
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import requests
import rclpy
from requests import exceptions as req_exc
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage
from tf2_msgs.msg import TFMessage


class NavDPRos2BagRunner(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("navdp_ros2bag_nogoal_runner")
        self.server_base = f"http://{args.server_host}:{args.server_port}"
        self.stop_threshold = args.stop_threshold
        self.send_hz = args.send_hz
        self.timeout = args.timeout
        self.reset_timeout = args.reset_timeout
        self.print_server_json = args.print_server_json
        self.vis_http_ref = args.vis_http_ref
        self.control_output = args.control_output

        self.rgb_topic = args.rgb_topic
        self.depth_topic = args.depth_topic
        self.camera_info_topic = args.camera_info_topic
        self.odom_topic = args.odom_topic
        self.tf_static_topic = args.tf_static_topic

        self.last_rgb: Optional[np.ndarray] = None
        self.last_depth_m: Optional[np.ndarray] = None
        self.last_camera_info: Optional[CameraInfo] = None
        self.last_odom: Optional[Odometry] = None
        # Accumulate static TF edges from all /tf_static messages.
        self.all_static_transforms: Dict[Tuple[str, str], np.ndarray] = {}

        self.last_send_time = 0.0
        self.last_health_log_time = 0.0
        self._reset_done = False
        self._vis_window_name = "NavDP HTTP Trajectory"
        self._vis_enabled = self.vis_http_ref
        self._last_tf_warn_time = 0.0
        self.base_frame = args.base_frame
        self.camera_optical_frame = args.camera_optical_frame
        self.cmd_vel_topic = args.cmd_vel_topic
        self.pp_gain_v = args.pp_gain_v
        self.pp_gain_w = args.pp_gain_w
        self.pp_gain_theta = args.pp_gain_theta
        self.pp_use_yaw_term = args.pp_use_yaw_term
        self.pp_speed_switch = args.pp_speed_switch
        self.pp_lookahead_slow = args.pp_lookahead_slow
        self.pp_lookahead_fast = args.pp_lookahead_fast
        self.max_linear_speed = args.max_linear_speed
        self.max_angular_speed = args.max_angular_speed
        self.cmd_filter_alpha = args.cmd_filter_alpha
        self.max_linear_acc = args.max_linear_acc
        self.max_angular_acc = args.max_angular_acc
        self._last_cmd_v = 0.0
        self._last_cmd_w = 0.0
        self._last_cmd_time = 0.0
        self._lock = threading.Lock()

        self._session = requests.Session()
        self._cmd_pub = None
        if self.control_output:
            self._cmd_pub = self.create_publisher(
                Twist, self.cmd_vel_topic, 20
            )

        self.create_subscription(
            CompressedImage, self.rgb_topic, self._on_rgb, 20
        )
        self.create_subscription(
            CompressedImage, self.depth_topic, self._on_depth, 20
        )
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info, 10
        )
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 20)
        self.create_subscription(
            TFMessage, self.tf_static_topic, self._on_tf_static, 10
        )

        self.create_timer(
            1.0 / max(self.send_hz, 1e-3), self._maybe_send_request
        )

        self.get_logger().info("Runner started.")
        self.get_logger().info(f"Server: {self.server_base}")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        if self.vis_http_ref:
            self.get_logger().info(
                "Visualization enabled: draw HTTP trajectory on uploaded RGB"
            )
            self.get_logger().info(
                f"Projection frames: base={self.base_frame}, "
                f"camera_optical={self.camera_optical_frame}"
            )
        if self.control_output:
            self.get_logger().info(
                f"Control output enabled: publish cmd_vel to "
                f"{self.cmd_vel_topic}"
            )

    def _on_rgb(self, msg: CompressedImage) -> None:
        image = self._decode_rgb(msg)
        if image is None:
            self.get_logger().warn("Failed to decode RGB compressed image")
            return
        with self._lock:
            self.last_rgb = image

    def _on_depth(self, msg: CompressedImage) -> None:
        depth_m = self._decode_depth_to_meters(msg)
        if depth_m is None:
            self.get_logger().warn("Failed to decode compressedDepth image")
            return
        with self._lock:
            self.last_depth_m = depth_m

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self.last_camera_info = msg

    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            self.last_odom = msg

    def _on_tf_static(self, msg: TFMessage) -> None:
        with self._lock:
            for ts in msg.transforms:
                parent = ts.header.frame_id.strip("/")
                child = ts.child_frame_id.strip("/")
                if not parent or not child:
                    continue
                self.all_static_transforms[(parent, child)] = (
                    self._transform_to_matrix(ts.transform)
                )

    @staticmethod
    def _decode_rgb(msg: CompressedImage) -> Optional[np.ndarray]:
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def _decode_depth_to_meters(msg: CompressedImage) -> Optional[np.ndarray]:
        np_arr = np.frombuffer(msg.data, np.uint8)
        depth_raw = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        # compressedDepth may prepend a small header before PNG payload.
        if depth_raw is None and len(msg.data) > 12:
            np_arr = np.frombuffer(msg.data[12:], np.uint8)
            depth_raw = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if depth_raw is None:
            return None

        if depth_raw.dtype == np.uint16:
            # RealSense depth is usually millimeters.
            return depth_raw.astype(np.float32) / 1000.0

        if depth_raw.dtype == np.float32:
            return depth_raw

        return depth_raw.astype(np.float32)

    @staticmethod
    def _camera_info_to_intrinsic(camera_info: CameraInfo) -> np.ndarray:
        k = np.array(camera_info.k, dtype=np.float32).reshape(3, 3)
        return k

    @staticmethod
    def _extract_traj_points(trajectory: np.ndarray) -> Optional[np.ndarray]:
        if trajectory.size == 0:
            return None

        # Common layouts: (1, N, 3), (N, 3), (1, N, 2), (N, 2)
        if trajectory.ndim == 3:
            if trajectory.shape[0] >= 1:
                pts = trajectory[0]
            else:
                return None
        elif trajectory.ndim == 2:
            pts = trajectory
        else:
            return None

        if pts.shape[-1] < 2:
            return None

        return pts[:, :2].astype(np.float32)

    @staticmethod
    def _extract_traj_pose(trajectory: np.ndarray) -> Optional[np.ndarray]:
        if trajectory.size == 0:
            return None
        if trajectory.ndim == 3 and trajectory.shape[0] >= 1:
            pts = trajectory[0]
        elif trajectory.ndim == 2:
            pts = trajectory
        else:
            return None
        if pts.shape[-1] < 2:
            return None
        return pts.astype(np.float32)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _select_lookahead_index(
        self, pts: np.ndarray, current_speed: float
    ) -> int:
        if pts.shape[0] == 0:
            return 0
        idx = self.pp_lookahead_fast
        if current_speed < self.pp_speed_switch:
            idx = self.pp_lookahead_slow
        return int(self._clamp(float(idx), 0.0, float(pts.shape[0] - 1)))

    def _apply_rate_limit(
        self, target: float, last: float, max_delta: float
    ) -> float:
        if target > last + max_delta:
            return last + max_delta
        if target < last - max_delta:
            return last - max_delta
        return target

    def _publish_cmd(self, v_cmd: float, w_cmd: float) -> None:
        if self._cmd_pub is None:
            return
        msg = Twist()
        msg.linear.x = float(v_cmd)
        msg.angular.z = float(w_cmd)
        self._cmd_pub.publish(msg)

    def _publish_stop_cmd(self) -> None:
        self._publish_cmd(0.0, 0.0)
        self._last_cmd_v = 0.0
        self._last_cmd_w = 0.0

    def _control_from_trajectory(
        self, trajectory: np.ndarray, odom: Optional[Odometry], now: float
    ) -> None:
        if not self.control_output:
            return

        pts = self._extract_traj_pose(trajectory)
        if pts is None or pts.shape[0] == 0:
            self._publish_stop_cmd()
            return

        current_speed = abs(self._last_cmd_v)
        if odom is not None:
            current_speed = abs(odom.twist.twist.linear.x)

        look_idx = self._select_lookahead_index(pts, current_speed)
        target = pts[look_idx]
        x = float(target[0])
        y = float(target[1])
        theta_target = float(target[2]) if target.shape[0] >= 3 else 0.0

        ld = max(math.hypot(x, y), 1e-3)
        alpha = math.atan2(y, x)

        v_des = self.pp_gain_v * ld
        v_des = self._clamp(v_des, 0.0, self.max_linear_speed)

        w_des = self.pp_gain_w * alpha
        if self.pp_use_yaw_term:
            theta_current = 0.0
            theta_err = self._normalize_angle(theta_target - theta_current)
            w_des += self.pp_gain_theta * theta_err
        w_des = self._clamp(
            w_des, -self.max_angular_speed, self.max_angular_speed
        )

        alpha = self._clamp(self.cmd_filter_alpha, 0.0, 0.999)
        v_filt = alpha * self._last_cmd_v + (1.0 - alpha) * v_des
        w_filt = alpha * self._last_cmd_w + (1.0 - alpha) * w_des

        dt = 1.0 / max(self.send_hz, 1e-3)
        if self._last_cmd_time > 0.0:
            dt = max(now - self._last_cmd_time, 1e-3)

        dv_max = self.max_linear_acc * dt
        dw_max = self.max_angular_acc * dt
        v_cmd = self._apply_rate_limit(v_filt, self._last_cmd_v, dv_max)
        w_cmd = self._apply_rate_limit(w_filt, self._last_cmd_w, dw_max)

        self._publish_cmd(v_cmd, w_cmd)
        self._last_cmd_v = v_cmd
        self._last_cmd_w = w_cmd
        self._last_cmd_time = now

    @staticmethod
    def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        n = qx * qx + qy * qy + qz * qz + qw * qw
        if n < 1e-12:
            return np.eye(3, dtype=np.float32)
        s = 2.0 / n
        xx = qx * qx * s
        yy = qy * qy * s
        zz = qz * qz * s
        xy = qx * qy * s
        xz = qx * qz * s
        yz = qy * qz * s
        wx = qw * qx * s
        wy = qw * qy * s
        wz = qw * qz * s
        return np.array(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ],
            dtype=np.float32,
        )

    @classmethod
    def _transform_to_matrix(cls, transform) -> np.ndarray:
        t = transform.translation
        q = transform.rotation
        r = cls._quat_to_rot(q.x, q.y, q.z, q.w)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = r
        mat[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float32)
        return mat

    @staticmethod
    def _invert_transform(mat: np.ndarray) -> np.ndarray:
        inv = np.eye(4, dtype=np.float32)
        r = mat[:3, :3]
        t = mat[:3, 3]
        rt = r.T
        inv[:3, :3] = rt
        inv[:3, 3] = -rt @ t
        return inv

    def _find_transform_matrix(
        self, source_frame: str, target_frame: str
    ) -> Optional[np.ndarray]:
        if source_frame == target_frame:
            return np.eye(4, dtype=np.float32)

        with self._lock:
            tf_snapshot = dict(self.all_static_transforms)

        graph = {}
        for (parent, child), t_parent_child in tf_snapshot.items():
            t_child_parent = self._invert_transform(t_parent_child)
            graph.setdefault(parent, []).append((child, t_parent_child))
            graph.setdefault(child, []).append((parent, t_child_parent))

        source = source_frame.strip("/")
        target = target_frame.strip("/")
        if source not in graph or target not in graph:
            return None

        queue = deque([(source, np.eye(4, dtype=np.float32))])
        visited = {source}

        while queue:
            cur, t_cur_source = queue.popleft()
            if cur == target:
                return t_cur_source
            for nxt, t_nxt_cur in graph.get(cur, []):
                if nxt in visited:
                    continue
                visited.add(nxt)
                t_nxt_source = t_nxt_cur @ t_cur_source
                queue.append((nxt, t_nxt_source))

        return None

    @staticmethod
    def _project_points(
        pts_base: np.ndarray, t_cam_base: np.ndarray, k: np.ndarray
    ) -> Optional[np.ndarray]:
        if pts_base.shape[0] == 0:
            return None

        ones = np.ones((pts_base.shape[0], 1), dtype=np.float32)
        pts_base_h = np.concatenate([pts_base, ones], axis=1)
        pts_cam_h = (t_cam_base @ pts_base_h.T).T
        pts_cam = pts_cam_h[:, :3]

        z = pts_cam[:, 2]
        valid = z > 1e-3
        if np.count_nonzero(valid) < 2:
            return None

        x = pts_cam[valid, 0] / z[valid]
        y = pts_cam[valid, 1] / z[valid]
        fx = float(k[0, 0])
        fy = float(k[1, 1])
        cx = float(k[0, 2])
        cy = float(k[1, 2])
        u = fx * x + cx
        v = fy * y + cy
        return np.stack([u, v], axis=1)

    def _draw_projected_trajectory(
        self,
        image_bgr: np.ndarray,
        trajectory: np.ndarray,
        camera_info: CameraInfo,
    ) -> Optional[np.ndarray]:
        if trajectory.size == 0:
            return None

        if trajectory.ndim == 3 and trajectory.shape[0] >= 1:
            pts = trajectory[0]
        elif trajectory.ndim == 2:
            pts = trajectory
        else:
            return None

        if pts.shape[1] < 2:
            return None

        # Trajectory is treated as (x_forward, y_left, yaw); z is ground plane.
        pts_base = np.stack(
            [
                pts[:, 0].astype(np.float32),
                pts[:, 1].astype(np.float32),
                np.zeros(pts.shape[0], dtype=np.float32),
            ],
            axis=1,
        )

        source_frame = self.base_frame
        camera_frame = camera_info.header.frame_id or self.camera_optical_frame
        t_cam_base = self._find_transform_matrix(source_frame, camera_frame)
        if t_cam_base is None and camera_frame != self.camera_optical_frame:
            t_cam_base = self._find_transform_matrix(
                source_frame, self.camera_optical_frame
            )

        if t_cam_base is None:
            now = time.time()
            if now - self._last_tf_warn_time > 2.0:
                self.get_logger().warn(
                    "No TF chain from base to camera optical frame for "
                    "3D projection. Fallback to 2D overlay. "
                    f"base={source_frame}, "
                    f"camera={camera_frame or self.camera_optical_frame}"
                )
                self._last_tf_warn_time = now
            return None

        k = self._camera_info_to_intrinsic(camera_info)
        uv = self._project_points(pts_base, t_cam_base, k)
        if uv is None or uv.shape[0] < 2:
            return None

        vis = image_bgr.copy()
        poly = np.round(uv).astype(np.int32)
        h, w = vis.shape[:2]
        keep = (
            (poly[:, 0] >= 0)
            & (poly[:, 0] < w)
            & (poly[:, 1] >= 0)
            & (poly[:, 1] < h)
        )
        poly = poly[keep]
        if poly.shape[0] < 2:
            return None

        cv2.polylines(
            vis, [poly], isClosed=False, color=(255, 255, 255), thickness=10
        )
        cv2.polylines(
            vis, [poly], isClosed=False, color=(255, 200, 30), thickness=6
        )

        step = max(2, poly.shape[0] // 6)
        for idx in range(0, poly.shape[0] - 1, step):
            cv2.arrowedLine(
                vis,
                tuple(poly[idx]),
                tuple(poly[min(idx + 1, poly.shape[0] - 1)]),
                color=(60, 120, 255),
                thickness=3,
                tipLength=0.35,
            )

        cv2.circle(vis, tuple(poly[0]), 7, (70, 255, 70), -1)
        cv2.circle(vis, tuple(poly[-1]), 8, (60, 120, 255), -1)
        cv2.putText(
            vis,
            "Projected 3D",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    @staticmethod
    def _draw_gaode_style_trajectory(
        image_bgr: np.ndarray, pts_xy: np.ndarray
    ) -> np.ndarray:
        vis = image_bgr.copy()
        h, w = vis.shape[:2]

        if pts_xy.shape[0] < 2:
            return vis

        x_forward = pts_xy[:, 0]
        y_left = pts_xy[:, 1]

        valid = np.isfinite(x_forward) & np.isfinite(y_left)
        if not np.any(valid):
            return vis

        x_forward = x_forward[valid]
        y_left = y_left[valid]
        if x_forward.size < 2:
            return vis

        # Bottom-center origin, projected to image for an intuitive
        # path overlay.
        origin_x = int(0.5 * w)
        origin_y = int(0.9 * h)

        max_forward = max(float(np.max(x_forward)), 0.5)
        max_lateral = max(float(np.max(np.abs(y_left))), 0.5)
        scale_y = (0.75 * h) / max_forward
        scale_x = (0.45 * w) / max_lateral
        px_per_m = max(10.0, min(scale_x, scale_y))

        poly = []
        for xf, yl in zip(x_forward, y_left):
            px = int(round(origin_x + yl * px_per_m))
            py = int(round(origin_y - xf * px_per_m))
            poly.append((px, py))

        if len(poly) < 2:
            return vis

        poly_np = np.array(poly, dtype=np.int32)

        # Gaode-like route style: white halo + cyan main path.
        cv2.polylines(
            vis, [poly_np], isClosed=False, color=(255, 255, 255), thickness=10
        )
        cv2.polylines(
            vis, [poly_np], isClosed=False, color=(255, 200, 30), thickness=6
        )

        # Add directional arrows along the route.
        arrow_step = max(2, len(poly) // 6)
        for idx in range(0, len(poly) - 1, arrow_step):
            p0 = poly[idx]
            p1 = poly[min(idx + 1, len(poly) - 1)]
            cv2.arrowedLine(
                vis,
                p0,
                p1,
                color=(60, 120, 255),
                thickness=3,
                tipLength=0.35,
            )

        # Start / end markers.
        cv2.circle(vis, poly[0], 7, (70, 255, 70), -1)
        cv2.circle(vis, poly[-1], 8, (60, 120, 255), -1)
        cv2.putText(
            vis,
            "Start",
            (poly[0][0] + 8, poly[0][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (70, 255, 70),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            "Goal",
            (poly[-1][0] + 8, poly[-1][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (60, 120, 255),
            1,
            cv2.LINE_AA,
        )

        return vis

    def _show_vis(self, image_bgr: np.ndarray, trajectory: np.ndarray) -> None:
        if not self._vis_enabled:
            return

        with self._lock:
            camera_info = self.last_camera_info
            has_static_tf = bool(self.all_static_transforms)

        pts_xy = self._extract_traj_points(trajectory)
        if pts_xy is None:
            return

        try:
            vis = None
            if camera_info is not None and has_static_tf:
                vis = self._draw_projected_trajectory(
                    image_bgr, trajectory, camera_info
                )
            if vis is None:
                vis = self._draw_gaode_style_trajectory(image_bgr, pts_xy)
            cv2.imshow(self._vis_window_name, vis)
            # Keep GUI responsive.
            cv2.waitKey(1)
        except cv2.error as exc:
            self.get_logger().error(
                "OpenCV visualization failed, disable vis_http_ref. "
                f"detail={exc}"
            )
            self._vis_enabled = False

    def _log_trajectory_values(self, trajectory: np.ndarray) -> None:
        if trajectory.size == 0:
            self.get_logger().info("trajectory_data: []")
            return

        traj_str = np.array2string(
            trajectory,
            precision=4,
            suppress_small=True,
            separator=", ",
            max_line_width=120,
            threshold=1000000,
        )
        self.get_logger().info(f"trajectory_data:\n{traj_str}")

    def _ensure_reset(self) -> bool:
        with self._lock:
            camera_info = self.last_camera_info

        if camera_info is None:
            self.get_logger().info(
                "Waiting for camera_info before navigator_reset..."
            )
            return False

        if self._reset_done:
            return True

        if not self._check_server_health():
            return False

        intrinsic = self._camera_info_to_intrinsic(camera_info)
        payload = {
            "intrinsic": intrinsic.tolist(),
            "stop_threshold": self.stop_threshold,
            "batch_size": 1,
        }

        url = f"{self.server_base}/navigator_reset"
        try:
            resp = self._session.post(
                url, json=payload, timeout=self.reset_timeout
            )
            resp.raise_for_status()
            self._reset_done = True
            self.get_logger().info(f"navigator_reset success: {resp.text}")
            return True
        except req_exc.ConnectionError as exc:
            self.get_logger().error(
                "navigator_reset connection error. "
                "Please check server process/listen address/firewall. "
                f"detail={exc}"
            )
            return False
        except req_exc.Timeout as exc:
            self.get_logger().error(
                "navigator_reset timeout. "
                "TCP may be reachable but server did not respond in time. "
                f"detail={exc}"
            )
            return False
        except req_exc.HTTPError as exc:
            self.get_logger().error(
                f"navigator_reset HTTP error: {exc}; body={resp.text[:500]}"
            )
            return False
        except Exception as exc:
            self.get_logger().error(f"navigator_reset failed: {exc}")
            return False

    def _check_server_health(self) -> bool:
        url = f"{self.server_base}/"
        try:
            resp = self._session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return True
            now = time.time()
            if now - self.last_health_log_time > 2.0:
                self.get_logger().warn(
                    f"Server health check got HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
                self.last_health_log_time = now
            return False
        except req_exc.ConnectionError as exc:
            now = time.time()
            if now - self.last_health_log_time > 2.0:
                self.get_logger().warn(
                    "Server not reachable yet (connection refused/reset). "
                    f"detail={exc}"
                )
                self.last_health_log_time = now
            return False
        except req_exc.Timeout as exc:
            now = time.time()
            if now - self.last_health_log_time > 2.0:
                self.get_logger().warn(
                    "Server health check timeout. "
                    f"detail={exc}"
                )
                self.last_health_log_time = now
            return False
        except Exception as exc:
            now = time.time()
            if now - self.last_health_log_time > 2.0:
                self.get_logger().warn(f"Server health check failed: {exc}")
                self.last_health_log_time = now
            return False

    def _build_request_payload(
        self, rgb_bgr: np.ndarray, depth_m: np.ndarray
    ) -> Tuple[dict, dict]:
        # Match server shape convention: (batch, H, W, C) flattened for
        # transport.
        depth_send = np.clip(depth_m * 10000.0, 0.0, 65535.0).astype(np.uint16)

        ok_rgb, rgb_jpg = cv2.imencode(".jpg", rgb_bgr)
        ok_depth, depth_png = cv2.imencode(".png", depth_send)
        if not ok_rgb or not ok_depth:
            raise RuntimeError(
                "Failed to encode image/depth before HTTP request"
            )

        files = {
            "image": ("image.jpg", rgb_jpg.tobytes(), "image/jpeg"),
            "depth": ("depth.png", depth_png.tobytes(), "image/png"),
        }
        data = {
            "rgb_time": str(time.time()),
            "depth_time": str(time.time()),
        }
        return files, data

    def _maybe_send_request(self) -> None:
        if not self._ensure_reset():
            return

        with self._lock:
            rgb = None if self.last_rgb is None else self.last_rgb.copy()
            depth_m = (
                None if self.last_depth_m is None else self.last_depth_m.copy()
            )
            odom = self.last_odom
            has_static_tf = bool(self.all_static_transforms)

        if rgb is None or depth_m is None:
            self.get_logger().info("Waiting for RGB + depth frames...")
            return

        now = time.time()
        if now - self.last_send_time < 1.0 / max(self.send_hz, 1e-3):
            return

        if odom is None:
            self.get_logger().warn("/mfla/dr_odom not received yet")
        if not has_static_tf:
            self.get_logger().warn("/tf_static not received yet")

        url = f"{self.server_base}/nogoal_step"
        try:
            files, data = self._build_request_payload(rgb, depth_m)
            resp = self._session.post(
                url, files=files, data=data, timeout=self.timeout
            )
            if resp.status_code != 200:
                self.get_logger().error(
                    f"nogoal_step HTTP {resp.status_code}: {resp.text[:500]}"
                )
                return
            self.last_send_time = now

            body = resp.text
            self.get_logger().info(f"nogoal_step HTTP {resp.status_code}")

            if (
                self.print_server_json
                or self._vis_enabled
                or self.control_output
            ):
                try:
                    parsed = json.loads(body)
                    traj = np.array(parsed.get("trajectory", []))
                    all_values = np.array(parsed.get("all_values", []))
                    if self.print_server_json:
                        self.get_logger().info(
                            "server response: "
                            "trajectory_shape="
                            f"{traj.shape}, "
                            f"all_values_shape={all_values.shape}"
                        )
                        self._log_trajectory_values(traj)
                    if self._vis_enabled:
                        self._show_vis(rgb, traj)
                    if self.control_output:
                        self._control_from_trajectory(traj, odom, now)
                except Exception:
                    self.get_logger().info(
                        f"server raw response: {body[:500]}"
                    )

        except req_exc.ConnectionError as exc:
            self.get_logger().error(
                "nogoal_step connection error. "
                "Check server availability and network route. "
                f"detail={exc}"
            )
        except req_exc.Timeout as exc:
            self.get_logger().error(
                "nogoal_step timeout. "
                "Server may be busy or blocked by heavy inference. "
                f"detail={exc}"
            )
        except Exception as exc:
            self.get_logger().error(f"nogoal_step failed: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROS2 bag -> NavDP no-goal runner"
    )
    parser.add_argument("--server-host", type=str, default="10.12.120.101")
    parser.add_argument("--server-port", type=int, default=8888)
    parser.add_argument("--send-hz", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--reset-timeout", type=float, default=120.0)
    parser.add_argument("--stop-threshold", type=float, default=-0.5)
    parser.add_argument("--print-server-json", action="store_true")
    parser.add_argument(
        "--vis-http-ref",
        action="store_true",
        help="Overlay HTTP trajectory on uploaded RGB and show in real time",
    )
    parser.add_argument(
        "--base-frame",
        type=str,
        default="base_link",
        help="Trajectory source frame used for projection",
    )
    parser.add_argument(
        "--camera-optical-frame",
        type=str,
        default="camera_color_optical_frame",
        help="Target camera optical frame for projection fallback",
    )
    parser.add_argument(
        "--control-output",
        action="store_true",
        help="Follow predicted trajectory and publish cmd_vel",
    )
    parser.add_argument(
        "--cmd-vel-topic",
        type=str,
        default="/cmd_vel",
    )
    parser.add_argument("--pp-gain-v", type=float, default=0.8)
    parser.add_argument("--pp-gain-w", type=float, default=1.6)
    parser.add_argument("--pp-gain-theta", type=float, default=0.5)
    parser.add_argument(
        "--pp-use-yaw-term",
        action="store_true",
        help="Use yaw term from trajectory theta for angular control",
    )
    parser.add_argument("--pp-speed-switch", type=float, default=0.4)
    parser.add_argument("--pp-lookahead-slow", type=int, default=2)
    parser.add_argument("--pp-lookahead-fast", type=int, default=6)
    parser.add_argument("--max-linear-speed", type=float, default=0.8)
    parser.add_argument("--max-angular-speed", type=float, default=1.6)
    parser.add_argument("--cmd-filter-alpha", type=float, default=0.7)
    parser.add_argument("--max-linear-acc", type=float, default=0.8)
    parser.add_argument("--max-angular-acc", type=float, default=1.5)

    parser.add_argument(
        "--rgb-topic",
        type=str,
        default="/camera/camera/color/image_raw/compressed",
    )
    parser.add_argument(
        "--depth-topic",
        type=str,
        default=(
            "/camera/camera/aligned_depth_to_color/"
            "image_raw/compressedDepth"
        ),
    )
    parser.add_argument(
        "--camera-info-topic",
        type=str,
        default="/camera/camera/color/camera_info",
    )
    parser.add_argument("--odom-topic", type=str, default="/mfla/dr_odom")
    parser.add_argument("--tf-static-topic", type=str, default="/tf_static")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    rclpy.init()
    node = NavDPRos2BagRunner(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.control_output:
            node._publish_stop_cmd()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
