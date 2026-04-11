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
import json
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import rclpy
from requests import exceptions as req_exc
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

        self.rgb_topic = args.rgb_topic
        self.depth_topic = args.depth_topic
        self.camera_info_topic = args.camera_info_topic
        self.odom_topic = args.odom_topic
        self.tf_static_topic = args.tf_static_topic

        self.last_rgb: Optional[np.ndarray] = None
        self.last_depth_m: Optional[np.ndarray] = None
        self.last_camera_info: Optional[CameraInfo] = None
        self.last_odom: Optional[Odometry] = None
        self.last_tf_static: Optional[TFMessage] = None

        self.last_send_time = 0.0
        self.last_health_log_time = 0.0
        self._reset_done = False
        self._vis_window_name = "NavDP HTTP Trajectory"
        self._vis_enabled = self.vis_http_ref
        self._lock = threading.Lock()

        self._session = requests.Session()

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
            self.last_tf_static = msg

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

        pts_xy = self._extract_traj_points(trajectory)
        if pts_xy is None:
            return

        try:
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
            tf_static = self.last_tf_static

        if rgb is None or depth_m is None:
            self.get_logger().info("Waiting for RGB + depth frames...")
            return

        now = time.time()
        if now - self.last_send_time < 1.0 / max(self.send_hz, 1e-3):
            return

        if odom is None:
            self.get_logger().warn("/mfla/dr_odom not received yet")
        if tf_static is None:
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

            if self.print_server_json or self._vis_enabled:
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
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
