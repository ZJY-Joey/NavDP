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

            if self.print_server_json:
                try:
                    parsed = json.loads(body)
                    traj = np.array(parsed.get("trajectory", []))
                    all_values = np.array(parsed.get("all_values", []))
                    self.get_logger().info(
                        "server response: "
                        "trajectory_shape="
                        f"{traj.shape}, all_values_shape={all_values.shape}"
                    )
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
