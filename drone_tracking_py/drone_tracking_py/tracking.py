import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor

import cv2 as cv
import pyrealsense2 as rs
import numpy as np

from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge


class CameraNode(Node):

    def __init__(self):
        super().__init__('realsense_camera')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- RealSense ---
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = self.pipeline.start(config)

        # Intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        depth_stream = profile.get_stream(rs.stream.depth)

        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        # Bridge
        self.bridge = CvBridge()

        # Publishers
        self.color_pub = self.create_publisher(
            CompressedImage,
            '/camera/color/compressed',
            qos
        )

        self.depth_pub = self.create_publisher(
            Image,
            '/camera/depth/image_raw',
            qos
        )

        self.color_info_pub = self.create_publisher(
            CameraInfo,
            '/camera/color/camera_info',
            qos
        )

        self.depth_info_pub = self.create_publisher(
            CameraInfo,
            '/camera/depth/camera_info',
            qos
        )

        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.capture_callback)

        self.get_logger().info("RealSense node iniciado")

    def create_camera_info(self, intrinsics, frame_id):

        msg = CameraInfo()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.width = intrinsics.width
        msg.height = intrinsics.height

        msg.k = [
            intrinsics.fx, 0.0, intrinsics.ppx,
            0.0, intrinsics.fy, intrinsics.ppy,
            0.0, 0.0, 1.0
        ]

        msg.p = [
            intrinsics.fx, 0.0, intrinsics.ppx, 0.0,
            0.0, intrinsics.fy, intrinsics.ppy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        msg.d = list(intrinsics.coeffs)

        msg.distortion_model = "plumb_bob"

        return msg

    def capture_callback(self):

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            return

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        timestamp = self.get_clock().now().to_msg()

        # ---------- RGB JPEG ----------
        success, buffer = cv.imencode(
            ".jpg",
            color_image,
            [int(cv.IMWRITE_JPEG_QUALITY), 80]
        )

        if success:

            color_msg = CompressedImage()

            color_msg.header.stamp = timestamp
            color_msg.header.frame_id = "camera_color"

            color_msg.format = "jpeg"
            color_msg.data = buffer.tobytes()

            self.color_pub.publish(color_msg)

        # ---------- DEPTH REAL ----------
        depth_msg = self.bridge.cv2_to_imgmsg(
            depth_image,
            encoding="16UC1"
        )

        depth_msg.header.stamp = timestamp
        depth_msg.header.frame_id = "camera_depth"

        self.depth_pub.publish(depth_msg)

        # ---------- CAMERA INFO ----------
        color_info = self.create_camera_info(
            self.color_intrinsics,
            "camera_color"
        )

        depth_info = self.create_camera_info(
            self.depth_intrinsics,
            "camera_depth"
        )

        color_info.header.stamp = timestamp
        depth_info.header.stamp = timestamp

        self.color_info_pub.publish(color_info)
        self.depth_info_pub.publish(depth_info)

    def destroy_node(self):

        self.pipeline.stop()
        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = CameraNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()