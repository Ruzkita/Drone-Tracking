import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2 as cv
import pyrealsense2 as rs
import numpy as np

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


class CameraNode(Node):

    def __init__(self):
        super().__init__('realsense_camera')

        # QoS confiável para imagens
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- RealSense pipeline ---
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Resolução menor e FPS baixo para Raspberry Pi
        config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 15)

        self.pipeline.start(config)

        # Bridge ROS <-> OpenCV
        self.bridge = CvBridge()

        # Publishers usando CompressedImage
        self.color_pub = self.create_publisher(CompressedImage, '/camera/color/compressed', qos)
        self.depth_pub = self.create_publisher(CompressedImage, '/camera/depth/compressed', qos)

        # Timer ~30Hz
        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.capture_callback)

        self.get_logger().info("RealSense node iniciado (CompressedImage)")

    def capture_callback(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)
        except RuntimeError:
            self.get_logger().warn("Nenhum frame recebido no tempo limite")
            return

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        # Converte para numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normaliza depth para visualização
        depth_colormap = cv.applyColorMap(
            cv.convertScaleAbs(depth_image, alpha=0.03),
            cv.COLORMAP_JET
        )

        # Converte para ROS CompressedImage
        color_msg = self.bridge.cv2_to_compressed_imgmsg(color_image, dst_format='jpeg')
        color_msg.header.frame_id = "camera_color"

        depth_msg = self.bridge.cv2_to_compressed_imgmsg(depth_colormap, dst_format='jpeg')
        depth_msg.header.frame_id = "camera_depth"

        # Publica
        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()