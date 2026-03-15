import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2 as cv
import pyrealsense2 as rs
import numpy as np

from sensor_msgs.msg import Image
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

        # Reduzido para 640x360 e 15 FPS, mais confiável no Raspberry Pi
        config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 15)

        self.pipeline.start(config)

        # Bridge ROS <-> OpenCV
        self.bridge = CvBridge()

        # Publishers
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', qos)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', qos)

        # Timer 30 FPS (o callback já trata a RealSense mais lenta)
        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.capture_callback)

        self.get_logger().info("RealSense node iniciado")

    def capture_callback(self):
        try:
            # Timeout maior para Pi (500ms)
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

        # Converte para ROS Image
        color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        depth_msg = self.bridge.cv2_to_imgmsg(depth_colormap, encoding='bgr8')

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