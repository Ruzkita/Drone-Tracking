import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import cv2 as cv

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class CompressedImageDecoder(Node):

    def __init__(self):
        super().__init__('compressed_image_decoder')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.bridge = CvBridge()

        # subscriber
        self.sub = self.create_subscription(
            CompressedImage,
            '/camera/color/compressed',
            self.callback,
            qos
        )

        # publisher
        self.pub = self.create_publisher(
            Image,
            '/camera/color/image_raw',
            qos
        )

        self.get_logger().info("Decoder de imagem comprimida iniciado")

    def callback(self, msg):

        # converte bytes → numpy
        np_arr = np.frombuffer(msg.data, np.uint8)

        # decodifica JPEG
        image = cv.imdecode(np_arr, cv.IMREAD_COLOR)

        if image is None:
            return

        # converte para ROS Image
        ros_img = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")

        ros_img.header = msg.header

        # publica
        self.pub.publish(ros_img)


def main(args=None):

    rclpy.init(args=args)

    node = CompressedImageDecoder()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()