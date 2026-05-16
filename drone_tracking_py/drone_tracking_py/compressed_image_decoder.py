import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import cv2 as cv
import time

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class CompressedImageDecoder(Node):

    def __init__(self):
        super().__init__('compressed_image_decoder')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            CompressedImage,
            '/camera/color/compressed',
            self.callback,
            qos
        )

        self.pub = self.create_publisher(
            Image,
            '/camera/color/image_raw',
            qos
        )

        self.get_logger().info("Decoder de imagem comprimida iniciado")

    def callback(self, msg):

        t0 = time.time()

        # bytes → numpy
        np_arr = np.frombuffer(msg.data, np.uint8)

        t1 = time.time()

        # decode JPEG
        image = cv.imdecode(np_arr, cv.IMREAD_COLOR)

        t2 = time.time()

        if image is None:
            return

        # bridge
        ros_img = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        ros_img.header = msg.header

        t3 = time.time()

        # publish
        self.pub.publish(ros_img)

        t4 = time.time()

        #print(f"""
#frombuffer: {t1-t0:.4f}
#decode:     {t2-t1:.4f}
#bridge:     {t3-t2:.4f}
#publish:    {t4-t3:.4f}
#total:      {t4-t0:.4f}
#------------------------
#""")


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