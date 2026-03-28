import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import os
import time


class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')

        self.bridge = CvBridge()

        # ===== parâmetros =====
        self.topic_name = '/camera/color/image_raw'  
        self.output_path = 'output.avi'
        self.fps = 15.0

        # ===== writer =====
        self.video_writer = None
        self.frame_size = None

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # subscriber
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            qos
        )

        self.get_logger().info(f"Gravando vídeo de: {self.topic_name}")


    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # inicializa writer na primeira imagem
        if self.video_writer is None:
            height, width, _ = frame.shape
            self.frame_size = (width, height)

            fourcc = cv.VideoWriter_fourcc(*'MJPG')  
            self.video_writer = cv.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                self.frame_size
            )

            self.get_logger().info(
                f"Vídeo iniciado: {self.output_path} | {width}x{height} @ {self.fps} FPS"
            )

        # escreve frame
        self.video_writer.write(frame)


    def destroy_node(self):
        # garante que salva corretamente
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info("Vídeo salvo com sucesso.")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()