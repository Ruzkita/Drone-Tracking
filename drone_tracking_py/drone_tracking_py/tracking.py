import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2 as cv
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import os
from ultralytics import YOLO
from ament_index_python import get_package_share_directory
import numpy as np
#from drone_tracking.run_ncnn import NCNNRunner


class Detection(Node):
    def __init__(self):
        super().__init__('detection')

        #Definição de QoS pra reduzir delay na transmissão da câmera
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # Inicializa câmera
        self.cap = cv.VideoCapture(0, cv.CAP_V4L2)
        self.cap.set(cv.CAP_PROP_FPS, 30)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)


        fps = self.cap.get(cv.CAP_PROP_FPS)
        self.get_logger().info(f"FPS configurado na câmera: {fps}")

        #Publisher de imagem
        self.camera_publishing = self.create_publisher(CompressedImage, '/camera/compressed', qos)
        self.bridge = CvBridge()

        #Variáveis de controle
        self.frame_c = 0
        self.annoted_frame = None

        #Timer para 30 FPS
        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.capture_callback)

    def capture_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Erro ao capturar frame da câmera")
            return

        #Converte para imagem comprimida (JPEG)
        ros_compressed_image = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpeg')
        self.camera_publishing.publish(ros_compressed_image)

    def destroy_node(self):
        # Libera recursos ao encerrar
        self.cap.release()
        #cv.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Detection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
