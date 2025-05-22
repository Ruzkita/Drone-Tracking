import rclpy
from rclpy.node import Node
import cv2 as cv
from ultralytics import YOLO
import os


class Detection(Node):
    def __init__(self):
        super().__init__('detection')

        # Inicializa câmera
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FPS, 60)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        fps = self.cap.get(cv.CAP_PROP_FPS)
        self.get_logger().info(f"FPS configurado: {fps}")

        # Carrega o modelo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "..", "net_train", "weights", "best.pt")
        model_path = os.path.abspath(model_path)
        self.model = YOLO(model_path)

        # Cria janela única
        cv.namedWindow("Detecção YOLO", cv.WINDOW_NORMAL)
        self.capture_callback()

    def capture_callback(self):
        ret, frame = self.cap.read()
        frame_count = 0
        annotated_frame = None
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Erro ao capturar frame")
                break

            frame_count += 1

            if frame_count % 3 == 0:
                results = self.model(frame, conf=0.5)
                annotated_frame = results[0].plot()

            frame_to_show = annotated_frame if annotated_frame is not None else frame

            # Mostra o frame capturado
            cv.imshow('Detecção YOLO', frame_to_show)

            # Sai do loop se a tecla 'q' for pressionada
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Libera a câmera e fecha as janelas
        self.cap.release()
        cv.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = Detection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()