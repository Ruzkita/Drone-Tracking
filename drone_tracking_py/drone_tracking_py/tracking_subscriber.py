import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2 as cv
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
from ament_index_python import get_package_share_directory
import os
import time


class YoloDetection:
    def __init__(self):
        package_dir = get_package_share_directory('drone_tracking_py')
        model_path = os.path.join(package_dir, 'net_train', 'weights', 'best.pt')
        self.model = YOLO(model_path)
    
    def detection(self, img):
        results = self.model(img, verbose=False, conf=0.5)
        annoted_frame = np.array(results[0].plot())
        return results, annoted_frame
    
    def bb_centers(self, img):
        bounding_box, annoted_frame = self.detection(img)

        centers = []
        bboxes = []

        if bounding_box is not None:
            boxes = bounding_box[0].boxes.xyxy
            for box in boxes:
                x1, y1, x2, y2 = box.tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centers.append((cx, cy))
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                
        if not bounding_box or not bounding_box[0].boxes.xyxy.shape[0]:
            return [], annoted_frame

        
        return centers, bboxes, annoted_frame


class LukasKanade():
    def __init__(self):
        self.prev_gray = None
        self.prev_pts = None
        self.last_time = None

        self.lk_params = dict(WinSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

    def reset(self, frame, bboxes):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.prev_pts = self._detect_features(gray, bboxes)
        self.last_time = time.time()

    
    def _detect_features(self, gray, bboxes):
        x1, x2, y1, y2 = bboxes

        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255

        pts = cv.goodFeaturesToTrack(
            gray,
            mask=mask,
            maxCorners=50,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        ) 

        return pts
    
    def update(self, frame):
        if self.prev_gray is None or self.prev_pts is None:
            return None, None, False

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        now = time.time()
        dt = now - self.last_time

        if dt <= 0:
            return None, None, False

        next_pts, status, _ = cv.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        if next_pts is None:
            self.prev_pts = None
            return None, None, False

        good_new = next_pts[status == 1]
        good_old = self.prev_pts[status == 1]

        if len(good_new) < 5:
            self.prev_pts = None
            return None, None, False

        disp = good_new - good_old
        vel = disp / dt

        vx, vy = np.median(vel, axis=0)

        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)
        self.last_time = now

        return vx, vy, True


class KalmanFilter:
    def __init__(self, dt):
        self.x = np.zeros((6, 1))                                   #px, py, h, vx, vy, vh, ax, ay -> variaveis 
        self.P = np.eye(6) * 1000                                   #matriz diagonal com valor 1000 -> incerteza
        self.dt = dt                                                #intervalo de tempo
        self.Q = np.eye(6) * 0.1                                    #matriz de covariancia do modelo -> representa a covariancia dos ruidos e perturbações do processo (modelo)
        self.R = np.eye(3) * 5.0                                    #matriz de covariancia do sensor -> diz o quanto vc confia na medição do sensor

        self.F = np.array([                                         #matriz de transição de estados
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

    
    def algorithm(self, H, z):
        x_next = self.F @ self.x
        P_next = self.F @ self.P @ self.F.T + self.Q

        K = P_next @ H.T @ np.linalg.inv(H @ P_next @ H.T + self.R)
        self.x = x_next + K @ (z - H @ x_next)   
        self.P = (np.eye(8) - K @ H) @ P_next

        return self.x


class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.yolo_detection = YoloDetection()
        self.kalman = KalmanFilter()
        self.lk = LukasKanade()

        #CONFIGURAÇÃO DA CÂMERA
        self.SHOW_WINDOW = False
        self.FPS = 30
        self.FRAME_SIZE = (640, 640)

        #VARIÁVEIS DE CONTROLE
        self.i = 0

        #QOS
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        #SUBSCRIBERS
        self.camera_subscription = self.create_subscription(CompressedImage, '/camera/compressed', self.image_callback, qos)
    

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)      
        frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)    #DECODIFICAR A IMAGEM COMPRIMIDA
        d = 0   #Enquanto eu n tenho a profundidade. Só para debug

        if self.i % 5 == 0:
            centers, bboxes, annoted_frame = self.yolo_detection.bb_centers(frame)

            if bboxes is not None:
                self.lk.reset(frame, bboxes)
            
            if centers is not None:
                H = np.array([                                         #matriz de observação -> define oq vc realmente consegue medir
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
                ])

                x, y = centers[0]
                z = [x, y, d]
                
                state = self.kalman.algorithm(H, z)

        else:
            centers, bboxes, annoted_frame = None, None, None
        
        vx, vy, _ = self.lk.update(frame)

        if vx is not None and vy is not None:
            H = np.array([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])

            state = self.kalman.algorithm(H, z)