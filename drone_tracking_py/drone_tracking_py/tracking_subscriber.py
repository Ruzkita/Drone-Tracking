import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2 as cv
from sensor_msgs.msg import Image
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
                print(f"Cx {cx} | Cy {cy}")
                centers.append((cx, cy))
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                
        if not bounding_box or not bounding_box[0].boxes.xyxy.shape[0]:
            return None, None, annoted_frame

        
        return centers, bboxes, annoted_frame


class LukasKanade():
    def __init__(self):
        self.prev_gray = None
        self.prev_pts = None
        self.last_time = None
        self.center = None

        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

    def reset(self, frame, bboxes):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.prev_pts = self._detect_features(gray, bboxes)
        
        if self.prev_pts is not None:
            self.center = np.median(self.prev_pts.reshape(-1, 2), axis=0)

        self.last_time = time.time()

    
    def _detect_features(self, gray, bboxes):
        if bboxes is None:
            return
        x1, y1, x2, y2 = bboxes[0]

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
            return None, None, None, False

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        now = time.time()
        dt = now - self.last_time

        if dt <= 0:
            return None, None, None, False

        next_pts, status, _ = cv.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        if next_pts is None:
            self.prev_pts = None
            return None, None, None, False

        good_new = next_pts[status == 1]
        self.center = np.median(good_new, axis=0)
        good_old = self.prev_pts[status == 1]

        if len(good_new) < 5:
            self.prev_pts = None
            return None, None, None, False

        disp = good_new - good_old
        vel = disp / dt

        vx, vy = np.median(vel, axis=0)

        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)
        self.last_time = now

        return vx, vy, self.center, True


class FrameTransform():
    def __init__(self):
        self.fx, self.fy = 382.00628662109375, 382.00628662109375      
        self.cx, self.cy = 326.43426513671875, 239.53616333007812     
        
        theta = np.deg2rad(45)
        self.R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        self.t = [0, 0, 0]  #a definir
    
    def pixel_to_body_position(self, px, py, depth):
        if depth <= 0:
            return None

        Xc = (px - self.cx) * depth / self.fx
        Yc = (py - self.cy) * depth / self.fy
        Zc = depth

        p_c = np.array([Xc, Yc, Zc])
        p_b = self.R @ p_c + self.t

        return p_b
    
    
    def body_to_pixel(self, p_b):
        p_b = np.array(p_b).reshape(3)

        # volta para frame da câmera
        p_c = self.R.T @ (p_b - self.t)

        Xc, Yc, Zc = p_c

        if Zc <= 0:
            return None

        u = self.fx * Xc / Zc + self.cx
        v = self.fy * Yc / Zc + self.cy

        return int(u), int(v)
    
    
    def pixel_to_body_velocity(self, vpx, vpy, depth):
        if depth <= 0:
            return None
        
        Vxc = depth * vpx / self.fx
        Vyc = depth * vpy / self.fy
        Vzc = 0.0

        v_c = np.array([Vxc, Vyc, Vzc])
        v_b = self.R @ v_c

        return v_b



class KalmanFilter:
    def __init__(self):
        self.x = np.zeros((6, 1))                                   #px, py, z, vx, vy, vz -> variaveis 
        self.P = np.eye(6) * 1000                                   #matriz diagonal com valor 1000 -> incerteza
        self.Q = np.eye(6) * 0.1                                    #matriz de covariancia do modelo -> representa a covariancia dos ruidos e perturbações do processo (modelo)
        #self.R = np.eye(3) * 5.0                                    #matriz de covariancia do sensor -> diz o quanto vc confia na medição do sensor

    
    def algorithm(self, H, z, dt, R):
        self.F = np.array([                                         #matriz de transição de estados
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        x_next = self.F @ self.x
        P_next = self.F @ self.P @ self.F.T + self.Q

        K = P_next @ H.T @ np.linalg.inv(H @ P_next @ H.T + R)
        self.x = x_next + K @ (z - H @ x_next)   
        self.P = (np.eye(6) - K @ H) @ P_next

        #print(
        #    f"Pos: ({self.x[0,0]:.2f}, {self.x[1,0]:.2f}, {self.x[2,0]:.2f}) | "
        #    f"Vel: ({self.x[3,0]:.2f}, {self.x[4,0]:.2f}, {self.x[5,0]:.2f})"
        #)
        return self.x


class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.yolo_detection = YoloDetection()
        self.kalman = KalmanFilter()
        self.lk = LukasKanade()
        self.frame_transform = FrameTransform()
        self.bridge = CvBridge()

        #CONFIGURAÇÃO DA CÂMERA
        self.SHOW_WINDOW = False
        self.FPS = 30
        self.FRAME_SIZE = (640, 640)

        #VARIÁVEIS DE CONTROLE
        self.i = 0
        self.depth_frame = None
        self.last_time = None

        #QOS
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos2 = QoSProfile(reliability = ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)

        #SUBSCRIBERS
        self.camera_subscription = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, qos)
        self.depth_subscription = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, qos)

        #PUBLISHERS
        self.annotated_frame_publisher = self.create_publisher(Image, '/image/annotated_frame', qos2)
    
    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    def frame_publish(self, frame, state):
        if frame is None:
            return

        if state is not None:
            # pega posição (3x1 → vetor 3)
            p = state[:3].reshape(3)

            pixel = self.frame_transform.body_to_pixel(p)

            if pixel is not None:
                px, py = pixel

                cv.circle(frame, (px, py), 6, (0, 0, 255), -1)

                # também desenhar texto
                text = f"X:{p[0]:.2f} Y:{p[1]:.2f} Z:{p[2]:.2f}"
                cv.putText(
                    frame,
                    text,
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA
                )

        # converter para mensagem ROS
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        print("olá")
        # publicar
        self.annotated_frame_publisher.publish(msg)

    def image_callback(self, msg):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            return
        dt = now - self.last_time
        self.last_time = now
        self.state = None

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        def distance(x, y):
            h, w = self.depth_frame.shape

            x = int(x)
            y = int(y)

            if 2 <= x < w-2 and 2 <= y < h-2:
                roi = self.depth_frame[y-2:y+3, x-2:x+3]
                valid = roi[roi > 0]

                if len(valid) > 0:
                    return np.median(valid) / 1000.0
            return None

        if self.i % 5 == 0:
            centers, bboxes, annoted_frame = self.yolo_detection.bb_centers(frame)

            if bboxes is not None:
                #print("oi")
                self.lk.reset(frame, bboxes)
            
            if centers is not None:
                H = np.array([                                         #matriz de observação -> define oq vc realmente consegue medir
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
                ])

                R = np.eye(3) * 0.01

                x, y = centers[0]
                d = distance(int(x), int(y))

                if d is not None:
                    z = self.frame_transform.pixel_to_body_position(x, y, d)
                    z = z.reshape(3, 1)
                    self.state = self.kalman.algorithm(H, z, dt, R)

        else:
            centers, bboxes, _ = None, None, None

        vx, vy, centroid, _ = self.lk.update(frame)

        if centroid is not None:
            cx = centroid[0]
            cy = centroid[1]
            d = distance(int(cx), int(cy))
        
        else:
            d = None

        if vx is not None and vy is not None and d is not None:
            H = np.array([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0]
            ])
            R = np.eye(2) * 0.1

            z = self.frame_transform.pixel_to_body_velocity(vx, vy, d)
            z = z[:2].reshape(2, 1)
            self.state = self.kalman.algorithm(H, z, dt, R)
        
        if self.state is not None:
            print("oi")
            self.frame_publish(frame, self.state)

        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    tracking_subscriber = TrackerNode()
    rclpy.spin(tracking_subscriber)
    tracking_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "main":
    main()