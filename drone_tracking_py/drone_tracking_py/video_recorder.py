import rosbag2_py
import cv2 as cv
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import os

def extract_video_from_bag(bag_path, target_topic, output_video):
    # 1. Configuração do Reader
    reader = rosbag2_py.SequentialReader()
    
    # Verifica se é pasta (db3) ou arquivo (mcap)
    storage_id = 'mcap' if bag_path.endswith('.mcap') else 'sqlite3'
    
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    bridge = CvBridge()
    video_writer = None
    count = 0
    
    print(f"🚀 Iniciando extração do tópico: {target_topic}")

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        # 🔥 Filtro de Tópico: Só processa se for o que você pediu
        if topic == target_topic:
            msg = deserialize_message(data, Image)
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Inicializa o arquivo de vídeo no primeiro frame recebido
            if video_writer is None:
                h, w, _ = frame.shape
                # Salvando em .avi com codec XVID
                fourcc = cv.VideoWriter_fourcc(*'XVID')
                video_writer = cv.VideoWriter(output_video, fourcc, 30.0, (w, h))
                print(f"🎬 Formato definido: {w}x{h} @ 30 FPS")

            video_writer.write(frame)
            count += 1
            if count % 100 == 0:
                print(f"📦 {count} frames processados...")

    if video_writer:
        video_writer.release()
        print(f"✅ Sucesso! Vídeo salvo em: {output_video}")
    else:
        print(f"❌ Erro: O tópico '{target_topic}' não foi encontrado no bag ou não contém imagens.")

# ==========================================
# EXECUTAR AQUI
# ==========================================
if __name__ == "__main__":
    # COLOQUE OS SEUS NOMES AQUI:
    MEU_BAG = 'rosbag.db3'
    MEU_TOPICO = '/image/annotated_frame'
    ARQUIVO_SAIDA = 'video_extraido.avi'

    extract_video_from_bag(MEU_BAG, MEU_TOPICO, ARQUIVO_SAIDA)