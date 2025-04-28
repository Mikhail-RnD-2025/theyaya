from recorder import CameraRecorder
from config_schema import CameraConfig, BitrateMode
import time

# Пример конфигурации камеры
config = CameraConfig(
    camera_name="front_camera",
    rtsp_url="rtsp://admin:password@192.168.1.100:554/stream",
    archive_dir="./video_archive",
    segment_duration=300,  # 5 минут
    timeout=10,
    bitrate_mode=BitrateMode.DYNAMIC,
    initial_bitrate=4000
)

# Создание и запуск рекордера
recorder = CameraRecorder(config)
recorder.start()

try:
    # Работа в течение 1 минуты для примера
    time.sleep(60)
finally:
    # Остановка записи
    recorder.stop()