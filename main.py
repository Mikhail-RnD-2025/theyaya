import threading
import time
import logging
from recorder import CameraRecorder
from config_schema import CameraConfig
from api_server import run_api_server
from web_interface import run_web_interface


def web_main():
    # Инициализация системы
    system = VideoSurveillanceSystem()

    # Запуск веб-интерфейса
    web_thread = run_web_interface()

    try:
        system.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        system.stop()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        system.stop()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('surveillance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VideoSurveillanceSystem:
    def __init__(self, config_file: str = "camera_config.json"):
        self.config_file = config_file
        self.recorders = {}
        self.running = False
        self.load_config()

    def load_config(self):
        """Load configuration from file"""
        try:
            configs = CameraConfig.load_configs(self.config_file)
            self.recorders = {
                config.camera_name: CameraRecorder(config)
                for config in configs
                if config.stream_priority == "archive"
            }
            logger.info(f"Loaded {len(self.recorders)} recorders from config")
        except Exception as e:
            logger.critical(f"Failed to load config: {str(e)}")
            raise

    def start(self):
        """Start all recorders"""
        if self.running:
            return

        for recorder in self.recorders.values():
            recorder.start()

        self.running = True
        logger.info("Surveillance system started")

    def stop(self):
        """Stop all recorders"""
        if not self.running:
            return

        for recorder in self.recorders.values():
            recorder.stop()

        self.running = False
        logger.info("Surveillance system stopped")


if __name__ == "__main__":
    web_main()
    system = None  # Инициализация переменной заранее
    try:
        # Initialize system
        system = VideoSurveillanceSystem()

        # Start API server
        api_thread = run_api_server()

        # Start recording
        system.start()

        # Main loop
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        if system:  # Проверка на существование объекта
            system.stop()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        if system:  # Проверка на существование объекта
            system.stop()