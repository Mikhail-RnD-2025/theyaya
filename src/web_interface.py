from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import json
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Surveillance Web Interface")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка статических файлов и шаблонов
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Глобальное состояние
system_state = {
    "running": False,
    "cameras": {},
    "config_file": "camera_config.json"
}


class CameraStatus:
    def __init__(self, name: str, rtsp_url: str, status: str = "stopped"):
        self.name = name
        self.rtsp_url = rtsp_url
        self.status = status
        self.last_update = datetime.now()
        self.frames_processed = 0


def is_authenticated(request: Request) -> bool:
    """Проверка аутентификации (заглушка)"""
    # В реальной системе реализуйте проверку токена или сессии
    return True


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    try:
        load_camera_config()
        logger.info("Web interface initialized")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")


def load_camera_config():
    """Загрузка конфигурации камер с обработкой ошибок"""
    try:
        config_path = Path(system_state["config_file"])
        if not config_path.exists():
            logger.warning("Config file not found, using empty configuration")
            system_state["cameras"] = {}
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            cameras = json.load(f)

        if not isinstance(cameras, list):
            raise ValueError("Config must contain a list of cameras")

        system_state["cameras"] = {
            cam["camera_name"]: CameraStatus(
                name=cam["camera_name"],
                rtsp_url=cam.get("rtsp_url", ""),
                status="stopped"
            )
            for cam in cameras if "camera_name" in cam
        }
        logger.info(f"Loaded {len(system_state['cameras'])} cameras from config")
    except Exception as e:
        logger.error(f"Config load error: {str(e)}")
        system_state["cameras"] = {}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Главная панель управления"""
    cameras = []
    for cam in system_state["cameras"].values():
        cameras.append({
            "name": cam.name,
            "status": cam.status,
            "rtsp_url": cam.rtsp_url,
            "last_update": cam.last_update.strftime("%Y-%m-%d %H:%M:%S") if cam.last_update else "N/A",
            "frames_processed": cam.frames_processed
        })

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "cameras": cameras,
            "system_status": "running" if system_state["running"] else "stopped"
        }
    )


@app.post("/api/system/start")
async def start_system(request: Request):
    """API для запуска системы с проверкой авторизации"""
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    system_state["running"] = True
    return JSONResponse(
        content={"status": "success", "message": "System started"},
        status_code=200
    )


def run_web_interface(host: str = "0.0.0.0", port: int = 9000):
    """Запуск веб-интерфейса с обработкой ошибок"""
    try:
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            timeout_keep_alive=60
        )
        server = uvicorn.Server(config)

        web_thread = threading.Thread(
            target=server.run,
            daemon=True,
            name="WebInterface"
        )
        web_thread.start()
        logger.info(f"Web interface started on http://{host}:{port}")
        return web_thread
    except Exception as e:
        logger.critical(f"Failed to start web interface: {str(e)}")
        raise


def create_template_files():
    """Создание базовых файлов шаблонов и статики, если их нет"""
    # Создаем CSS файл
    css_path = Path("static/css/style.css")
    if not css_path.exists():
        css_path.parent.mkdir(parents=True, exist_ok=True)
        with open(css_path, "w") as f:
            f.write("""
            /* Basic styles will be auto-generated here */
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
            .camera-card { border: 1px solid #ddd; padding: 10px; margin: 10px; }
            .status-running { color: green; }
            .status-stopped { color: red; }
            """)

    # Создаем JS файл
    js_path = Path("static/js/app.js")
    if not js_path.exists():
        js_path.parent.mkdir(parents=True, exist_ok=True)
        with open(js_path, "w") as f:
            f.write("// JavaScript for the web interface")

    # Создаем шаблоны
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    if not (templates_dir / "dashboard.html").exists():
        with open(templates_dir / "dashboard.html", "w") as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Video Surveillance</title>
                <link rel="stylesheet" href="/static/css/style.css">
            </head>
            <body>
                <h1>Video Surveillance System</h1>
                <!-- Content will be auto-generated -->
            </body>
            </html>
            """)

    if not (templates_dir / "camera.html").exists():
        with open(templates_dir / "camera.html", "w") as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Camera View</title>
                <link rel="stylesheet" href="/static/css/style.css">
            </head>
            <body>
                <h1>Camera Detail View</h1>
                <!-- Content will be auto-generated -->
            </body>
            </html>
            """)

if __name__ == "__main__":
    # Создание необходимых директорий
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    # Создание базовых файлов, если их нет
    create_template_files()

    # Запуск интерфейса
    try:
        run_web_interface()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Web interface stopped")
    except Exception as e:
        logger.error(f"Web interface error: {str(e)}")

