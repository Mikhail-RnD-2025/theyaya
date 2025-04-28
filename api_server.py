from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading
import json
import shutil
import os
import time
from typing import List
from enum import Enum
import logging
from datetime import datetime
import psutil
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Surveillance API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data models
class BitrateMode(str, Enum):
    CONSTANT = "constant"
    VARIABLE = "variable"
    DYNAMIC = "dynamic"
    UNKNOWN = "unknown"


class CameraConfigModel(BaseModel):
    rtsp_url: str
    camera_name: str
    stream_priority: str
    rtsp_transport: str
    bitrate_mode: BitrateMode
    initial_bitrate: int
    timeout: int
    segment_duration: int
    archive_dir: str


class SystemStatusModel(BaseModel):
    running: bool
    cameras: List[dict]
    cpu_usage: float
    memory_usage: float
    timestamp: datetime


# Global state (in production use Redis or database)
system_state = {
    "running": False,
    "recorders": {},
    "config_file": "camera_config.json"
}


def load_recorders():
    """Load recorders from config file"""
    from config_schema import CameraConfig
    from recorder import CameraRecorder

    try:
        configs = CameraConfig.load_configs(system_state["config_file"])
        system_state["recorders"] = {
            config.camera_name: CameraRecorder(config)
            for config in configs
            if config.stream_priority == "archive"
        }
        logger.info(f"Loaded {len(system_state['recorders'])} recorders from config")
    except Exception as e:
        logger.error(f"Failed to load recorders: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    try:
        load_recorders()
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")


@app.post("/cameras/add", status_code=201)
async def add_camera(config: CameraConfigModel):
    """Add new camera"""
    try:
        # In production: validate and save to config file
        return {"message": "Camera added (not implemented)"}
    except Exception as e:
        logger.error(f"Error adding camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/status", response_model=SystemStatusModel)
async def get_status():
    """Get system status"""
    try:
        cameras = []
        for name, recorder in system_state["recorders"].items():
            cameras.append({
                "name": name,
                "status": "running" if recorder.running else "stopped",
                "recording": recorder.current_segment_path if recorder.running else None
            })

        return {
            "running": system_state["running"],
            "cameras": cameras,
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system/start")
async def start_system():
    """Start recording system"""
    try:
        if system_state["running"]:
            return {"message": "System already running"}

        for recorder in system_state["recorders"].values():
            recorder.start()

        system_state["running"] = True
        return {"message": "System started"}
    except Exception as e:
        logger.error(f"Start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system/stop")
async def stop_system():
    """Stop recording system"""
    try:
        if not system_state["running"]:
            return {"message": "System already stopped"}

        for recorder in system_state["recorders"].values():
            recorder.stop()

        system_state["running"] = False
        return {"message": "System stopped"}
    except Exception as e:
        logger.error(f"Stop error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/update")
async def update_config(file: UploadFile = File(...)):
    """Update configuration file"""
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{int(time.time())}.json"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate new config
        from config_schema import CameraConfig
        CameraConfig.load_configs(temp_path)

        # Backup current config
        backup_path = f"{system_state['config_file']}.bak"
        shutil.copyfile(system_state['config_file'], backup_path)

        # Replace config
        shutil.move(temp_path, system_state['config_file'])

        # Reload configuration
        load_recorders()

        return {"message": "Config updated successfully"}
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Config update failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid configuration: {str(e)}"
        )
    finally:
        if hasattr(file.file, 'close'):
            file.file.close()


@app.get("/config/current")
async def get_current_config():
    """Get current configuration"""
    try:
        with open(system_state["config_file"], "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Config read error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run API server"""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)

    api_thread = threading.Thread(
        target=server.run,
        daemon=True
    )
    api_thread.start()
    logger.info(f"API server started on http://{host}:{port}")
    return api_thread


if __name__ == "__main__":
    run_api_server()
    while True:
        time.sleep(1)