from enum import Enum
from pydantic import BaseModel, validator
from typing import List, Dict
import json
from pathlib import Path
import os
import threading  # Добавлен этот импорт
import time      # Добавлен для работы с time.sleep

class StreamPriority(str, Enum):
    ARCHIVE = "archive"
    LIVE = "live"

class RTSPTransport(str, Enum):
    TCP = "tcp"
    UDP = "udp"
    AUTO = "auto"

class BitrateMode(str, Enum):
    CONSTANT = "constant"
    VARIABLE = "variable"
    DYNAMIC = "dynamic"
    UNKNOWN = "unknown"

class CameraConfig(BaseModel):
    rtsp_url: str
    camera_name: str
    stream_priority: StreamPriority = StreamPriority.ARCHIVE
    rtsp_transport: RTSPTransport = RTSPTransport.TCP
    bitrate_mode: BitrateMode = BitrateMode.UNKNOWN
    initial_bitrate: int = 4000
    timeout: int = 10
    buffer_size: int = 1048576
    max_reconnect_attempts: int = 5
    archive_dir: str = "archive"
    segment_duration: int = 300

    @validator('initial_bitrate')
    def validate_bitrate(cls, v):
        if v < 500 or v > 20000:
            raise ValueError('Bitrate must be between 500 and 20000 kbps')
        return v

    @classmethod
    def load_configs(cls, file_path: str) -> List['CameraConfig']:
        """Load configuration from JSON file"""
        config_path = Path(file_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            raw_configs = json.load(f)

        # Create archive directories
        for config in raw_configs:
            archive_dir = config.get('archive_dir', 'archive')
            camera_dir = os.path.join(archive_dir, config['camera_name'])
            os.makedirs(camera_dir, exist_ok=True)

        return [cls(**config) for config in raw_configs]

    def get_rtsp_options(self) -> Dict[str, str]:
        """Get RTSP connection options"""
        return {
            'rtsp_transport': self.rtsp_transport.value,
            'buffer_size': str(self.buffer_size),
            'stimeout': str(self.timeout * 1000000),
            'max_delay': '500000',
            'fflags': 'nobuffer',
            'flags': 'low_delay',
            'analyzeduration': '1000000',
            'probesize': '1000000'
        }