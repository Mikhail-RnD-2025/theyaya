import av
import psutil
import numpy as np
import logging
import time
import os
import threading
from datetime import datetime
from typing import Optional, Dict, Tuple
from config_schema import BitrateMode
from av.error import AVError, FFmpegError, HTTPNotFoundError, PermissionError, InvalidDataError

logger = logging.getLogger(__name__)


class BitrateAnalyzer:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.bitrates = []
        self.packet_sizes = []

    def _convert_to_float(self, value):
        """Convert Fraction or other types to float safely"""
        try:
            if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
                return float(value.numerator) / float(value.denominator)
            return float(value)
        except (TypeError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"Conversion error: {str(e)}")
            return 0.0

    def add_packet(self, packet: av.Packet) -> Optional[BitrateMode]:
        """Analyze packet and detect bitrate mode with enhanced error handling"""
        try:
            if packet.size == 0 or packet.duration is None or packet.time_base == 0:
                return None

            duration = self._convert_to_float(packet.duration)
            time_base = self._convert_to_float(packet.time_base)

            if duration <= 0 or time_base <= 0:
                return None

            bitrate = (packet.size * 8) / (duration * time_base)
            self.bitrates.append(bitrate)
            self.packet_sizes.append(packet.size)

            # Maintain sliding window
            if len(self.bitrates) > self.window_size * 2:
                self.bitrates = self.bitrates[-self.window_size:]
                self.packet_sizes = self.packet_sizes[-self.window_size:]

            if len(self.bitrates) < self.window_size:
                return None

            # Calculate statistics safely
            try:
                bitrates_array = np.array(self.bitrates[-self.window_size:], dtype=np.float64)
                sizes_array = np.array(self.packet_sizes[-self.window_size:], dtype=np.float64)

                bitrate_std = np.std(bitrates_array)
                size_std = np.std(sizes_array)

                if bitrate_std < 50000 and size_std < 500:
                    return BitrateMode.CONSTANT
                elif bitrate_std > 100000:
                    return BitrateMode.VARIABLE
            except Exception as calc_error:
                logger.warning(f"Bitrate calculation error: {str(calc_error)}")

        except Exception as e:
            logger.warning(f"Bitrate analysis error: {str(e)}", exc_info=True)
        return None


class BitrateController:
    def __init__(self, mode: BitrateMode = BitrateMode.UNKNOWN,
                 initial_bitrate: int = 4000,
                 min_bitrate: int = 500,
                 max_bitrate: int = 8000):
        self.mode = mode
        self.current_bitrate = initial_bitrate
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        self.last_adjustment = time.time()
        self.adjustment_interval = 10
        self.analyzer = BitrateAnalyzer()
        self.detection_complete = False

    def validate_bitrate(self, bitrate: int) -> int:
        """Ensure bitrate is within valid range"""
        try:
            bitrate = int(bitrate)
            return max(self.min_bitrate, min(self.max_bitrate, bitrate))
        except (TypeError, ValueError):
            return self.min_bitrate

    def detect_mode(self, packet: av.Packet) -> bool:
        """Detect stream bitrate mode with validation"""
        if self.detection_complete:
            return True

        try:
            detected_mode = self.analyzer.add_packet(packet)
            if detected_mode in [BitrateMode.CONSTANT, BitrateMode.VARIABLE]:
                self.mode = detected_mode
                self.detection_complete = True
                logger.info(f"Detected stream bitrate mode: {self.mode.name}")
                return True
        except Exception as e:
            logger.error(f"Bitrate detection failed: {str(e)}", exc_info=True)
        return False

    def adjust_bitrate(self) -> Tuple[int, Dict]:
        """Get current bitrate settings with enhanced validation"""
        try:
            self.current_bitrate = self.validate_bitrate(self.current_bitrate)

            effective_mode = self.mode if self.detection_complete else BitrateMode.DYNAMIC

            if effective_mode == BitrateMode.CONSTANT:
                options = {
                    'maxrate': f'{self.current_bitrate}k',
                    'bufsize': f'{self.current_bitrate * 2}k',
                    'nal-hrd': 'cbr'
                }
                return self.current_bitrate, options

            elif effective_mode == BitrateMode.VARIABLE:
                options = {
                    'crf': '23',
                    'maxrate': f'{self.max_bitrate}k',
                    'bufsize': f'{self.max_bitrate * 2}k'
                }
                return 0, options

            else:  # DYNAMIC or UNKNOWN
                if time.time() - self.last_adjustment >= self.adjustment_interval:
                    self._auto_adjust()

                options = {
                    'maxrate': f'{self.current_bitrate}k',
                    'bufsize': f'{self.current_bitrate * 2}k'
                }
                return self.current_bitrate, options
        except Exception as e:
            logger.error(f"Bitrate adjustment error: {str(e)}", exc_info=True)
            return self.min_bitrate, {}

    def _auto_adjust(self):
        """Auto-adjust bitrate based on system load with limits"""
        try:
            cpu_usage = psutil.cpu_percent()
            mem_usage = psutil.virtual_memory().percent

            if cpu_usage > 80 or mem_usage > 80:
                self.current_bitrate = max(self.current_bitrate * 0.8, self.min_bitrate)
            elif cpu_usage < 50 and mem_usage < 70 and self.current_bitrate < self.max_bitrate:
                self.current_bitrate = min(self.current_bitrate * 1.2, self.max_bitrate)

            self.current_bitrate = self.validate_bitrate(self.current_bitrate)
            self.last_adjustment = time.time()
            logger.info(
                f"Adjusted bitrate: {self.current_bitrate}kbps "
                f"(CPU: {cpu_usage}%, Mem: {mem_usage}%)"
            )
        except Exception as e:
            logger.error(f"Auto-adjust error: {str(e)}", exc_info=True)


class CameraRecorder:
    def __init__(self, config):
        self.config = config
        self.input_container = None
        self.output_container = None
        self.input_stream = None
        self.output_stream = None
        self.current_segment_start = None
        self.current_segment_path = None
        self.running = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.bitrate_controller = BitrateController(
            mode=config.bitrate_mode,
            initial_bitrate=config.initial_bitrate,
            min_bitrate=max(500, config.initial_bitrate // 2),
            max_bitrate=min(20000, config.initial_bitrate * 2)
        )
        self.last_packet_time = 0
        self.thread = None

    def _validate_archive_dir(self):
        """Validate archive directory exists and writable"""
        archive_path = os.path.abspath(self.config.archive_dir)
        try:
            os.makedirs(archive_path, exist_ok=True)
            test_file = os.path.join(archive_path, '.test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except Exception as e:
            logger.error(f"Archive directory error: {str(e)}")
            return False

    def _get_segment_filename(self) -> str:
        """Generate segment filename with proper path handling"""
        if not self._validate_archive_dir():
            raise IOError("Invalid archive directory")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.abspath(self.config.archive_dir)
        camera_dir = os.path.normpath(os.path.join(archive_path, self.config.camera_name))

        try:
            os.makedirs(camera_dir, exist_ok=True)
            filename = f"{self.config.camera_name}_{timestamp}.mp4"
            return os.path.normpath(os.path.join(camera_dir, filename))
        except Exception as e:
            logger.error(f"Directory creation error: {str(e)}")
            raise

    def _close_current_segment(self):
        """Properly close current recording segment with error handling"""
        if self.output_container:
            try:
                # Flush encoder if stream exists
                if self.output_stream:
                    try:
                        for packet in self.output_stream.encode():
                            try:
                                self.output_container.mux(packet)
                            except (ValueError, AVError) as e:
                                logger.error(f"Muxing error during close: {str(e)}")
                                break
                    except Exception as flush_error:
                        logger.error(f"Flush error: {str(flush_error)}")

                # Close container
                try:
                    self.output_container.close()
                    logger.info(f"Closed segment: {self.current_segment_path}")
                except Exception as close_error:
                    logger.error(f"Close error: {str(close_error)}")

            except Exception as e:
                logger.error(f"Segment close error: {str(e)}", exc_info=True)
            finally:
                self._cleanup_output()

        self.current_segment_path = None
        self.current_segment_start = None

    def _safe_open_stream(self) -> bool:
        """Safely open RTSP stream with comprehensive error handling"""
        try:
            options = {
                'rtsp_transport': 'tcp',
                'stimeout': str(self.config.timeout * 1000000),
                'max_delay': '500000',
                'rtsp_flags': 'prefer_tcp',
                'allowed_media_types': 'video'
            }

            logger.info(f"Connecting to {self.config.camera_name} ({self.config.rtsp_url})...")

            self.input_container = av.open(
                self.config.rtsp_url,
                timeout=(self.config.timeout, self.config.timeout * 2),
                options=options
            )

            # Find video stream
            self.input_stream = next(
                (s for s in self.input_container.streams if s.type == 'video'),
                None
            )

            if not self.input_stream:
                logger.error(f"No video stream found in {self.config.camera_name}")
                self._cleanup_input()
                return False

            # Test read first packet
            try:
                first_packet = next(self.input_container.demux(self.input_stream))
                if not first_packet:
                    raise InvalidDataError("Empty first packet")
            except Exception as e:
                logger.error(f"Camera test read failed: {str(e)}")
                self._cleanup_input()
                return False

            self.reconnect_attempts = 0
            self.last_packet_time = time.time()
            logger.info(f"Successfully connected to {self.config.camera_name}")
            return True

        except HTTPNotFoundError:
            logger.error(f"Camera {self.config.camera_name} not found (404)")
        except PermissionError:
            logger.error(f"Access denied to {self.config.camera_name} (check credentials)")
        except InvalidDataError as e:
            if "1414092869" in str(e):  # Immediate exit requested
                logger.error(f"Camera {self.config.camera_name} unavailable (immediate exit)")
            else:
                logger.error(f"Invalid data from {self.config.camera_name}: {str(e)}")
        except FFmpegError as e:
            logger.error(f"FFmpeg error: {str(e)}")
        except AVError as e:
            logger.error(f"AV error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected connection error: {str(e)}", exc_info=True)

        self._cleanup_input()
        return False

    def _cleanup_input(self):
        """Cleanup input resources safely"""
        try:
            if self.input_container:
                try:
                    self.input_container.close()
                except Exception as e:
                    logger.debug(f"Input close error: {str(e)}")
        finally:
            self.input_container = None
            self.input_stream = None

    def _cleanup_output(self):
        """Cleanup output resources safely"""
        try:
            if self.output_container:
                try:
                    self.output_container.close()
                except Exception as e:
                    logger.debug(f"Output close error: {str(e)}")
        finally:
            self.output_container = None
            self.output_stream = None

    def _reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff and max attempts"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(
                f"Max reconnect attempts ({self.max_reconnect_attempts}) reached for {self.config.camera_name}")
            return False

        wait_time = min(2 ** self.reconnect_attempts, 30)  # Max 30 seconds
        self.reconnect_attempts += 1

        logger.warning(
            f"Reconnect attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} "
            f"for {self.config.camera_name}, waiting {wait_time}s..."
        )
        time.sleep(wait_time)

        return self._safe_open_stream()

    def _check_stream_timeout(self):
        """Check for stream timeout with validation"""
        current_time = time.time()
        if current_time - self.last_packet_time > self.config.timeout * 2:
            logger.warning(f"Stream timeout detected for {self.config.camera_name}")
            raise AVError("Stream timeout")

    def _start_new_segment(self):
        """Initialize new recording segment with comprehensive checks"""
        try:
            self._cleanup_output()
            self.current_segment_path = self._get_segment_filename()

            # Verify we can write to the file
            try:
                test_file = self.current_segment_path + '.test'
                with open(test_file, 'wb') as f:
                    f.write(b'test')
                os.remove(test_file)
            except IOError as e:
                logger.error(f"Filesystem error: {str(e)}")
                raise

            # Check codec availability
            if 'libx264' not in av.codec.codecs_available:
                raise RuntimeError("libx264 codec not available")

            self.current_segment_start = time.time()
            self.output_container = av.open(self.current_segment_path, mode='w')
            self.output_stream = self.output_container.add_stream('libx264')

            bitrate, options = self.bitrate_controller.adjust_bitrate()
            if bitrate > 0:
                self.output_stream.bit_rate = bitrate * 1000

            self.output_stream.options = options
            logger.info(f"Started new segment: {self.current_segment_path}")

        except Exception as e:
            logger.error(f"Segment initialization failed: {str(e)}", exc_info=True)
            self._cleanup_output()
            raise

    def _process_stream(self):
        """Main stream processing loop with enhanced error handling"""
        while not self.stop_event.is_set():
            try:
                # Segment management
                if (not self.current_segment_start or
                        (time.time() - self.current_segment_start) >= self.config.segment_duration):
                    try:
                        self._close_current_segment()
                        self._start_new_segment()
                    except Exception as segment_error:
                        logger.error(f"Segment error: {str(segment_error)}")
                        time.sleep(1)
                        continue

                # Stream connection
                if not self.input_container and not self._safe_open_stream():
                    if not self._reconnect():
                        break

                # Packet processing
                for packet in self.input_container.demux(self.input_stream):
                    if self.stop_event.is_set():
                        break

                    if not packet.dts or packet.size == 0:
                        continue

                    self.last_packet_time = time.time()

                    # Bitrate detection
                    if not self.bitrate_controller.detection_complete:
                        self.bitrate_controller.detect_mode(packet)

                    # Frame processing
                    if self.output_container and self.output_stream:
                        try:
                            for frame in packet.decode():
                                try:
                                    encoded_packet = self.output_stream.encode(frame)
                                    try:
                                        self.output_container.mux(encoded_packet)
                                    except (ValueError, AVError) as mux_error:
                                        if "Invalid argument" in str(mux_error):
                                            logger.error("File write error, recreating segment")
                                            self._close_current_segment()
                                            self._start_new_segment()
                                            break
                                        raise
                                except AVError as encode_error:
                                    if encode_error.errno == 1094995529:  # Invalid data
                                        continue
                                    raise
                        except AVError as e:
                            logger.warning(f"AV processing error: {str(e)}")
                            continue

                    self._check_stream_timeout()

            except (AVError, FFmpegError) as e:
                error_msg = f"Stream error in {self.config.camera_name}"
                if "1414092869" in str(e):
                    error_msg = f"Camera {self.config.camera_name} unavailable"
                logger.warning(f"{error_msg}: {str(e)}")
                self._cleanup_input()
                time.sleep(5)
            except Exception as e:
                logger.error(f"Processing error: {str(e)}", exc_info=True)
                self._cleanup_input()
                time.sleep(1)

    def start(self):
        """Start recording with validation"""
        if self.running:
            logger.warning(f"Recorder for {self.config.camera_name} already running")
            return

        if not self._validate_archive_dir():
            logger.error(f"Cannot start recorder - invalid archive directory")
            return

        self.running = True
        self.stop_event.clear()
        try:
            self.thread = threading.Thread(
                target=self._process_stream,
                name=f"Recorder-{self.config.camera_name}",
                daemon=True
            )
            self.thread.start()
            logger.info(f"Started recorder for {self.config.camera_name}")
        except Exception as e:
            self.running = False
            logger.error(f"Failed to start recorder: {str(e)}", exc_info=True)
            raise

    def stop(self):
        """Stop recording safely"""
        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        try:
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    logger.warning(f"Recorder thread for {self.config.camera_name} did not stop gracefully")
        except Exception as e:
            logger.error(f"Thread stop error: {str(e)}")

        try:
            self._close_current_segment()
            self._cleanup_input()
            logger.info(f"Stopped recorder for {self.config.camera_name}")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}", exc_info=True)

    def check_camera_status(self) -> Dict[str, str]:
        """Check camera connection status with detailed info"""
        status = {
            'camera_name': self.config.camera_name,
            'status': 'unknown',
            'last_error': None,
            'rtsp_url': self.config.rtsp_url,
            'last_packet_time': self.last_packet_time,
            'reconnect_attempts': self.reconnect_attempts,
            'current_segment': self.current_segment_path
        }

        try:
            if self._safe_open_stream():
                status['status'] = 'available'
                self._cleanup_input()
            else:
                status['status'] = 'unavailable'
        except HTTPNotFoundError:
            status.update({'status': 'not_found', 'last_error': '404 Not Found'})
        except PermissionError:
            status.update({'status': 'access_denied', 'last_error': 'Permission denied'})
        except InvalidDataError as e:
            status.update({'status': 'invalid_data', 'last_error': str(e)})
        except Exception as e:
            status.update({'status': 'error', 'last_error': str(e)})

        return status