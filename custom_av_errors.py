"""
Кастомные исключения для совместимости с разными версиями PyAV
"""

class AVError(Exception):
    """Базовое исключение PyAV"""
    pass

class FFmpegError(Exception):
    """Ошибка FFmpeg"""
    pass

class HTTPNotFoundError(Exception):
    """Ресурс не найден (404)"""
    pass

class PermissionError(Exception):
    """Ошибка доступа"""
    pass

class InvalidDataError(Exception):
    """Неверные данные"""
    pass