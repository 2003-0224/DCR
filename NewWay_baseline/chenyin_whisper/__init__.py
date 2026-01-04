# chenyin_whisper/__init__.py

# 从 modules 模块中导出常用类
from .modules import CustomWhisperEncoderLayer, BaseModel

# 可选：定义包的元信息
__all__ = ["CustomWhisperEncoderLayer", "BaseModel"]
__version__ = "0.1.0"