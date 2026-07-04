# Feature extractor modules
from .dinov2_extractor import DINOv2Extractor
from .resnet_extractor import ResNet50Extractor
from .clip_extractor import CLIPExtractor

__all__ = ["DINOv2Extractor", "ResNet50Extractor", "CLIPExtractor"]
