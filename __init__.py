from .cascade_detector import CascadeDetector

NODE_CLASS_MAPPINGS = {
    "CascadeDetectorAdvanced": CascadeDetector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CascadeDetectorAdvanced": "🎯 Cascade Detector Advanced (Bbox/Segm, Staged)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

