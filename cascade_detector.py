import torch
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
from nodes import MAX_RESOLUTION
import comfy
from PIL import Image, ImageDraw
import cv2
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CascadeDetector:
    @classmethod
    def INPUT_TYPES(cls):
        detector_options = ["bbox", "segm"]
        scale_modes = ["bbox", "crop_region", "fixed"]
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["sequential", "parallel", "parallel_per_segment"], {"default": "sequential"}),
                "target_size": ("INT", {"default": 640, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "iou_threshold": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "include_masks_in_output": ("BOOLEAN", {"default": True}),
                "simplify_masks": ("BOOLEAN", {"default": True}),
                "simplify_kernel_size": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2}),
                "simplify_iterations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "segs_input": ("SEGS", {}),
                # Stage 1 Parameters + NEW INPUT FILTER
                "stage_1_enabled": ("BOOLEAN", {"default": True}),
                "stage_1_detector_type": (detector_options, {"default": "bbox"}),
                "stage_1_bbox_detector": ("BBOX_DETECTOR", {}),
                "stage_1_segm_detector": ("SEGM_DETECTOR", {}),
                "stage_1_confidence": ("FLOAT", {"default": 0.25, "min": 0.00, "max": 1.00, "step": 0.01}),
                "stage_1_iou_threshold": ("FLOAT", {"default": 0.45, "min": 0.00, "max": 1.00, "step": 0.01}),
                "stage_1_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "stage_1_classes": ("STRING", {"default": "", "multiline": False}),
                "stage_1_crop_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "stage_1_scale_mode": (scale_modes, {"default": "bbox"}),
                "stage_1_target_size": ("INT", {"default": 640, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "stage_1_max_size": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "stage_1_process_empty": ("BOOLEAN", {"default": True}),
                "stage1_input_filter_labels": ("STRING", {"default": "", "multiline": False, "tooltip": "Comma-separated labels to PROCESS on Stage 1 (e.g., 'person,car'). Empty = process all input segments."}),  # NEW!
                # Filtering & Output Control
                "min_confidence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_bbox_width": ("INT", {"default": 1, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "min_bbox_height": ("INT", {"default": 1, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                # Stage 2 Parameters + NEW INPUT FILTER
                "stage_2_enabled": ("BOOLEAN", {"default": True}),
                "stage_2_detector_type": (detector_options, {"default": "bbox"}),
                "stage_2_bbox_detector": ("BBOX_DETECTOR", {}),
                "stage_2_segm_detector": ("SEGM_DETECTOR", {}),
                "stage_2_confidence": ("FLOAT", {"default": 0.25, "min": 0.00, "max": 1.00, "step": 0.01}),
                "stage_2_iou_threshold": ("FLOAT", {"default": 0.45, "min": 0.00, "max": 1.00, "step": 0.01}),
                "stage_2_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "stage_2_classes": ("STRING", {"default": "", "multiline": False}),
                "stage_2_crop_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "stage_2_scale_mode": (scale_modes, {"default": "bbox"}),
                "stage_2_target_size": ("INT", {"default": 640, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "stage_2_max_size": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "stage_2_process_empty": ("BOOLEAN", {"default": False}),
                "stage2_input_filter_labels": ("STRING", {"default": "", "multiline": False, "tooltip": "Comma-separated labels to PROCESS on Stage 2 (e.g., 'face'). Empty = process all Stage 1 results."}),  # NEW!
                # Stage 3 Parameters + NEW INPUT FILTER
                "stage_3_enabled": ("BOOLEAN", {"default": True}),
                "stage_3_detector_type": (detector_options, {"default": "bbox"}),
                "stage_3_bbox_detector": ("BBOX_DETECTOR", {}),
                "stage_3_segm_detector": ("SEGM_DETECTOR", {}),
                "stage_3_confidence": ("FLOAT", {"default": 0.25, "min": 0.00, "max": 1.00, "step": 0.01}),
                "stage_3_iou_threshold": ("FLOAT", {"default": 0.45, "min": 0.00, "max": 1.00, "step": 0.01}),
                "stage_3_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "stage_3_classes": ("STRING", {"default": "", "multiline": False}),
                "stage_3_crop_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "stage_3_scale_mode": (scale_modes, {"default": "bbox"}),
                "stage_3_target_size": ("INT", {"default": 640, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "stage_3_max_size": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "stage_3_process_empty": ("BOOLEAN", {"default": False}),
                "stage3_input_filter_labels": ("STRING", {"default": "", "multiline": False, "tooltip": "Comma-separated labels to PROCESS on Stage 3 (e.g., 'eyes,nose'). Empty = process all Stage 2 results."}),  # NEW!
                "drop_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}
        }

    RETURN_TYPES = ("SEGS", "IMAGE", "IMAGE", "SEGS", "SEGS", "SEGS", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "segs_output_all_stages (Combined)",
        "preview_image (Combined Detections)",
        "cropped_fragments_image (All Detections)",
        "stage1_segs (Stage 1 Results)",
        "stage2_segs (Stage 2 Results)",
        "stage3_segs (Stage 3 Results)",
        "masked_fragments_image (Masked Fragments)",
        "image_bypass (Original if No Detections)"
    )
    FUNCTION = "process"
    CATEGORY = "Detection/Cascade"
    DESCRIPTION = """Cascaded detector with INPUT LABEL FILTERING before detection on each stage.
NEW: stageX_input_filter_labels filters WHICH segments get processed on each stage (e.g., Stage 2 processes ONLY 'face' segments from Stage 1).
This prevents wasted detection attempts (e.g., searching for eyes on arms/torso) and enables efficient multi-stage workflows.
FIXED: Correctly handles non-square resolutions (e.g., 1152x1280) and Impact Pack compatibility by using numpy masks."""

    def __init__(self):
        self.device = comfy.model_management.get_torch_device()
        self.stage_results = {}
        self.IMPACT_AVAILABLE = self._check_impact_availability_at_init()

    def _check_impact_availability_at_init(self):
        try:
            from impact.core import SEG
            self.SEG_IMPACT = SEG
            return True
        except ImportError as e:
            logger.error(f"Impact Pack not available: {e}")
            return False

    def tensor_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        return tensor.cpu().numpy()

    def np_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        if array.dtype != np.float32:
            array = array.astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def resize_image(self, image: np.ndarray, target_size: int, max_size: int) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        scale = target_size / min(h, w)
        if scale * max(h, w) > max_size:
            scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale

    def calculate_crop_region(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int], crop_factor: float = 3.0) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        img_h, img_w = image_shape
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        new_size = int(max(bbox_w, bbox_h) * crop_factor)
        new_x1 = max(0, center_x - new_size // 2)
        new_y1 = max(0, center_y - new_size // 2)
        new_x2 = min(img_w, center_x + new_size // 2)
        new_y2 = min(img_h, center_y + new_size // 2)
        return (new_x1, new_y1, new_x2, new_y2)

    def crop_image(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = region
        return image[y1:y2, x1:x2]

    def filter_by_classes(self, segs: List[Dict], classes_filter: str) -> List[Dict]:
        """Filters detection RESULTS by class (AFTER detection)"""
        if not classes_filter or classes_filter.strip() == "":
            return segs
        allowed_classes = [c.strip().lower() for c in classes_filter.split(",") if c.strip()]
        return [seg for seg in segs if seg.get("label", "").lower() in allowed_classes]

    # NEW: Filters INPUT segments by label BEFORE detection
    def filter_segs_by_input_labels(self, segs: List[Dict], labels_filter: str, stage_name: str) -> List[Dict]:
        if not labels_filter or labels_filter.strip() == "":
            return segs  # No filtering
        allowed_labels = [l.strip().lower() for l in labels_filter.split(",") if l.strip()]
        filtered = [seg for seg in segs if seg.get("label", "").lower() in allowed_labels]
        # Log filtering action
        if len(segs) > 0:
            logger.info(f"{stage_name}: Filtered {len(segs)} → {len(filtered)} segments by input labels: {allowed_labels}")
        return filtered

    def apply_nms(self, segs: List[Dict], iou_threshold: float) -> List[Dict]:
        if not segs:
            return []
        segs.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        selected = []
        while segs:
            current = segs.pop(0)
            selected.append(current)
            segs = [seg for seg in segs if self.calculate_iou(current["bbox"], seg["bbox"]) < iou_threshold]
        return selected

    def calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def recalculate_mask(self, original_image_np: np.ndarray, seg_result: Dict, original_image_shape_hw: Tuple[int, int], simplify_masks: bool, kernel_size: int, iterations: int):
        crop_region = seg_result["crop_region"]
        x1_cr, y1_cr, x2_cr, y2_cr = crop_region
        x1_cr = max(0, min(original_image_shape_hw[1], x1_cr))
        y1_cr = max(0, min(original_image_shape_hw[0], y1_cr))
        x2_cr = max(0, min(original_image_shape_hw[1], x2_cr))
        y2_cr = max(0, min(original_image_shape_hw[0], y2_cr))
        fragment_h = y2_cr - y1_cr
        fragment_w = x2_cr - x1_cr
        full_mask = np.zeros(original_image_shape_hw, dtype=np.uint8)
        old_cropped_mask = seg_result.get("cropped_mask")
        if old_cropped_mask is not None and old_cropped_mask.size > 0:
            if isinstance(old_cropped_mask, torch.Tensor):
                old_cropped_mask = old_cropped_mask.cpu().numpy()
            if old_cropped_mask.ndim == 3 and old_cropped_mask.shape[0] == 1:
                old_cropped_mask = old_cropped_mask[0]
            elif old_cropped_mask.ndim > 2:
                old_cropped_mask = old_cropped_mask.max(axis=0) if old_cropped_mask.ndim == 3 else old_cropped_mask
            if old_cropped_mask.dtype == np.float32:
                if old_cropped_mask.max() > 1.0:
                    old_cropped_mask = (old_cropped_mask * 255).astype(np.uint8)
                else:
                    old_cropped_mask = (old_cropped_mask * 255).astype(np.uint8)
            elif old_cropped_mask.dtype != np.uint8:
                old_cropped_mask = old_cropped_mask.astype(np.uint8)
            if old_cropped_mask.shape[:2] != (fragment_h, fragment_w):
                resized_mask = cv2.resize(old_cropped_mask, (fragment_w, fragment_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized_mask = old_cropped_mask
            if simplify_masks:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                resized_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            if resized_mask.shape[:2] == (fragment_h, fragment_w):
                full_mask[y1_cr:y2_cr, x1_cr:x2_cr] = resized_mask
            new_cropped_mask = full_mask[y1_cr:y2_cr, x1_cr:x2_cr].astype(np.float32) / 255.0
            if new_cropped_mask.shape[:2] != (fragment_h, fragment_w):
                new_cropped_mask = np.zeros((fragment_h, fragment_w), dtype=np.float32)
            seg_result["cropped_mask"] = new_cropped_mask
        return seg_result

    def detect_with_model(self, detector, detector_type: str, image_tensor: torch.Tensor, confidence: float, dilation: int, crop_factor: float, drop_size: int) -> List[Dict]:
        if not self.IMPACT_AVAILABLE or detector is None:
            return []
        try:
            shape, segs_list_impact = detector.detect(image_tensor, confidence, dilation, crop_factor, drop_size, detailer_hook=None)
            unified_segs = []
            for seg_impact in segs_list_impact:
                if seg_impact is None or not hasattr(seg_impact, "bbox") or not hasattr(seg_impact, "confidence"):
                    continue
                bbox = seg_impact.bbox
                if bbox is None:
                    continue
                if isinstance(bbox, (list, tuple)):
                    bbox = tuple(bbox)
                elif hasattr(bbox, "tolist"):
                    bbox = tuple(bbox.tolist())
                else:
                    try:
                        bbox = tuple(bbox)
                    except TypeError:
                        continue
                confidence_val = seg_impact.confidence
                if confidence_val is None:
                    continue
                if hasattr(confidence_val, "item"):
                    confidence_val = confidence_val.item()
                elif isinstance(confidence_val, (list, tuple, np.ndarray)):
                    if len(confidence_val) > 0:
                        confidence_val = float(confidence_val[0])
                    else:
                        continue
                else:
                    try:
                        confidence_val = float(confidence_val)
                    except TypeError:
                        continue
                cropped_mask = getattr(seg_impact, "cropped_mask", None)
                unified_segs.append({
                    "bbox": bbox,
                    "crop_region": getattr(seg_impact, "crop_region", bbox),
                    "label": getattr(seg_impact, "label", "object"),
                    "confidence": confidence_val,
                    "cropped_mask": cropped_mask,
                    "orig_shape": shape,
                })
            return unified_segs
        except Exception as e:
            logger.error(f"Error during {detector_type} detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_color_for_stage(self, stage_idx: int) -> Tuple[int, int, int, int]:
        colors = [
            (255, 0, 0, 255),    # Stage 1: Red
            (0, 255, 0, 255),    # Stage 2: Green
            (0, 0, 255, 255),    # Stage 3: Blue
            (255, 255, 0, 255),  # Combined: Yellow
        ]
        return colors[stage_idx % len(colors)]

    def create_preview_image_staged(self, image: torch.Tensor, segs_list: List[Dict], stage_assignments: List[int]) -> torch.Tensor:
        img_np = self.tensor_to_np(image)
        if img_np.ndim == 3 and img_np.shape[2] in [3, 4]:
            img_pil = Image.fromarray((img_np[:, :, :3] * 255).astype(np.uint8))
        else:
            img_pil = Image.fromarray((img_np[..., :3] * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_pil, mode="RGBA")
        for seg, stage_idx in zip(segs_list, stage_assignments):
            x1, y1, x2, y2 = seg["bbox"]
            color = self.get_color_for_stage(stage_idx)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{seg.get('label', 'obj')}: {seg.get('confidence', 0):.2f}"
            draw.text((x1, y1 - 12), label, fill=color)
        preview_np = np.array(img_pil).astype(np.float32) / 255.0
        if preview_np.ndim == 2:
            preview_np = np.stack([preview_np] * 3, axis=-1)
        return torch.from_numpy(preview_np).unsqueeze(0)

    def create_cropped_fragments_image(self, image_tensor: torch.Tensor, segs_list: List[Dict], padding: int = 10) -> torch.Tensor:
        img_np = self.tensor_to_np(image_tensor)
        h, w, c = img_np.shape
        bg_color = (0, 0, 0, 0) if c == 4 else (0, 0, 0)
        if not segs_list:
            canvas_img = Image.new("RGBA" if c == 4 else "RGB", (padding * 2, padding * 2), color=bg_color)
            final_np = np.array(canvas_img).astype(np.float32) / 255.0
            if final_np.ndim == 2:
                final_np = np.stack([final_np] * 3, axis=-1)
            if final_np.shape[2] == 3 and c == 4:
                final_np = np.concatenate([final_np, np.ones((final_np.shape[0], final_np.shape[1], 1), dtype=final_np.dtype)], axis=-1)
            return torch.from_numpy(final_np).unsqueeze(0)
        canvas_size = max(h, w) + 2 * padding
        canvas_img = Image.new("RGBA" if c == 4 else "RGB", (canvas_size, canvas_size), color=bg_color)
        current_x = padding
        current_y = padding
        row_height = 0
        max_row_width = canvas_size - 2 * padding
        for seg in segs_list:
            crop_region = seg["crop_region"]
            frag_np = self.crop_image(img_np, crop_region)
            if frag_np.size == 0:
                continue
            frag_pil = Image.fromarray((frag_np * 255).astype(np.uint8))
            frag_w, frag_h = frag_pil.size
            if current_x + frag_w > max_row_width:
                current_x = padding
                current_y += row_height + padding
                row_height = 0
            canvas_img.paste(frag_pil, (current_x, current_y))
            current_x += frag_w + padding
            row_height = max(row_height, frag_h)
        final_canvas = canvas_img.crop((0, 0, min(canvas_size, current_x + padding), min(canvas_size, current_y + row_height + padding)))
        final_np = np.array(final_canvas).astype(np.float32) / 255.0
        if final_np.ndim == 2:
            final_np = np.stack([final_np] * 3, axis=-1)
        if final_np.shape[2] == 3 and c == 4:
            final_np = np.concatenate([final_np, np.ones((final_np.shape[0], final_np.shape[1], 1), dtype=final_np.dtype)], axis=-1)
        return torch.from_numpy(final_np).unsqueeze(0)

    def create_masked_fragments_image(self, image: torch.Tensor, segs_list: List[Dict], padding: int = 10) -> torch.Tensor:
        img_np = self.tensor_to_np(image)
        h, w, c = img_np.shape
        bg_color = (0, 0, 0, 0) if c == 4 else (0, 0, 0)
        if not segs_list:
            canvas_img = Image.new("RGBA" if c == 4 else "RGB", (padding * 2, padding * 2), color=bg_color)
            final_np = np.array(canvas_img).astype(np.float32) / 255.0
            if final_np.ndim == 2:
                final_np = np.stack([final_np] * 3, axis=-1)
            if final_np.shape[2] == 3 and c == 4:
                final_np = np.concatenate([final_np, np.ones((final_np.shape[0], final_np.shape[1], 1), dtype=final_np.dtype)], axis=-1)
            return torch.from_numpy(final_np).unsqueeze(0)
        canvas_size = max(h, w) + 2 * padding
        canvas_img = Image.new("RGBA" if c == 4 else "RGB", (canvas_size, canvas_size), color=bg_color)
        current_x = padding
        current_y = padding
        row_height = 0
        max_row_width = canvas_size - 2 * padding
        for seg in segs_list:
            crop_region = seg["crop_region"]
            cropped_mask_np = seg.get("cropped_mask")
            frag_np = self.crop_image(img_np, crop_region)
            if frag_np.size == 0 or cropped_mask_np is None:
                continue
            if isinstance(cropped_mask_np, torch.Tensor):
                cropped_mask_np = cropped_mask_np.cpu().numpy()
            if cropped_mask_np.ndim == 3 and cropped_mask_np.shape[0] == 1:
                cropped_mask_np = cropped_mask_np[0]
            elif cropped_mask_np.ndim != 2:
                continue
            expanded_mask = np.stack([cropped_mask_np] * 3, axis=-1) if frag_np.shape[2] == 3 else np.stack([cropped_mask_np] * 4, axis=-1)
            masked_frag_np = frag_np * expanded_mask
            frag_pil = Image.fromarray((masked_frag_np * 255).astype(np.uint8))
            frag_w, frag_h = frag_pil.size
            if current_x + frag_w > max_row_width:
                current_x = padding
                current_y += row_height + padding
                row_height = 0
            canvas_img.paste(frag_pil, (current_x, current_y))
            current_x += frag_w + padding
            row_height = max(row_height, frag_h)
        final_canvas = canvas_img.crop((0, 0, min(canvas_size, current_x + padding), min(canvas_size, current_y + row_height + padding)))
        final_np = np.array(final_canvas).astype(np.float32) / 255.0
        if final_np.ndim == 2:
            final_np = np.stack([final_np] * 3, axis=-1)
        if final_np.shape[2] == 3 and c == 4:
            final_np = np.concatenate([final_np, np.ones((final_np.shape[0], final_np.shape[1], 1), dtype=final_np.dtype)], axis=-1)
        return torch.from_numpy(final_np).unsqueeze(0)

    # CRITICAL FIX: Impact Pack requires numpy.ndarray masks, NOT torch.Tensor
    def convert_to_segs_format(self, segs_list: List[Dict], image_shape_wh: Tuple[int, int]) -> Optional[Tuple]:
        if not self.IMPACT_AVAILABLE:
            logger.error("Impact Pack not available. Cannot convert to SEGS format.")
            return None
        try:
            segs_objects = []
            for seg in segs_list:
                # ALWAYS use numpy.ndarray for cropped_mask (Impact Pack requirement)
                cropped_mask_np = seg.get("cropped_mask")
                if cropped_mask_np is not None:
                    if isinstance(cropped_mask_np, torch.Tensor):
                        cropped_mask_np = cropped_mask_np.cpu().numpy()
                    if cropped_mask_np.ndim == 3:
                        if cropped_mask_np.shape[0] == 1:
                            cropped_mask_np = cropped_mask_np.squeeze(0)
                        else:
                            cropped_mask_np = cropped_mask_np.mean(axis=0)
                    if cropped_mask_np.max() > 1.0:
                        cropped_mask_np = cropped_mask_np / 255.0
                    # CRITICAL: Verify dimensions match crop_region EXACTLY
                    crop_region = seg.get("crop_region", seg["bbox"])
                    expected_h = crop_region[3] - crop_region[1]
                    expected_w = crop_region[2] - crop_region[0]
                    if cropped_mask_np.shape != (expected_h, expected_w):
                        logger.warning(
                            f"Mask size mismatch! Expected ({expected_h}, {expected_w}), got {cropped_mask_np.shape}. "
                            f"Creating fallback empty mask."
                        )
                        cropped_mask_np = np.zeros((expected_h, expected_w), dtype=np.float32)
                else:
                    # ALWAYS create valid empty mask with CORRECT dimensions as numpy array
                    crop_region = seg.get("crop_region", seg["bbox"])
                    h = crop_region[3] - crop_region[1]
                    w = crop_region[2] - crop_region[0]
                    cropped_mask_np = np.zeros((h, w), dtype=np.float32)
                # Create SEG object with numpy mask (NOT torch tensor) - THIS FIXES THE ERROR
                seg_obj = self.SEG_IMPACT(
                    cropped_image=None,
                    cropped_mask=cropped_mask_np,  # ← MUST be numpy.ndarray for Impact Pack compatibility
                    confidence=seg.get("confidence", 0.5),
                    crop_region=seg.get("crop_region", seg["bbox"]),
                    bbox=seg["bbox"],
                    label=seg.get("label", "object"),
                    control_net_wrapper=None,
                )
                segs_objects.append(seg_obj)
            return (image_shape_wh, segs_objects)
        except Exception as e:
            logger.error(f"Failed to convert to SEGS format: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_sequential(self, image: torch.Tensor, initial_segs: Optional[List[Dict]], enabled_stages: List[bool], detectors_info: List, settings: List[Dict], include_masks_in_output: bool, original_image_shape_hw: Tuple[int, int], original_image_np: np.ndarray, simplify_masks: bool, kernel_size: int, iterations: int, stage_1_process_empty: bool, stage_2_process_empty: bool, stage_3_process_empty: bool, stage1_input_filter_labels="", stage2_input_filter_labels="", stage3_input_filter_labels="") -> Dict[str, List[Dict]]:
        results = {"stage1": [], "stage2": [], "stage3": [], "combined": []}
        stage_assignments = {"stage1": [], "stage2": [], "stage3": [], "combined": []}
        current_segs = initial_segs if initial_segs else []

        # === Stage 1 ===
        if enabled_stages[0] and detectors_info[0] is not None:
            det, det_type, _ = detectors_info[0]
            if det is not None:
                # NEW: Apply input filter BEFORE processing
                if current_segs:  # Only filter if we have input segments (from segs_input)
                    current_segs = self.filter_segs_by_input_labels(current_segs, stage1_input_filter_labels, "Stage 1 Input Filter")
                if not current_segs and stage_1_process_empty:
                    img_np = self.tensor_to_np(image)
                    if settings[0]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(img_np, settings[0]["target_size"], settings[0]["max_size"])
                        img_tensor = self.np_to_tensor(scaled_img)
                    else:
                        img_tensor = image
                    scale = 1.0
                    stage1_segs_raw = self.detect_with_model(det, det_type, img_tensor, settings[0]["confidence"], settings[0]["dilation"], settings[0]["crop_factor"], settings[0]["drop_size"])
                    stage1_segs = []
                    for seg_dict in stage1_segs_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                            orig_crop_region = self.calculate_crop_region(orig_bbox, original_image_shape_hw, settings[0]["crop_factor"])
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                        else:
                            crop_region = seg_dict["crop_region"]
                            h = crop_region[3] - crop_region[1]
                            w = crop_region[2] - crop_region[0]
                            seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                        stage1_segs.append(seg_dict)
                        stage_assignments["stage1"].append(0)
                    results["stage1"] = stage1_segs
                    current_segs = stage1_segs
                elif current_segs:
                    img_np = self.tensor_to_np(image)
                    if settings[0]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(img_np, settings[0]["target_size"], settings[0]["max_size"])
                        img_tensor = self.np_to_tensor(scaled_img)
                    else:
                        img_tensor = image
                    scale = 1.0
                    stage1_segs_raw = self.detect_with_model(det, det_type, img_tensor, settings[0]["confidence"], settings[0]["dilation"], settings[0]["crop_factor"], settings[0]["drop_size"])
                    stage1_segs = []
                    for seg_dict in stage1_segs_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                            orig_crop_region = self.calculate_crop_region(orig_bbox, original_image_shape_hw, settings[0]["crop_factor"])
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                        else:
                            crop_region = seg_dict["crop_region"]
                            h = crop_region[3] - crop_region[1]
                            w = crop_region[2] - crop_region[0]
                            seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                        stage1_segs.append(seg_dict)
                        stage_assignments["stage1"].append(0)
                    if settings[0]["classes"]:
                        stage1_segs = self.filter_by_classes(stage1_segs, settings[0]["classes"])
                    stage1_segs = self.apply_nms(stage1_segs, settings[0]["iou"])
                    results["stage1"] = stage1_segs
                    current_segs = stage1_segs
                else:
                    results["stage1"] = []
            else:
                results["stage1"] = []
        else:
            results["stage1"] = []

        # === Stage 2 ===
        if enabled_stages[1] and detectors_info[1] is not None:
            det, det_type, _ = detectors_info[1]
            if det is not None:
                # NEW: Apply input filter BEFORE processing Stage 2
                current_segs = self.filter_segs_by_input_labels(current_segs, stage2_input_filter_labels, "Stage 2 Input Filter")
                if not current_segs and stage_2_process_empty:
                    img_np = self.tensor_to_np(image)
                    if settings[1]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(img_np, settings[1]["target_size"], settings[1]["max_size"])
                        img_tensor = self.np_to_tensor(scaled_img)
                    else:
                        img_tensor = image
                    scale = 1.0
                    detected_raw = self.detect_with_model(det, det_type, img_tensor, settings[1]["confidence"], settings[1]["dilation"], settings[1]["crop_factor"], settings[1]["drop_size"])
                    stage2_segs = []
                    for seg_dict in detected_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                            orig_crop_region = self.calculate_crop_region(orig_bbox, original_image_shape_hw, settings[1]["crop_factor"])
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                        else:
                            crop_region = seg_dict["crop_region"]
                            h = crop_region[3] - crop_region[1]
                            w = crop_region[2] - crop_region[0]
                            seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                        stage2_segs.append(seg_dict)
                        stage_assignments["stage2"].append(1)
                    results["stage2"] = stage2_segs
                    current_segs = stage2_segs
                elif current_segs:
                    stage2_segs = []
                    img_np = self.tensor_to_np(image)
                    for parent_seg in current_segs:
                        crop_region = parent_seg.get("crop_region", parent_seg["bbox"])
                        cropped = self.crop_image(img_np, crop_region)
                        if cropped.size == 0:
                            continue
                        scale = 1.0
                        if settings[1]["scale_mode"] == "fixed":
                            scaled_crop, scale = self.resize_image(cropped, settings[1]["target_size"], settings[1]["max_size"])
                            cropped_tensor = self.np_to_tensor(scaled_crop)
                        elif settings[1]["scale_mode"] == "bbox":
                            bbox_in_crop = (parent_seg["bbox"][0] - crop_region[0], parent_seg["bbox"][1] - crop_region[1], parent_seg["bbox"][2] - crop_region[0], parent_seg["bbox"][3] - crop_region[1])
                            bbox_size = max(bbox_in_crop[2] - bbox_in_crop[0], bbox_in_crop[3] - bbox_in_crop[1])
                            if bbox_size > 0:
                                scale = settings[1]["target_size"] / bbox_size
                                max_crop_dim = max(cropped.shape[:2])
                                if scale * max_crop_dim > settings[1]["max_size"]:
                                    scale = settings[1]["max_size"] / max_crop_dim
                                new_h = int(cropped.shape[0] * scale)
                                new_w = int(cropped.shape[1] * scale)
                                if new_h > 0 and new_w > 0:
                                    scaled_crop = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                                    cropped_tensor = self.np_to_tensor(scaled_crop)
                                else:
                                    cropped_tensor = self.np_to_tensor(cropped)
                                    scale = 1.0
                            else:
                                cropped_tensor = self.np_to_tensor(cropped)
                                scale = 1.0
                        else:
                            cropped_tensor = self.np_to_tensor(cropped)
                            scale = 1.0
                        detected_raw = self.detect_with_model(det, det_type, cropped_tensor, settings[1]["confidence"], settings[1]["dilation"], settings[1]["crop_factor"], settings[1]["drop_size"])
                        for seg_dict in detected_raw:
                            if settings[1]["classes"]:
                                if seg_dict.get("label", "").lower() not in [c.strip().lower() for c in settings[1]["classes"].split(",") if c.strip()]:
                                    continue
                            crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
                            seg_x1_scaled, seg_y1_scaled, seg_x2_scaled, seg_y2_scaled = seg_dict["bbox"]
                            if scale != 1.0:
                                seg_x1_unscaled = seg_x1_scaled / scale
                                seg_y1_unscaled = seg_y1_scaled / scale
                                seg_x2_unscaled = seg_x2_scaled / scale
                                seg_y2_unscaled = seg_y2_scaled / scale
                            else:
                                seg_x1_unscaled = seg_x1_scaled
                                seg_y1_unscaled = seg_y1_scaled
                                seg_x2_unscaled = seg_x2_scaled
                                seg_y2_unscaled = seg_y2_scaled
                            abs_x1 = int(crop_x1 + seg_x1_unscaled)
                            abs_y1 = int(crop_y1 + seg_y1_unscaled)
                            abs_x2 = int(crop_x1 + seg_x2_unscaled)
                            abs_y2 = int(crop_y1 + seg_y2_unscaled)
                            abs_x1 = max(0, min(original_image_shape_hw[1], abs_x1))
                            abs_y1 = max(0, min(original_image_shape_hw[0], abs_y1))
                            abs_x2 = max(0, min(original_image_shape_hw[1], abs_x2))
                            abs_y2 = max(0, min(original_image_shape_hw[0], abs_y2))
                            seg_dict["bbox"] = (abs_x1, abs_y1, abs_x2, abs_y2)
                            seg_dict["crop_region"] = self.calculate_crop_region(seg_dict["bbox"], original_image_shape_hw, settings[1]["crop_factor"])
                            seg_dict["orig_shape"] = original_image_shape_hw
                            if include_masks_in_output:
                                seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                            else:
                                crop_region = seg_dict["crop_region"]
                                h = crop_region[3] - crop_region[1]
                                w = crop_region[2] - crop_region[0]
                                seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                            stage2_segs.append(seg_dict)
                        stage_assignments["stage2"].append(1)
                    stage2_segs = self.apply_nms(stage2_segs, settings[1]["iou"])
                    results["stage2"] = stage2_segs
                    current_segs = stage2_segs
                else:
                    results["stage2"] = []
            else:
                results["stage2"] = []
        else:
            results["stage2"] = []

        # === Stage 3 ===
        if enabled_stages[2] and detectors_info[2] is not None:
            det, det_type, _ = detectors_info[2]
            if det is not None:
                # NEW: Apply input filter BEFORE processing Stage 3
                current_segs = self.filter_segs_by_input_labels(current_segs, stage3_input_filter_labels, "Stage 3 Input Filter")
                if not current_segs and stage_3_process_empty:
                    img_np = self.tensor_to_np(image)
                    if settings[2]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(img_np, settings[2]["target_size"], settings[2]["max_size"])
                        img_tensor = self.np_to_tensor(scaled_img)
                    else:
                        img_tensor = image
                    scale = 1.0
                    detected_raw = self.detect_with_model(det, det_type, img_tensor, settings[2]["confidence"], settings[2]["dilation"], settings[2]["crop_factor"], settings[2]["drop_size"])
                    stage3_segs = []
                    for seg_dict in detected_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                            orig_crop_region = self.calculate_crop_region(orig_bbox, original_image_shape_hw, settings[2]["crop_factor"])
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                        else:
                            crop_region = seg_dict["crop_region"]
                            h = crop_region[3] - crop_region[1]
                            w = crop_region[2] - crop_region[0]
                            seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                        stage3_segs.append(seg_dict)
                        stage_assignments["stage3"].append(2)
                    results["stage3"] = stage3_segs
                    current_segs = stage3_segs
                elif current_segs:
                    stage3_segs = []
                    img_np = self.tensor_to_np(image)
                    for parent_seg in current_segs:
                        crop_region = parent_seg.get("crop_region", parent_seg["bbox"])
                        cropped = self.crop_image(img_np, crop_region)
                        if cropped.size == 0:
                            continue
                        if settings[2]["scale_mode"] == "fixed":
                            scaled_crop, scale = self.resize_image(cropped, settings[2]["target_size"], settings[2]["max_size"])
                            cropped_tensor = self.np_to_tensor(scaled_crop)
                        else:
                            cropped_tensor = self.np_to_tensor(cropped)
                            scale = 1.0
                        detected_raw = self.detect_with_model(det, det_type, cropped_tensor, settings[2]["confidence"], settings[2]["dilation"], settings[2]["crop_factor"], settings[2]["drop_size"])
                        for seg_dict in detected_raw:
                            if settings[2]["classes"]:
                                if seg_dict.get("label", "").lower() not in [c.strip().lower() for c in settings[2]["classes"].split(",") if c.strip()]:
                                    continue
                            crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
                            seg_x1_scaled, seg_y1_scaled, seg_x2_scaled, seg_y2_scaled = seg_dict["bbox"]
                            abs_x1 = int(crop_x1 + seg_x1_scaled)
                            abs_y1 = int(crop_y1 + seg_y1_scaled)
                            abs_x2 = int(crop_x1 + seg_x2_scaled)
                            abs_y2 = int(crop_y1 + seg_y2_scaled)
                            abs_x1 = max(0, min(original_image_shape_hw[1], abs_x1))
                            abs_y1 = max(0, min(original_image_shape_hw[0], abs_y1))
                            abs_x2 = max(0, min(original_image_shape_hw[1], abs_x2))
                            abs_y2 = max(0, min(original_image_shape_hw[0], abs_y2))
                            seg_dict["bbox"] = (abs_x1, abs_y1, abs_x2, abs_y2)
                            seg_dict["crop_region"] = self.calculate_crop_region(seg_dict["bbox"], original_image_shape_hw, settings[2]["crop_factor"])
                            seg_dict["orig_shape"] = original_image_shape_hw
                            if include_masks_in_output:
                                seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                            else:
                                crop_region = seg_dict["crop_region"]
                                h = crop_region[3] - crop_region[1]
                                w = crop_region[2] - crop_region[0]
                                seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                            stage3_segs.append(seg_dict)
                        stage_assignments["stage3"].append(2)
                    stage3_segs = self.apply_nms(stage3_segs, settings[2]["iou"])
                    results["stage3"] = stage3_segs
                    current_segs = stage3_segs
                else:
                    results["stage3"] = []
            else:
                results["stage3"] = []
        else:
            results["stage3"] = []

        # Combine results
        all_segs = []
        for seg in results["stage1"]:
            all_segs.append(seg)
            stage_assignments["combined"].append(0)
        for seg in results["stage2"]:
            all_segs.append(seg)
            stage_assignments["combined"].append(1)
        for seg in results["stage3"]:
            all_segs.append(seg)
            stage_assignments["combined"].append(2)
        results["combined"] = self.apply_nms(all_segs, settings[0]["iou_threshold"])
        return results, stage_assignments
    def process_parallel_per_segment(self, image: torch.Tensor, initial_segs: Optional[List[Dict]], enabled_stages: List[bool], detectors_info: List, settings: List[Dict], include_masks_in_output: bool, original_image_shape_hw: Tuple[int, int], original_image_np: np.ndarray, simplify_masks: bool, kernel_size: int, iterations: int, stage_1_process_empty: bool, stage_2_process_empty: bool, stage_3_process_empty: bool, stage1_input_filter_labels="", stage2_input_filter_labels="", stage3_input_filter_labels="") -> Dict[str, List[Dict]]:
        """
        NEW MODE: For EACH input segment, run ALL enabled stages INDEPENDENTLY (not sequentially).
        Stage 1 does NOT feed into Stage 2 — all stages process the SAME original segment.
        FALLBACK: If no detections found on segments AND stage_X_process_empty=True → run detection on FULL IMAGE.
        """
        results = {"stage1": [], "stage2": [], "stage3": [], "combined": []}
        stage_assignments = {"stage1": [], "stage2": [], "stage3": [], "combined": []}
        
        # Apply global input filter BEFORE processing any stages
        current_segs = initial_segs if initial_segs else []
        if not current_segs and any([stage_1_process_empty, stage_2_process_empty, stage_3_process_empty]):
            # No input segments + process_empty enabled → run on full image
            img_np = self.tensor_to_np(image)
            for i in range(3):
                if enabled_stages[i] and detectors_info[i] is not None and [stage_1_process_empty, stage_2_process_empty, stage_3_process_empty][i]:
                    det, det_type, _ = detectors_info[i]
                    if det is None:
                        continue
                    if settings[i]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(img_np, settings[i]["target_size"], settings[i]["max_size"])
                        img_tensor = self.np_to_tensor(scaled_img)
                    else:
                        img_tensor = image
                        scale = 1.0
                    detected_raw = self.detect_with_model(det, det_type, img_tensor, settings[i]["confidence"], settings[i]["dilation"], settings[i]["crop_factor"], settings[i]["drop_size"])
                    detected = []
                    for seg_dict in detected_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                            orig_crop_region = self.calculate_crop_region(orig_bbox, original_image_shape_hw, settings[i]["crop_factor"])
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                        else:
                            crop_region = seg_dict["crop_region"]
                            h = crop_region[3] - crop_region[1]
                            w = crop_region[2] - crop_region[0]
                            seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                        if settings[i]["classes"]:
                            if seg_dict.get("label", "").lower() not in [c.strip().lower() for c in settings[i]["classes"].split(",") if c.strip()]:
                                continue
                        detected.append(seg_dict)
                        if i == 0:
                            stage_assignments["stage1"].append(0)
                        elif i == 1:
                            stage_assignments["stage2"].append(1)
                        elif i == 2:
                            stage_assignments["stage3"].append(2)
                    detected = self.apply_nms(detected, settings[i]["iou"])
                    if i == 0:
                        results["stage1"].extend(detected)
                    elif i == 1:
                        results["stage2"].extend(detected)
                    elif i == 2:
                        results["stage3"].extend(detected)
            # Combine all stages
            all_segs = results["stage1"] + results["stage2"] + results["stage3"]
            for _ in results["stage1"]:
                stage_assignments["combined"].append(0)
            for _ in results["stage2"]:
                stage_assignments["combined"].append(1)
            for _ in results["stage3"]:
                stage_assignments["combined"].append(2)
            results["combined"] = self.apply_nms(all_segs, settings[0]["iou_threshold"])
            return results, stage_assignments
        
        # Process EACH input segment INDEPENDENTLY with ALL enabled stages
        for parent_seg in current_segs:
            # Apply per-stage input filters BEFORE processing this segment
            should_process_stage1 = not stage1_input_filter_labels or parent_seg.get("label", "").lower() in [l.strip().lower() for l in stage1_input_filter_labels.split(",") if l.strip()]
            should_process_stage2 = not stage2_input_filter_labels or parent_seg.get("label", "").lower() in [l.strip().lower() for l in stage2_input_filter_labels.split(",") if l.strip()]
            should_process_stage3 = not stage3_input_filter_labels or parent_seg.get("label", "").lower() in [l.strip().lower() for l in stage3_input_filter_labels.split(",") if l.strip()]
            
            crop_region = parent_seg.get("crop_region", parent_seg["bbox"])
            cropped = self.crop_image(original_image_np, crop_region)
            if cropped.size == 0:
                continue
            
            # Run Stage 1 on this segment (if enabled and passes filter)
            if enabled_stages[0] and detectors_info[0] is not None and should_process_stage1:
                det, det_type, _ = detectors_info[0]
                if det is not None:
                    if settings[0]["scale_mode"] == "fixed":
                        scaled_crop, scale = self.resize_image(cropped, settings[0]["target_size"], settings[0]["max_size"])
                        cropped_tensor = self.np_to_tensor(scaled_crop)
                    else:
                        cropped_tensor = self.np_to_tensor(cropped)
                        scale = 1.0
                    detected_raw = self.detect_with_model(det, det_type, cropped_tensor, settings[0]["confidence"], settings[0]["dilation"], settings[0]["crop_factor"], settings[0]["drop_size"])
                    for seg_dict in detected_raw:
                        if settings[0]["classes"]:
                            if seg_dict.get("label", "").lower() not in [c.strip().lower() for c in settings[0]["classes"].split(",") if c.strip()]:
                                continue
                        # Convert bbox back to original image coordinates
                        crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
                        seg_x1_scaled, seg_y1_scaled, seg_x2_scaled, seg_y2_scaled = seg_dict["bbox"]
                        if scale != 1.0:
                            seg_x1_unscaled = seg_x1_scaled / scale
                            seg_y1_unscaled = seg_y1_scaled / scale
                            seg_x2_unscaled = seg_x2_scaled / scale
                            seg_y2_unscaled = seg_y2_scaled / scale
                        else:
                            seg_x1_unscaled = seg_x1_scaled
                            seg_y1_unscaled = seg_y1_scaled
                            seg_x2_unscaled = seg_x2_scaled
                            seg_y2_unscaled = seg_y2_scaled
                        abs_x1 = int(crop_x1 + seg_x1_unscaled)
                        abs_y1 = int(crop_y1 + seg_y1_unscaled)
                        abs_x2 = int(crop_x1 + seg_x2_unscaled)
                        abs_y2 = int(crop_y1 + seg_y2_unscaled)
                        abs_x1 = max(0, min(original_image_shape_hw[1], abs_x1))
                        abs_y1 = max(0, min(original_image_shape_hw[0], abs_y1))
                        abs_x2 = max(0, min(original_image_shape_hw[1], abs_x2))
                        abs_y2 = max(0, min(original_image_shape_hw[0], abs_y2))
                        seg_dict["bbox"] = (abs_x1, abs_y1, abs_x2, abs_y2)
                        seg_dict["crop_region"] = self.calculate_crop_region(seg_dict["bbox"], original_image_shape_hw, settings[0]["crop_factor"])
                        seg_dict["orig_shape"] = original_image_shape_hw
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                        else:
                            cr = seg_dict["crop_region"]
                            seg_dict["cropped_mask"] = np.zeros((cr[3]-cr[1], cr[2]-cr[0]), dtype=np.float32)
                        results["stage1"].append(seg_dict)
                        stage_assignments["stage1"].append(0)
            
            # Run Stage 2 on THIS SAME segment (independent of Stage 1 results!)
            if enabled_stages[1] and detectors_info[1] is not None and should_process_stage2:
                det, det_type, _ = detectors_info[1]
                if det is not None:
                    if settings[1]["scale_mode"] == "fixed":
                        scaled_crop, scale = self.resize_image(cropped, settings[1]["target_size"], settings[1]["max_size"])
                        cropped_tensor = self.np_to_tensor(scaled_crop)
                    else:
                        cropped_tensor = self.np_to_tensor(cropped)
                        scale = 1.0
                    detected_raw = self.detect_with_model(det, det_type, cropped_tensor, settings[1]["confidence"], settings[1]["dilation"], settings[1]["crop_factor"], settings[1]["drop_size"])
                    for seg_dict in detected_raw:
                        if settings[1]["classes"]:
                            if seg_dict.get("label", "").lower() not in [c.strip().lower() for c in settings[1]["classes"].split(",") if c.strip()]:
                                continue
                        crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
                        seg_x1_scaled, seg_y1_scaled, seg_x2_scaled, seg_y2_scaled = seg_dict["bbox"]
                        if scale != 1.0:
                            seg_x1_unscaled = seg_x1_scaled / scale
                            seg_y1_unscaled = seg_y1_scaled / scale
                            seg_x2_unscaled = seg_x2_scaled / scale
                            seg_y2_unscaled = seg_y2_scaled / scale
                        else:
                            seg_x1_unscaled = seg_x1_scaled
                            seg_y1_unscaled = seg_y1_scaled
                            seg_x2_unscaled = seg_x2_scaled
                            seg_y2_unscaled = seg_y2_scaled
                        abs_x1 = int(crop_x1 + seg_x1_unscaled)
                        abs_y1 = int(crop_y1 + seg_y1_unscaled)
                        abs_x2 = int(crop_x1 + seg_x2_unscaled)
                        abs_y2 = int(crop_y1 + seg_y2_unscaled)
                        abs_x1 = max(0, min(original_image_shape_hw[1], abs_x1))
                        abs_y1 = max(0, min(original_image_shape_hw[0], abs_y1))
                        abs_x2 = max(0, min(original_image_shape_hw[1], abs_x2))
                        abs_y2 = max(0, min(original_image_shape_hw[0], abs_y2))
                        seg_dict["bbox"] = (abs_x1, abs_y1, abs_x2, abs_y2)
                        seg_dict["crop_region"] = self.calculate_crop_region(seg_dict["bbox"], original_image_shape_hw, settings[1]["crop_factor"])
                        seg_dict["orig_shape"] = original_image_shape_hw
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                        else:
                            cr = seg_dict["crop_region"]
                            seg_dict["cropped_mask"] = np.zeros((cr[3]-cr[1], cr[2]-cr[0]), dtype=np.float32)
                        results["stage2"].append(seg_dict)
                        stage_assignments["stage2"].append(1)
            
            # Run Stage 3 on THIS SAME segment (independent of Stage 1/2 results!)
            if enabled_stages[2] and detectors_info[2] is not None and should_process_stage3:
                det, det_type, _ = detectors_info[2]
                if det is not None:
                    if settings[2]["scale_mode"] == "fixed":
                        scaled_crop, scale = self.resize_image(cropped, settings[2]["target_size"], settings[2]["max_size"])
                        cropped_tensor = self.np_to_tensor(scaled_crop)
                    else:
                        cropped_tensor = self.np_to_tensor(cropped)
                        scale = 1.0
                    detected_raw = self.detect_with_model(det, det_type, cropped_tensor, settings[2]["confidence"], settings[2]["dilation"], settings[2]["crop_factor"], settings[2]["drop_size"])
                    for seg_dict in detected_raw:
                        if settings[2]["classes"]:
                            if seg_dict.get("label", "").lower() not in [c.strip().lower() for c in settings[2]["classes"].split(",") if c.strip()]:
                                continue
                        crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
                        seg_x1_scaled, seg_y1_scaled, seg_x2_scaled, seg_y2_scaled = seg_dict["bbox"]
                        abs_x1 = int(crop_x1 + seg_x1_scaled)
                        abs_y1 = int(crop_y1 + seg_y1_scaled)
                        abs_x2 = int(crop_x1 + seg_x2_scaled)
                        abs_y2 = int(crop_y1 + seg_y2_scaled)
                        abs_x1 = max(0, min(original_image_shape_hw[1], abs_x1))
                        abs_y1 = max(0, min(original_image_shape_hw[0], abs_y1))
                        abs_x2 = max(0, min(original_image_shape_hw[1], abs_x2))
                        abs_y2 = max(0, min(original_image_shape_hw[0], abs_y2))
                        seg_dict["bbox"] = (abs_x1, abs_y1, abs_x2, abs_y2)
                        seg_dict["crop_region"] = self.calculate_crop_region(seg_dict["bbox"], original_image_shape_hw, settings[2]["crop_factor"])
                        seg_dict["orig_shape"] = original_image_shape_hw
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                        else:
                            cr = seg_dict["crop_region"]
                            seg_dict["cropped_mask"] = np.zeros((cr[3]-cr[1], cr[2]-cr[0]), dtype=np.float32)
                        results["stage3"].append(seg_dict)
                        stage_assignments["stage3"].append(2)
        
        # === FALLBACK LOGIC: Check if any stage found nothing and has process_empty enabled ===
        img_np = self.tensor_to_np(image)
        
        # Stage 1 fallback
        if enabled_stages[0] and detectors_info[0] is not None and stage_1_process_empty and len(results["stage1"]) == 0:
            logger.info("Stage 1: No detections on segments → fallback to full image detection")
            det, det_type, _ = detectors_info[0]
            if det is not None:
                if settings[0]["scale_mode"] == "fixed":
                    scaled_img, scale = self.resize_image(img_np, settings[0]["target_size"], settings[0]["max_size"])
                    img_tensor = self.np_to_tensor(scaled_img)
                else:
                    img_tensor = image
                    scale = 1.0
                detected_raw = self.detect_with_model(det, det_type, img_tensor, settings[0]["confidence"], settings[0]["dilation"], settings[0]["crop_factor"], settings[0]["drop_size"])
                for seg_dict in detected_raw:
                    if settings[0]["classes"]:
                        if seg_dict.get("label", "").lower() not in [c.strip().lower() for c in settings[0]["classes"].split(",") if c.strip()]:
                            continue
                    if scale != 1.0:
                        orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                        orig_crop_region = self.calculate_crop_region(orig_bbox, original_image_shape_hw, settings[0]["crop_factor"])
                        seg_dict["bbox"] = orig_bbox
                        seg_dict["crop_region"] = orig_crop_region
                        seg_dict["orig_shape"] = original_image_shape_hw
                    if include_masks_in_output:
                        seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                    else:
                        cr = seg_dict["crop_region"]
                        seg_dict["cropped_mask"] = np.zeros((cr[3]-cr[1], cr[2]-cr[0]), dtype=np.float32)
                    results["stage1"].append(seg_dict)
                    stage_assignments["stage1"].append(0)
        
        # Stage 2 fallback
        if enabled_stages[1] and detectors_info[1] is not None and stage_2_process_empty and len(results["stage2"]) == 0:
            logger.info("Stage 2: No detections on segments → fallback to full image detection")
            det, det_type, _ = detectors_info[1]
            if det is not None:
                if settings[1]["scale_mode"] == "fixed":
                    scaled_img, scale = self.resize_image(img_np, settings[1]["target_size"], settings[1]["max_size"])
                    img_tensor = self.np_to_tensor(scaled_img)
                else:
                    img_tensor = image
                    scale = 1.0
                detected_raw = self.detect_with_model(det, det_type, img_tensor, settings[1]["confidence"], settings[1]["dilation"], settings[1]["crop_factor"], settings[1]["drop_size"])
                for seg_dict in detected_raw:
                    if settings[1]["classes"]:
                        if seg_dict.get("label", "").lower() not in [c.strip().lower() for c in settings[1]["classes"].split(",") if c.strip()]:
                            continue
                    if scale != 1.0:
                        orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                        orig_crop_region = self.calculate_crop_region(orig_bbox, original_image_shape_hw, settings[1]["crop_factor"])
                        seg_dict["bbox"] = orig_bbox
                        seg_dict["crop_region"] = orig_crop_region
                        seg_dict["orig_shape"] = original_image_shape_hw
                    if include_masks_in_output:
                        seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                    else:
                        cr = seg_dict["crop_region"]
                        seg_dict["cropped_mask"] = np.zeros((cr[3]-cr[1], cr[2]-cr[0]), dtype=np.float32)
                    results["stage2"].append(seg_dict)
                    stage_assignments["stage2"].append(1)
        
        # Stage 3 fallback
        if enabled_stages[2] and detectors_info[2] is not None and stage_3_process_empty and len(results["stage3"]) == 0:
            logger.info("Stage 3: No detections on segments → fallback to full image detection")
            det, det_type, _ = detectors_info[2]
            if det is not None:
                if settings[2]["scale_mode"] == "fixed":
                    scaled_img, scale = self.resize_image(img_np, settings[2]["target_size"], settings[2]["max_size"])
                    img_tensor = self.np_to_tensor(scaled_img)
                else:
                    img_tensor = image
                    scale = 1.0
                detected_raw = self.detect_with_model(det, det_type, img_tensor, settings[2]["confidence"], settings[2]["dilation"], settings[2]["crop_factor"], settings[2]["drop_size"])
                for seg_dict in detected_raw:
                    if settings[2]["classes"]:
                        if seg_dict.get("label", "").lower() not in [c.strip().lower() for c in settings[2]["classes"].split(",") if c.strip()]:
                            continue
                    if scale != 1.0:
                        orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                        orig_crop_region = self.calculate_crop_region(orig_bbox, original_image_shape_hw, settings[2]["crop_factor"])
                        seg_dict["bbox"] = orig_bbox
                        seg_dict["crop_region"] = orig_crop_region
                        seg_dict["orig_shape"] = original_image_shape_hw
                    if include_masks_in_output:
                        seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                    else:
                        cr = seg_dict["crop_region"]
                        seg_dict["cropped_mask"] = np.zeros((cr[3]-cr[1], cr[2]-cr[0]), dtype=np.float32)
                    results["stage3"].append(seg_dict)
                    stage_assignments["stage3"].append(2)
        
        # Combine all stages
        all_segs = results["stage1"] + results["stage2"] + results["stage3"]
        for _ in results["stage1"]:
            stage_assignments["combined"].append(0)
        for _ in results["stage2"]:
            stage_assignments["combined"].append(1)
        for _ in results["stage3"]:
            stage_assignments["combined"].append(2)
        results["combined"] = self.apply_nms(all_segs, settings[0]["iou_threshold"])
        return results, stage_assignments
    
    def process_parallel(self, image: torch.Tensor, initial_segs: Optional[List[Dict]], enabled_stages: List[bool], detectors_info: List, settings: List[Dict], include_masks_in_output: bool, original_image_shape_hw: Tuple[int, int], original_image_np: np.ndarray, simplify_masks: bool, kernel_size: int, iterations: int, stage_1_process_empty: bool, stage_2_process_empty: bool, stage_3_process_empty: bool, stage1_input_filter_labels="", stage2_input_filter_labels="", stage3_input_filter_labels="") -> Dict[str, List[Dict]]:
        # В режиме parallel фильтрация входов не применима — все стадии работают независимо на полном изображении
        if stage1_input_filter_labels or stage2_input_filter_labels or stage3_input_filter_labels:
            logger.warning("Input label filtering (stageX_input_filter_labels) is IGNORED in 'parallel' mode. Use 'sequential' mode for input filtering.")
        results = {"stage1": [], "stage2": [], "stage3": [], "combined": []}
        stage_assignments = {"stage1": [], "stage2": [], "stage3": [], "combined": []}
        img_np = self.tensor_to_np(image)
        for i in range(3):
            if enabled_stages[i] and detectors_info[i] is not None:
                det, det_type, _ = detectors_info[i]
                if det is not None:
                    if settings[i]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(img_np, settings[i]["target_size"], settings[i]["max_size"])
                        img_tensor = self.np_to_tensor(scaled_img)
                    else:
                        img_tensor = image
                    scale = 1.0
                    detected_raw = self.detect_with_model(det, det_type, img_tensor, settings[i]["confidence"], settings[i]["dilation"], settings[i]["crop_factor"], settings[i]["drop_size"])
                    detected = []
                    for seg_dict in detected_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                            orig_crop_region = self.calculate_crop_region(orig_bbox, original_image_shape_hw, settings[i]["crop_factor"])
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                        else:
                            crop_region = seg_dict["crop_region"]
                            h = crop_region[3] - crop_region[1]
                            w = crop_region[2] - crop_region[0]
                            seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                        detected.append(seg_dict)
                    if i == 0:
                        stage_assignments["stage1"].append(0)
                    elif i == 1:
                        stage_assignments["stage2"].append(1)
                    elif i == 2:
                        stage_assignments["stage3"].append(2)
                    if settings[i]["classes"]:
                        detected = self.filter_by_classes(detected, settings[i]["classes"])
                    detected = self.apply_nms(detected, settings[i]["iou"])
                    if i == 0:
                        results["stage1"] = detected
                    elif i == 1:
                        results["stage2"] = detected
                    elif i == 2:
                        results["stage3"] = detected
        all_segs = results["stage1"] + results["stage2"] + results["stage3"]
        for _ in results["stage1"]:
            stage_assignments["combined"].append(0)
        for _ in results["stage2"]:
            stage_assignments["combined"].append(1)
        for _ in results["stage3"]:
            stage_assignments["combined"].append(2)
        results["combined"] = self.apply_nms(all_segs, settings[0]["iou_threshold"])
        return results, stage_assignments

    def process(self, image: torch.Tensor, mode: str, target_size: int, max_size: int, iou_threshold: float, include_masks_in_output: bool, simplify_masks: bool, simplify_kernel_size: int, simplify_iterations: int, segs_input=None, stage_1_enabled=True, stage_1_detector_type="bbox", stage_1_bbox_detector=None, stage_1_segm_detector=None, stage_1_confidence=0.25, stage_1_iou_threshold=0.45, stage_1_dilation=0, stage_1_classes="", stage_1_crop_factor=3.0, stage_1_scale_mode="bbox", stage_1_target_size=640, stage_1_max_size=1024, stage_1_process_empty=False, stage1_input_filter_labels="", min_confidence=0.0, min_bbox_width=1, min_bbox_height=1, stage_2_enabled=True, stage_2_detector_type="bbox", stage_2_bbox_detector=None, stage_2_segm_detector=None, stage_2_confidence=0.25, stage_2_iou_threshold=0.45, stage_2_dilation=0, stage_2_classes="", stage_2_crop_factor=3.0, stage_2_scale_mode="bbox", stage_2_target_size=640, stage_2_max_size=1024, stage_2_process_empty=False, stage2_input_filter_labels="", stage_3_enabled=True, stage_3_detector_type="bbox", stage_3_bbox_detector=None, stage_3_segm_detector=None, stage_3_confidence=0.25, stage_3_iou_threshold=0.45, stage_3_dilation=0, stage_3_classes="", stage_3_crop_factor=3.0, stage_3_scale_mode="bbox", stage_3_target_size=640, stage_3_max_size=1024, stage_3_process_empty=False, stage3_input_filter_labels="", drop_size=1, extra_pnginfo=None, prompt=None):
        
        logger.info(f"Starting Cascade Detector (Parallel-Per-Segment) in mode: {mode}")
        if not self.IMPACT_AVAILABLE:
            empty_img = torch.zeros_like(image)
            return (None, self.create_preview_image_staged(image, [], []), self.create_cropped_fragments_image(image, [], padding=10), None, None, None, empty_img, image)

        if stage_1_enabled:
            if (stage_1_detector_type == "bbox" and stage_1_segm_detector is not None) or (stage_1_detector_type == "segm" and stage_1_bbox_detector is not None):
                raise Exception(f"Stage 1: Detector type mismatch")
        if stage_2_enabled:
            if (stage_2_detector_type == "bbox" and stage_2_segm_detector is not None) or (stage_2_detector_type == "segm" and stage_2_bbox_detector is not None):
                raise Exception(f"Stage 2: Detector type mismatch")
        if stage_3_enabled:
            if (stage_3_detector_type == "bbox" and stage_3_segm_detector is not None) or (stage_3_detector_type == "segm" and stage_3_bbox_detector is not None):
                raise Exception(f"Stage 3: Detector type mismatch")

        det_1 = stage_1_bbox_detector if stage_1_detector_type == "bbox" else stage_1_segm_detector
        det_2 = stage_2_bbox_detector if stage_2_detector_type == "bbox" else stage_2_segm_detector
        det_3 = stage_3_bbox_detector if stage_3_detector_type == "bbox" else stage_3_segm_detector

        enabled_stages = [stage_1_enabled, stage_2_enabled, stage_3_enabled]
        detectors_info = [
            (det_1, stage_1_detector_type, f"stage_1_{stage_1_detector_type}_detector") if det_1 is not None else None,
            (det_2, stage_2_detector_type, f"stage_2_{stage_2_detector_type}_detector") if det_2 is not None else None,
            (det_3, stage_3_detector_type, f"stage_3_{stage_3_detector_type}_detector") if det_3 is not None else None,
        ]

        settings = [
            {"scale_mode": stage_1_scale_mode, "target_size": stage_1_target_size, "max_size": stage_1_max_size, "confidence": stage_1_confidence, "iou": stage_1_iou_threshold, "iou_threshold": iou_threshold, "dilation": stage_1_dilation, "classes": stage_1_classes, "crop_factor": stage_1_crop_factor, "drop_size": drop_size},
            {"scale_mode": stage_2_scale_mode, "target_size": stage_2_target_size, "max_size": stage_2_max_size, "confidence": stage_2_confidence, "iou": stage_2_iou_threshold, "iou_threshold": iou_threshold, "dilation": stage_2_dilation, "classes": stage_2_classes, "crop_factor": stage_2_crop_factor, "drop_size": drop_size},
            {"scale_mode": stage_3_scale_mode, "target_size": stage_3_target_size, "max_size": stage_3_max_size, "confidence": stage_3_confidence, "iou": stage_3_iou_threshold, "iou_threshold": iou_threshold, "dilation": stage_3_dilation, "classes": stage_3_classes, "crop_factor": stage_3_crop_factor, "drop_size": drop_size},
        ]

        initial_segs = None
        if segs_input is not None and self.IMPACT_AVAILABLE and isinstance(segs_input, tuple) and len(segs_input) == 2:
            _, segs_list_impact = segs_input
            initial_segs = []
            for seg_impact in segs_list_impact:
                if hasattr(seg_impact, "bbox"):
                    initial_segs.append({
                        "bbox": seg_impact.bbox,
                        "crop_region": getattr(seg_impact, "crop_region", seg_impact.bbox),
                        "label": getattr(seg_impact, "label", "object"),
                        "confidence": getattr(seg_impact, "confidence", 0.5),
                        "cropped_mask": getattr(seg_impact, "cropped_mask", None),
                        "orig_shape": segs_input[0],
                    })

        original_shape_hw = image.shape[1:3]
        original_image_np = self.tensor_to_np(image)

        if mode == "sequential":
            results, stage_assignments = self.process_sequential(
                image, initial_segs, enabled_stages, detectors_info, settings,
                include_masks_in_output, original_shape_hw, original_image_np,
                simplify_masks, simplify_kernel_size, simplify_iterations,
                stage_1_process_empty, stage_2_process_empty, stage_3_process_empty,
                stage1_input_filter_labels, stage2_input_filter_labels, stage3_input_filter_labels  # NEW: pass filters
            )
        elif mode == "parallel":
            results, stage_assignments = self.process_parallel(
                image, initial_segs, enabled_stages, detectors_info, settings,
                include_masks_in_output, original_shape_hw, original_image_np,
                simplify_masks, simplify_kernel_size, simplify_iterations,
                stage_1_process_empty, stage_2_process_empty, stage_3_process_empty,
                stage1_input_filter_labels, stage2_input_filter_labels, stage3_input_filter_labels  # NEW: pass filters (will be ignored with warning)
            )
        elif mode == "parallel_per_segment":  # NEW MODE
            results, stage_assignments = self.process_parallel_per_segment(
                image, initial_segs, enabled_stages, detectors_info, settings,
                include_masks_in_output, original_shape_hw, original_image_np,
                simplify_masks, simplify_kernel_size, simplify_iterations,
                stage_1_process_empty, stage_2_process_empty, stage_3_process_empty,
                stage1_input_filter_labels, stage2_input_filter_labels, stage3_input_filter_labels
            )
        def apply_filters(segs_list):
            filtered = []
            for seg in segs_list:
                if seg.get("confidence", 0) < min_confidence:
                    continue
                x1, y1, x2, y2 = seg["bbox"]
                width = x2 - x1
                height = y2 - y1
                if width < min_bbox_width or height < min_bbox_height:
                    continue
                filtered.append(seg)
            return filtered

        results["stage1"] = apply_filters(results["stage1"])
        results["stage2"] = apply_filters(results["stage2"])
        results["stage3"] = apply_filters(results["stage3"])
        results["combined"] = apply_filters(results["combined"])

        image_shape_wh = (image.shape[2], image.shape[1])
        segs_output = self.convert_to_segs_format(results["combined"], image_shape_wh)
        segs_stage1 = self.convert_to_segs_format(results["stage1"], image_shape_wh)
        segs_stage2 = self.convert_to_segs_format(results["stage2"], image_shape_wh)
        segs_stage3 = self.convert_to_segs_format(results["stage3"], image_shape_wh)

        preview_image = self.create_preview_image_staged(image, results["combined"], stage_assignments["combined"])
        cropped_fragments_image = self.create_cropped_fragments_image(image, results["combined"], padding=10)
        masked_fragments_image = self.create_masked_fragments_image(image, results["combined"], padding=10) if include_masks_in_output else torch.zeros_like(image)

        count_out = len(results["combined"]) if segs_output else 0
        num_enabled = sum(enabled_stages)
        num_with_process_empty = sum([1 for i in range(3) if enabled_stages[i] and [stage_1_process_empty, stage_2_process_empty, stage_3_process_empty][i]])
        image_bypass = image if count_out == 0 and num_enabled == num_with_process_empty else (image if count_out > 0 else torch.zeros_like(image))

        logger.info(f"Output counts - Combined SEGS: {count_out}")
        return (segs_output, preview_image, cropped_fragments_image, segs_stage1, segs_stage2, segs_stage3, masked_fragments_image, image_bypass)

NODE_CLASS_MAPPINGS = {"CascadeDetectorAdvanced": CascadeDetector}
NODE_DISPLAY_NAME_MAPPINGS = {"CascadeDetectorAdvanced": "🎯 Cascade Detector Advanced (Input Filter)"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
