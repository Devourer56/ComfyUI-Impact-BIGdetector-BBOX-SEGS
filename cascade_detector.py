import torch
import numpy as np
import math
import logging
from typing import List, Tuple, Dict, Any, Optional
from nodes import MAX_RESOLUTION
import folder_paths
import comfy
import comfy.utils
from PIL import Image, ImageDraw, ImageFont
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("CascadeDetector script loaded (FIXED VERSION).")

class CascadeDetector:
    @classmethod
    def INPUT_TYPES(cls):
        detector_options = ["bbox", "segm"]
        scale_modes = ["bbox", "crop_region", "fixed"]
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["sequential", "parallel"], {"default": "sequential"}),
                "target_size": (
                    "INT",
                    {"default": 640, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "max_size": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "iou_threshold": (
                    "FLOAT",
                    {"default": 0.50, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "include_masks_in_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If True, recalculates masks for all stages to match crop_region dimensions (required for SEGS Preview compatibility). If False, masks are set to None to prevent errors.",
                    },
                ),
                "simplify_masks": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If True, applies morphological closing to smooth recalculated masks (only active if include_masks_in_output is True).",
                    },
                ),
                "simplify_kernel_size": (
                    "INT",
                    {"default": 5, "min": 1, "max": 21, "step": 2},
                ),
                "simplify_iterations": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
            },
            "optional": {
                "segs_input": (
                    "SEGS",
                    {
                        "tooltip": "Used only in 'sequential' mode. Provides initial segments to start the cascade. Ignored in 'parallel' mode. Recommended to disable Stage 1 when using this.",
                    },
                ),
                # Stage 1 Parameters
                "stage_1_enabled": ("BOOLEAN", {"default": True}),
                "stage_1_detector_type": (detector_options, {"default": "bbox"}),
                "stage_1_bbox_detector": (
                    "BBOX_DETECTOR",
                    {"tooltip": "BBOX detector for Stage 1"},
                ),
                "stage_1_segm_detector": (
                    "SEGM_DETECTOR",
                    {"tooltip": "SEGM detector for Stage 1"},
                ),
                "stage_1_confidence": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "stage_1_iou_threshold": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "stage_1_dilation": (
                    "INT",
                    {"default": 0, "min": -512, "max": 512, "step": 1},
                ),
                "stage_1_classes": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Class filters for Stage 1, comma-separated",
                    },
                ),
                "stage_1_crop_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                "stage_1_scale_mode": (scale_modes, {"default": "bbox"}),
                "stage_1_target_size": (
                    "INT",
                    {"default": 640, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "stage_1_max_size": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "stage_1_process_empty": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If True and Stage 1 receives no input segments, it will run its detector on the full input image.",
                    },
                ),
                # Min Confidence and BBox Size for Final Output
                "min_confidence": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "min_bbox_width": (
                    "INT",
                    {"default": 1, "min": 1, "max": MAX_RESOLUTION, "step": 1},
                ),
                "min_bbox_height": (
                    "INT",
                    {"default": 1, "min": 1, "max": MAX_RESOLUTION, "step": 1},
                ),
                # Stage 2 Parameters
                "stage_2_enabled": ("BOOLEAN", {"default": True}),
                "stage_2_detector_type": (detector_options, {"default": "bbox"}),
                "stage_2_bbox_detector": (
                    "BBOX_DETECTOR",
                    {"tooltip": "BBOX detector for Stage 2"},
                ),
                "stage_2_segm_detector": (
                    "SEGM_DETECTOR",
                    {"tooltip": "SEGM detector for Stage 2"},
                ),
                "stage_2_confidence": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "stage_2_iou_threshold": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "stage_2_dilation": (
                    "INT",
                    {"default": 0, "min": -512, "max": 512, "step": 1},
                ),
                "stage_2_classes": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Class filters for Stage 2, comma-separated",
                    },
                ),
                "stage_2_crop_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                "stage_2_scale_mode": (scale_modes, {"default": "bbox"}),
                "stage_2_target_size": (
                    "INT",
                    {"default": 640, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "stage_2_max_size": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "stage_2_process_empty": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If True and Stage 2 receives no input segments from Stage 1, it will run its detector on the full input image.",
                    },
                ),
                # Stage 3 Parameters
                "stage_3_enabled": ("BOOLEAN", {"default": True}),
                "stage_3_detector_type": (detector_options, {"default": "bbox"}),
                "stage_3_bbox_detector": (
                    "BBOX_DETECTOR",
                    {"tooltip": "BBOX detector for Stage 3"},
                ),
                "stage_3_segm_detector": (
                    "SEGM_DETECTOR",
                    {"tooltip": "SEGM detector for Stage 3"},
                ),
                "stage_3_confidence": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "stage_3_iou_threshold": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "stage_3_dilation": (
                    "INT",
                    {"default": 0, "min": -512, "max": 512, "step": 1},
                ),
                "stage_3_classes": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Class filters for Stage 3, comma-separated",
                    },
                ),
                "stage_3_crop_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                "stage_3_scale_mode": (scale_modes, {"default": "bbox"}),
                "stage_3_target_size": (
                    "INT",
                    {"default": 640, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "stage_3_max_size": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "stage_3_process_empty": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If True and Stage 3 receives no input segments from Stage 2, it will run its detector on the full input image.",
                    },
                ),
                "drop_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            }
        }

    RETURN_TYPES = (
        "SEGS",  # segs_output_all_stages
        "IMAGE",  # preview_image
        "IMAGE",  # cropped_fragments_image
        "SEGS",  # stage1_segs
        "SEGS",  # stage2_segs
        "SEGS",  # stage3_segs
        "IMAGE",  # masked_fragments_image
        "IMAGE",  # image_bypass
    )
    RETURN_NAMES = (
        "segs_output_all_stages (Combined)",
        "preview_image (Combined Detections)",
        "cropped_fragments_image (All Detections)",
        "stage1_segs (Stage 1 Results)",
        "stage2_segs (Stage 2 Results)",
        "stage3_segs (Stage 3 Results)",
        "masked_fragments_image (Masked Fragments)",
        "image_bypass (Original if No Detections)",
    )
    FUNCTION = "process"
    CATEGORY = "Detection/Cascade"
    DESCRIPTION = """Cascaded detector for ComfyUI. Supports sequential and parallel processing with bbox/segm models. Requires Impact Pack.
Fixed version: Masks are ALWAYS recalculated to match crop_region dimensions after detection in ALL execution paths, ensuring compatibility with SEGS Preview and other Impact Pack nodes.
Use 'segs_input' only in 'sequential' mode to start the cascade (recommended with Stage 1 disabled).
The main output for further processing (e.g., SEGSPaste, SEGSDetailer) is 'segs_output_all_stages', which combines results from all stages.
Includes checks for detector type mismatches, per-stage crop factors, filtering by confidence/BBox size, and an image bypass output if no detections are found.
Per-stage 'process_empty' toggles allow a stage to run its detector on the full input image if it receives no segments from the previous stage."""

    def __init__(self):
        self.device = comfy.model_management.get_torch_device()
        self.stage_results = {}
        self.IMPACT_AVAILABLE = self._check_impact_availability_at_init()

    def _check_impact_availability_at_init(self):
        try:
            from impact.core import SEG
            self.SEG_IMPACT = SEG
            logger.info("Impact Pack core components imported successfully at init.")
            return True
        except ImportError as e:
            logger.error(f"Impact Pack core not available at init: {e}")
            return False

    def tensor_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        return tensor.cpu().numpy()

    def np_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        if array.dtype != np.float32:
            array = array.astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def resize_image(
        self, image: np.ndarray, target_size: int, max_size: int
    ) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        scale = target_size / min(h, w)
        if scale * max(h, w) > max_size:
            scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale

    def calculate_crop_region(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
        crop_factor: float = 3.0,
    ) -> Tuple[int, int, int, int]:
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

    def crop_image(
        self, image: np.ndarray, region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        x1, y1, x2, y2 = region
        return image[y1:y2, x1:x2]

    def filter_by_classes(self, segs: List[Dict], classes_filter: str) -> List[Dict]:
        if not classes_filter or classes_filter.strip() == "":
            return segs
        allowed_classes = [
            c.strip().lower() for c in classes_filter.split(",") if c.strip()
        ]
        filtered = []
        for seg in segs:
            if seg.get("label", "").lower() in allowed_classes:
                filtered.append(seg)
        return filtered

    def apply_nms(self, segs: List[Dict], iou_threshold: float) -> List[Dict]:
        if not segs:
            return []
        segs.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        selected = []
        while segs:
            current = segs.pop(0)
            selected.append(current)
            segs = [
                seg
                for seg in segs
                if self.calculate_iou(current["bbox"], seg["bbox"]) < iou_threshold
            ]
        return selected

    def calculate_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def transform_coordinates(
        self,
        seg: Dict,
        parent_bbox: Tuple[int, int, int, int],
        scale: float,
        image_shape_hw: Tuple[int, int],
    ) -> Dict:
        crop_x1, crop_y1, _, _ = parent_bbox
        seg_x1, seg_y1, seg_x2, seg_y2 = seg["bbox"]
        orig_x1 = crop_x1 + int(seg_x1 / scale)
        orig_y1 = crop_y1 + int(seg_y1 / scale)
        orig_x2 = crop_x1 + int(seg_x2 / scale)
        orig_y2 = crop_y1 + int(seg_y2 / scale)
        orig_x1 = max(0, min(image_shape_hw[1], orig_x1))
        orig_y1 = max(0, min(image_shape_hw[0], orig_y1))
        orig_x2 = max(0, min(image_shape_hw[1], orig_x2))
        orig_y2 = max(0, min(image_shape_hw[0], orig_y2))
        seg["bbox"] = (orig_x1, orig_y1, orig_x2, orig_y2)
        seg["crop_region"] = self.calculate_crop_region(seg["bbox"], image_shape_hw)
        return seg

    # --- FIXED: recalculate_mask function - ALWAYS returns mask matching crop_region dimensions ---
    def recalculate_mask(self, original_image_np: np.ndarray, seg_result: Dict, original_image_shape_hw: Tuple[int, int], simplify_masks: bool, kernel_size: int, iterations: int):
        """
        FIXED VERSION: Always returns a mask matching the crop_region dimensions.
        Never returns None - creates empty mask if needed to prevent SEGS Preview errors.
        """
        crop_region = seg_result["crop_region"]
        x1_cr, y1_cr, x2_cr, y2_cr = crop_region
        
        # Clamp crop region to image bounds
        x1_cr = max(0, min(original_image_shape_hw[1], x1_cr))
        y1_cr = max(0, min(original_image_shape_hw[0], y1_cr))
        x2_cr = max(0, min(original_image_shape_hw[1], x2_cr))
        y2_cr = max(0, min(original_image_shape_hw[0], y2_cr))
        
        # Calculate fragment dimensions from crop_region (CRITICAL FIX)
        fragment_h = y2_cr - y1_cr
        fragment_w = x2_cr - x1_cr
        
        # Create full-size mask initialized to zeros
        full_mask = np.zeros(original_image_shape_hw, dtype=np.uint8)
        
        old_cropped_mask = seg_result.get("cropped_mask")
        if old_cropped_mask is not None and old_cropped_mask.size > 0:
            # Handle potential batch dimension
            if old_cropped_mask.ndim == 3 and old_cropped_mask.shape[0] == 1:
                old_cropped_mask = old_cropped_mask[0]
            elif old_cropped_mask.ndim > 2:
                old_cropped_mask = old_cropped_mask.max(axis=0) if old_cropped_mask.ndim == 3 else old_cropped_mask
            
            # Ensure proper dtype for cv2.resize
            if old_cropped_mask.dtype == np.float32:
                if old_cropped_mask.max() > 1.0:
                    old_cropped_mask = (old_cropped_mask * 255).astype(np.uint8)
                else:
                    old_cropped_mask = (old_cropped_mask * 255).astype(np.uint8)
            elif old_cropped_mask.dtype != np.uint8:
                old_cropped_mask = old_cropped_mask.astype(np.uint8)
            
            # Resize mask to fragment dimensions if needed
            if old_cropped_mask.shape[:2] != (fragment_h, fragment_w):
                resized_mask = cv2.resize(old_cropped_mask, (fragment_w, fragment_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized_mask = old_cropped_mask
            
            # Apply morphological simplification if requested
            if simplify_masks:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                resized_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
            # Paste resized mask into full_mask at crop_region location
            if resized_mask.shape[:2] == (fragment_h, fragment_w):
                full_mask[y1_cr:y2_cr, x1_cr:x2_cr] = resized_mask
            else:
                logger.warning(f"Mask shape mismatch after resize: {resized_mask.shape} vs fragment {(fragment_h, fragment_w)}")
        
        # ALWAYS extract mask for the crop_region (never return None)
        new_cropped_mask = full_mask[y1_cr:y2_cr, x1_cr:x2_cr].astype(np.float32) / 255.0
        
        # Safety check: ensure mask has correct dimensions
        if new_cropped_mask.shape[:2] != (fragment_h, fragment_w):
            logger.warning(f"Creating fallback empty mask for crop_region {(fragment_w, fragment_h)}")
            new_cropped_mask = np.zeros((fragment_h, fragment_w), dtype=np.float32)
        
        seg_result["cropped_mask"] = new_cropped_mask
        return seg_result

    def detect_with_model(
        self,
        detector,
        detector_type: str,
        image_tensor: torch.Tensor,
        confidence: float,
        dilation: int,
        crop_factor: float,
        drop_size: int,
    ) -> List[Dict]:
        logger.debug(
            f"Attempting detection with {detector_type} detector, confidence={confidence}, dilation={dilation}"
        )
        if not self.IMPACT_AVAILABLE or detector is None:
            logger.warning(
                f"Impact Pack available (at init): {self.IMPACT_AVAILABLE}, Detector is None: {detector is None}"
            )
            return []
        try:
            shape, segs_list_impact = detector.detect(
                image_tensor,
                confidence,
                dilation,
                crop_factor,
                drop_size,
                detailer_hook=None,
            )
            unified_segs = []
            for seg_impact in segs_list_impact:
                if seg_impact is None:
                    continue
                if not hasattr(seg_impact, "bbox") or not hasattr(seg_impact, "confidence"):
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
                unified_segs.append(
                    {
                        "bbox": bbox,
                        "crop_region": getattr(seg_impact, "crop_region", bbox),
                        "label": getattr(seg_impact, "label", "object"),
                        "confidence": confidence_val,
                        "cropped_mask": cropped_mask,
                        "orig_shape": shape,
                    }
                )
            return unified_segs
        except Exception as e:
            logger.error(f"Error during {detector_type} detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def create_preview_image(
        self,
        image: torch.Tensor,
        segs_list: List[Dict],
        color: Tuple[int, int, int, int] = (255, 0, 0, 255),
    ) -> torch.Tensor:
        img_np = self.tensor_to_np(image)
        if img_np.ndim == 3 and img_np.shape[2] in [3, 4]:
            img_pil = Image.fromarray((img_np[:, :, :3] * 255).astype(np.uint8))
        else:
            img_pil = Image.fromarray((img_np[..., :3] * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_pil, mode="RGBA")
        for seg in segs_list:
            x1, y1, x2, y2 = seg["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=10)
            label = f"{seg.get('label', 'obj')}: {seg.get('confidence', 0):.2f}"
            draw.text((x1, y1 - 10), label, fill=color)
        preview_np = np.array(img_pil).astype(np.float32) / 255.0
        if preview_np.ndim == 2:
            preview_np = np.stack([preview_np] * 3, axis=-1)
        return torch.from_numpy(preview_np).unsqueeze(0)

    def create_cropped_fragments_image(
        self, image_tensor: torch.Tensor, segs_list: List[Dict], padding: int = 10
    ) -> torch.Tensor:
        img_np = self.tensor_to_np(image_tensor)
        h, w, c = img_np.shape
        if c == 4:
            bg_color = (0, 0, 0, 0)
        else:
            bg_color = (0, 0, 0)
        if not segs_list:
            canvas_img = Image.new(
                "RGBA" if c == 4 else "RGB", (padding * 2, padding * 2), color=bg_color
            )
            final_np = np.array(canvas_img).astype(np.float32) / 255.0
            if final_np.ndim == 2:
                final_np = np.stack([final_np] * 3, axis=-1)
            if final_np.shape[2] == 3 and c == 4:
                final_np = np.concatenate(
                    [
                        final_np,
                        np.ones(
                            (final_np.shape[0], final_np.shape[1], 1),
                            dtype=final_np.dtype,
                        ),
                    ],
                    axis=-1,
                )
            return torch.from_numpy(final_np).unsqueeze(0)
        canvas_size = max(h, w) + 2 * padding
        canvas_img = Image.new(
            "RGBA" if c == 4 else "RGB", (canvas_size, canvas_size), color=bg_color
        )
        current_x = padding
        current_y = padding
        row_height = 0
        max_row_width = canvas_size - 2 * padding
        for i, seg in enumerate(segs_list):
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
        final_canvas = canvas_img.crop(
            (
                0,
                0,
                min(canvas_size, current_x + padding),
                min(canvas_size, current_y + row_height + padding),
            )
        )
        final_np = np.array(final_canvas).astype(np.float32) / 255.0
        if final_np.ndim == 2:
            final_np = np.stack([final_np] * 3, axis=-1)
        if final_np.shape[2] == 3 and c == 4:
            final_np = np.concatenate(
                [
                    final_np,
                    np.ones(
                        (final_np.shape[0], final_np.shape[1], 1), dtype=final_np.dtype
                    ),
                ],
                axis=-1,
            )
        return torch.from_numpy(final_np).unsqueeze(0)

    def create_masked_fragments_image(
        self, image: torch.Tensor, segs_list: List[Dict], padding: int = 10
    ) -> torch.Tensor:
        img_np = self.tensor_to_np(image)
        h, w, c = img_np.shape
        if c == 4:
            bg_color = (0, 0, 0, 0)
        else:
            bg_color = (0, 0, 0)
        if not segs_list:
            canvas_img = Image.new(
                "RGBA" if c == 4 else "RGB", (padding * 2, padding * 2), color=bg_color
            )
            final_np = np.array(canvas_img).astype(np.float32) / 255.0
            if final_np.ndim == 2:
                final_np = np.stack([final_np] * 3, axis=-1)
            if final_np.shape[2] == 3 and c == 4:
                final_np = np.concatenate(
                    [
                        final_np,
                        np.ones(
                            (final_np.shape[0], final_np.shape[1], 1),
                            dtype=final_np.dtype,
                        ),
                    ],
                    axis=-1,
                )
            return torch.from_numpy(final_np).unsqueeze(0)
        canvas_size = max(h, w) + 2 * padding
        canvas_img = Image.new(
            "RGBA" if c == 4 else "RGB", (canvas_size, canvas_size), color=bg_color
        )
        current_x = padding
        current_y = padding
        row_height = 0
        max_row_width = canvas_size - 2 * padding
        for i, seg in enumerate(segs_list):
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
            if frag_np.shape[2] == 3:
                expanded_mask = np.stack([cropped_mask_np] * 3, axis=-1)
            elif frag_np.shape[2] == 4:
                expanded_mask = np.stack([cropped_mask_np] * 4, axis=-1)
            else:
                continue
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
        final_canvas = canvas_img.crop(
            (
                0,
                0,
                min(canvas_size, current_x + padding),
                min(canvas_size, current_y + row_height + padding),
            )
        )
        final_np = np.array(final_canvas).astype(np.float32) / 255.0
        if final_np.ndim == 2:
            final_np = np.stack([final_np] * 3, axis=-1)
        if final_np.shape[2] == 3 and c == 4:
            final_np = np.concatenate(
                [
                    final_np,
                    np.ones(
                        (final_np.shape[0], final_np.shape[1], 1), dtype=final_np.dtype
                    ),
                ],
                axis=-1,
            )
        return torch.from_numpy(final_np).unsqueeze(0)

    def convert_to_segs_format(
        self, segs_list: List[Dict], image_shape_wh: Tuple[int, int]
    ) -> Optional[Tuple]:
        if not self.IMPACT_AVAILABLE:
            logger.error("Impact Pack not available. Cannot convert to SEGS format.")
            return None
        try:
            segs_objects = []
            for seg in segs_list:
                cropped_mask_tensor = seg.get("cropped_mask")
                if cropped_mask_tensor is not None:
                    if not isinstance(cropped_mask_tensor, torch.Tensor):
                        cropped_mask_tensor = torch.from_numpy(cropped_mask_tensor)
                    if cropped_mask_tensor.ndim == 2:
                        cropped_mask_tensor = cropped_mask_tensor.unsqueeze(0)
                    if cropped_mask_tensor.ndim != 3:
                        logger.warning(f"Skipping seg with incorrect mask dims: {cropped_mask_tensor.ndim}")
                        continue
                    if cropped_mask_tensor.max() > 1.0:
                        cropped_mask_tensor = cropped_mask_tensor / 255.0
                else:
                    # CRITICAL FIX: Always provide a valid mask tensor matching crop_region dimensions
                    crop_region = seg.get("crop_region", seg["bbox"])
                    h = crop_region[3] - crop_region[1]
                    w = crop_region[2] - crop_region[0]
                    cropped_mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)
                seg_obj = self.SEG_IMPACT(
                    cropped_image=None,
                    cropped_mask=cropped_mask_tensor,
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

    def process_sequential(
        self,
        image: torch.Tensor,
        initial_segs: Optional[List[Dict]],
        enabled_stages: List[bool],
        detectors_info: List,
        settings: List[Dict],
        include_masks_in_output: bool,
        original_image_shape_hw: Tuple[int, int],
        original_image_np: np.ndarray,
        simplify_masks: bool,
        kernel_size: int,
        iterations: int,
        stage_1_process_empty: bool,
        stage_2_process_empty: bool,
        stage_3_process_empty: bool,
    ) -> Dict[str, List[Dict]]:
        results = {"stage1": [], "stage2": [], "stage3": [], "combined": []}
        current_segs = initial_segs if initial_segs else []

        # Stage 1
        if enabled_stages[0] and detectors_info[0] is not None:
            logger.info("Stage 1: Detecting on full image or input segments")
            det, det_type, name = detectors_info[0]
            if det is not None:
                if not current_segs and stage_1_process_empty:
                    logger.info("Stage 1: Input empty, running detector on full image (process_empty=True)")
                    img_np = self.tensor_to_np(image)
                    if settings[0]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(
                            img_np, settings[0]["target_size"], settings[0]["max_size"]
                        )
                        img_tensor = self.np_to_tensor(scaled_img)
                    else:
                        img_tensor = image
                        scale = 1.0
                    stage1_segs_raw = self.detect_with_model(
                        det, det_type, img_tensor, settings[0]["confidence"],
                        settings[0]["dilation"], settings[0]["crop_factor"], settings[0]["drop_size"]
                    )
                    stage1_segs = []
                    for seg_dict in stage1_segs_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                            orig_crop_region = self.calculate_crop_region(
                                orig_bbox, original_image_shape_hw, settings[0]["crop_factor"]
                            )
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        
                        # === CRITICAL FIX: Recalculate mask IMMEDIATELY after detection ===
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(
                                original_image_np, seg_dict, original_image_shape_hw,
                                simplify_masks, kernel_size, iterations
                            )
                        else:
                            # Always set to empty mask (never None) to prevent SEGS Preview errors
                            crop_region = seg_dict["crop_region"]
                            h = crop_region[3] - crop_region[1]
                            w = crop_region[2] - crop_region[0]
                            seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                        
                        stage1_segs.append(seg_dict)
                    results["stage1"] = stage1_segs
                    current_segs = stage1_segs
                elif current_segs:
                    img_np = self.tensor_to_np(image)
                    if settings[0]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(
                            img_np, settings[0]["target_size"], settings[0]["max_size"]
                        )
                        img_tensor = self.np_to_tensor(scaled_img)
                    else:
                        img_tensor = image
                        scale = 1.0
                    stage1_segs_raw = self.detect_with_model(
                        det, det_type, img_tensor, settings[0]["confidence"],
                        settings[0]["dilation"], settings[0]["crop_factor"], settings[0]["drop_size"]
                    )
                    stage1_segs = []
                    for seg_dict in stage1_segs_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                            orig_crop_region = self.calculate_crop_region(
                                orig_bbox, original_image_shape_hw, settings[0]["crop_factor"]
                            )
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        
                        # === CRITICAL FIX: Recalculate mask IMMEDIATELY after detection (normal branch) ===
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(
                                original_image_np, seg_dict, original_image_shape_hw,
                                simplify_masks, kernel_size, iterations
                            )
                        else:
                            crop_region = seg_dict["crop_region"]
                            h = crop_region[3] - crop_region[1]
                            w = crop_region[2] - crop_region[0]
                            seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                        
                        stage1_segs.append(seg_dict)
                    if settings[0]["classes"]:
                        stage1_segs = self.filter_by_classes(stage1_segs, settings[0]["classes"])
                    stage1_segs = self.apply_nms(stage1_segs, settings[0]["iou"])
                    results["stage1"] = stage1_segs
                    current_segs = stage1_segs
                else:
                    logger.info("Stage 1: Input empty and process_empty=False, skipping detection")
                    results["stage1"] = []
            else:
                logger.info("Detector 1 is None, skipping Stage 1")
        else:
            logger.info("Stage 1 disabled, skipping")

        # Stage 2
        if enabled_stages[1] and detectors_info[1] is not None:
            logger.info(f"Stage 2: Processing {len(current_segs)} input segments")
            det, det_type, name = detectors_info[1]
            if det is not None:
                if not current_segs and stage_2_process_empty:
                    logger.info("Stage 2: Input empty, running detector on full image (process_empty=True)")
                    img_np = self.tensor_to_np(image)
                    if settings[1]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(
                            img_np, settings[1]["target_size"], settings[1]["max_size"]
                        )
                        img_tensor = self.np_to_tensor(scaled_img)
                    else:
                        img_tensor = image
                        scale = 1.0
                    detected_raw = self.detect_with_model(
                        det, det_type, img_tensor, settings[1]["confidence"],
                        settings[1]["dilation"], settings[1]["crop_factor"], settings[1]["drop_size"]
                    )
                    stage2_segs = []
                    for seg_dict in detected_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                            orig_crop_region = self.calculate_crop_region(
                                orig_bbox, original_image_shape_hw, settings[1]["crop_factor"]
                            )
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        
                        # === CRITICAL FIX: Recalculate mask IMMEDIATELY after detection ===
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(
                                original_image_np, seg_dict, original_image_shape_hw,
                                simplify_masks, kernel_size, iterations
                            )
                        else:
                            crop_region = seg_dict["crop_region"]
                            h = crop_region[3] - crop_region[1]
                            w = crop_region[2] - crop_region[0]
                            seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                        
                        stage2_segs.append(seg_dict)
                    results["stage2"] = stage2_segs
                    current_segs = stage2_segs
                elif current_segs:
                    logger.info(f"Stage 2: Detecting on {len(current_segs)} crops")
                    stage2_segs = []
                    img_np = self.tensor_to_np(image)
                    for i, parent_seg in enumerate(current_segs):
                        crop_region = parent_seg.get("crop_region", parent_seg["bbox"])
                        cropped = self.crop_image(img_np, crop_region)
                        if cropped.size == 0:
                            continue
                        scale = 1.0
                        if settings[1]["scale_mode"] == "fixed":
                            scaled_crop, scale = self.resize_image(
                                cropped, settings[1]["target_size"], settings[1]["max_size"]
                            )
                            cropped_tensor = self.np_to_tensor(scaled_crop)
                        elif settings[1]["scale_mode"] == "bbox":
                            bbox_in_crop = (
                                parent_seg["bbox"][0] - crop_region[0],
                                parent_seg["bbox"][1] - crop_region[1],
                                parent_seg["bbox"][2] - crop_region[0],
                                parent_seg["bbox"][3] - crop_region[1],
                            )
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
                        detected_raw = self.detect_with_model(
                            det, det_type, cropped_tensor, settings[1]["confidence"],
                            settings[1]["dilation"], settings[1]["crop_factor"], settings[1]["drop_size"]
                        )
                        for seg_dict in detected_raw:
                            if settings[1]["classes"]:
                                if seg_dict.get("label", "").lower() not in [
                                    c.strip().lower() for c in settings[1]["classes"].split(",") if c.strip()
                                ]:
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
                            seg_dict["crop_region"] = self.calculate_crop_region(
                                seg_dict["bbox"], original_image_shape_hw, settings[1]["crop_factor"]
                            )
                            seg_dict["orig_shape"] = original_image_shape_hw
                            
                            # === CRITICAL FIX: Recalculate mask IMMEDIATELY after coordinate transformation ===
                            if include_masks_in_output:
                                seg_dict = self.recalculate_mask(
                                    original_image_np, seg_dict, original_image_shape_hw,
                                    simplify_masks, kernel_size, iterations
                                )
                            else:
                                crop_region = seg_dict["crop_region"]
                                h = crop_region[3] - crop_region[1]
                                w = crop_region[2] - crop_region[0]
                                seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                            
                            stage2_segs.append(seg_dict)
                    stage2_segs = self.apply_nms(stage2_segs, settings[1]["iou"])
                    results["stage2"] = stage2_segs
                    current_segs = stage2_segs
                else:
                    logger.info("Stage 2: Input empty and process_empty=False, skipping detection")
                    results["stage2"] = []
            else:
                logger.info("Detector 2 is None, skipping Stage 2")
        else:
            logger.info("Stage 2 disabled, skipping")

        # Stage 3
        if enabled_stages[2] and detectors_info[2] is not None:
            logger.info(f"Stage 3: Processing {len(current_segs)} input segments")
            det, det_type, name = detectors_info[2]
            if det is not None:
                if not current_segs and stage_3_process_empty:
                    logger.info("Stage 3: Input empty, running detector on full image (process_empty=True)")
                    img_np = self.tensor_to_np(image)
                    if settings[2]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(
                            img_np, settings[2]["target_size"], settings[2]["max_size"]
                        )
                        img_tensor = self.np_to_tensor(scaled_img)
                    else:
                        img_tensor = image
                        scale = 1.0
                    detected_raw = self.detect_with_model(
                        det, det_type, img_tensor, settings[2]["confidence"],
                        settings[2]["dilation"], settings[2]["crop_factor"], settings[2]["drop_size"]
                    )
                    stage3_segs = []
                    for seg_dict in detected_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                            orig_crop_region = self.calculate_crop_region(
                                orig_bbox, original_image_shape_hw, settings[2]["crop_factor"]
                            )
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        
                        # === CRITICAL FIX: Recalculate mask IMMEDIATELY after detection ===
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(
                                original_image_np, seg_dict, original_image_shape_hw,
                                simplify_masks, kernel_size, iterations
                            )
                        else:
                            crop_region = seg_dict["crop_region"]
                            h = crop_region[3] - crop_region[1]
                            w = crop_region[2] - crop_region[0]
                            seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                        
                        stage3_segs.append(seg_dict)
                    results["stage3"] = stage3_segs
                    current_segs = stage3_segs
                elif current_segs:
                    logger.info(f"Stage 3: Detecting on {len(current_segs)} crops")
                    stage3_segs = []
                    img_np = self.tensor_to_np(image)
                    for i, parent_seg in enumerate(current_segs):
                        crop_region = parent_seg.get("crop_region", parent_seg["bbox"])
                        cropped = self.crop_image(img_np, crop_region)
                        if cropped.size == 0:
                            continue
                        if settings[2]["scale_mode"] == "fixed":
                            scaled_crop, scale = self.resize_image(
                                cropped, settings[2]["target_size"], settings[2]["max_size"]
                            )
                            cropped_tensor = self.np_to_tensor(scaled_crop)
                        else:
                            cropped_tensor = self.np_to_tensor(cropped)
                            scale = 1.0
                        detected_raw = self.detect_with_model(
                            det, det_type, cropped_tensor, settings[2]["confidence"],
                            settings[2]["dilation"], settings[2]["crop_factor"], settings[2]["drop_size"]
                        )
                        for seg_dict in detected_raw:
                            if settings[2]["classes"]:
                                if seg_dict.get("label", "").lower() not in [
                                    c.strip().lower() for c in settings[2]["classes"].split(",") if c.strip()
                                ]:
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
                            seg_dict["crop_region"] = self.calculate_crop_region(
                                seg_dict["bbox"], original_image_shape_hw, settings[2]["crop_factor"]
                            )
                            seg_dict["orig_shape"] = original_image_shape_hw
                            
                            # === CRITICAL FIX: Recalculate mask IMMEDIATELY after coordinate transformation ===
                            if include_masks_in_output:
                                seg_dict = self.recalculate_mask(
                                    original_image_np, seg_dict, original_image_shape_hw,
                                    simplify_masks, kernel_size, iterations
                                )
                            else:
                                crop_region = seg_dict["crop_region"]
                                h = crop_region[3] - crop_region[1]
                                w = crop_region[2] - crop_region[0]
                                seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                            
                            stage3_segs.append(seg_dict)
                    stage3_segs = self.apply_nms(stage3_segs, settings[2]["iou"])
                    results["stage3"] = stage3_segs
                    current_segs = stage3_segs
                else:
                    logger.info("Stage 3: Input empty and process_empty=False, skipping detection")
                    results["stage3"] = []
            else:
                logger.info("Detector 3 is None, skipping Stage 3")
        else:
            logger.info("Stage 3 disabled, skipping")

        # Combine all stages (NO redundant recalculation here - masks already fixed)
        all_segs = results["stage1"] + results["stage2"] + results["stage3"]
        results["combined"] = self.apply_nms(all_segs, settings[0]["iou_threshold"])
        logger.info(f"Sequential processing finished. Combined detections: {len(results['combined'])}")
        return results

    def process_parallel(
        self,
        image: torch.Tensor,
        initial_segs: Optional[List[Dict]],
        enabled_stages: List[bool],
        detectors_info: List,
        settings: List[Dict],
        include_masks_in_output: bool,
        original_image_shape_hw: Tuple[int, int],
        original_image_np: np.ndarray,
        simplify_masks: bool,
        kernel_size: int,
        iterations: int,
        stage_1_process_empty: bool,
        stage_2_process_empty: bool,
        stage_3_process_empty: bool,
    ) -> Dict[str, List[Dict]]:
        results = {"stage1": [], "stage2": [], "stage3": [], "combined": []}
        img_np = self.tensor_to_np(image)
        for i in range(3):
            if enabled_stages[i] and detectors_info[i] is not None:
                det, det_type, name = detectors_info[i]
                if det is not None:
                    logger.info(f"Parallel processing with model {i + 1} ({name})")
                    if settings[i]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(
                            img_np, settings[i]["target_size"], settings[i]["max_size"]
                        )
                        img_tensor = self.np_to_tensor(scaled_img)
                    else:
                        img_tensor = image
                        scale = 1.0
                    detected_raw = self.detect_with_model(
                        det, det_type, img_tensor, settings[i]["confidence"],
                        settings[i]["dilation"], settings[i]["crop_factor"], settings[i]["drop_size"]
                    )
                    detected = []
                    for seg_dict in detected_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(int(coord * (1 / scale)) for coord in seg_dict["bbox"])
                            orig_crop_region = self.calculate_crop_region(
                                orig_bbox, original_image_shape_hw, settings[i]["crop_factor"]
                            )
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        
                        # === CRITICAL FIX: Recalculate mask IMMEDIATELY after detection ===
                        if include_masks_in_output:
                            seg_dict = self.recalculate_mask(
                                original_image_np, seg_dict, original_image_shape_hw,
                                simplify_masks, kernel_size, iterations
                            )
                        else:
                            crop_region = seg_dict["crop_region"]
                            h = crop_region[3] - crop_region[1]
                            w = crop_region[2] - crop_region[0]
                            seg_dict["cropped_mask"] = np.zeros((h, w), dtype=np.float32)
                        
                        detected.append(seg_dict)
                    if settings[i]["classes"]:
                        detected = self.filter_by_classes(detected, settings[i]["classes"])
                    detected = self.apply_nms(detected, settings[i]["iou"])
                    if i == 0:
                        results["stage1"] = detected
                    elif i == 1:
                        results["stage2"] = detected
                    elif i == 2:
                        results["stage3"] = detected
                else:
                    logger.info(f"Detector {i+1} is None, skipping Stage {i+1}")
            else:
                logger.info(f"Stage {i + 1} disabled, skipping")
        all_segs = results["stage1"] + results["stage2"] + results["stage3"]
        results["combined"] = self.apply_nms(all_segs, settings[0]["iou_threshold"])
        logger.info(f"Parallel processing finished. Combined detections: {len(results['combined'])}")
        return results

    def process(
        self,
        image: torch.Tensor,
        mode: str,
        target_size: int,
        max_size: int,
        iou_threshold: float,
        include_masks_in_output: bool,
        simplify_masks: bool,
        simplify_kernel_size: int,
        simplify_iterations: int,
        segs_input=None,
        stage_1_enabled=True,
        stage_1_detector_type="bbox",
        stage_1_bbox_detector=None,
        stage_1_segm_detector=None,
        stage_1_confidence=0.25,
        stage_1_iou_threshold=0.45,
        stage_1_dilation=0,
        stage_1_classes="",
        stage_1_crop_factor=3.0,
        stage_1_scale_mode="bbox",
        stage_1_target_size=640,
        stage_1_max_size=1024,
        stage_1_process_empty=False,
        min_confidence=0.0,
        min_bbox_width=1,
        min_bbox_height=1,
        stage_2_enabled=True,
        stage_2_detector_type="bbox",
        stage_2_bbox_detector=None,
        stage_2_segm_detector=None,
        stage_2_confidence=0.25,
        stage_2_iou_threshold=0.45,
        stage_2_dilation=0,
        stage_2_classes="",
        stage_2_crop_factor=3.0,
        stage_2_scale_mode="bbox",
        stage_2_target_size=640,
        stage_2_max_size=1024,
        stage_2_process_empty=False,
        stage_3_enabled=True,
        stage_3_detector_type="bbox",
        stage_3_bbox_detector=None,
        stage_3_segm_detector=None,
        stage_3_confidence=0.25,
        stage_3_iou_threshold=0.45,
        stage_3_dilation=0,
        stage_3_classes="",
        stage_3_crop_factor=3.0,
        stage_3_scale_mode="bbox",
        stage_3_target_size=640,
        stage_3_max_size=1024,
        stage_3_process_empty=False,
        drop_size=1,
        extra_pnginfo=None,
        prompt=None,
    ):
        logger.info(f"Starting Cascade Detector (FIXED) in mode: {mode}, include_masks: {include_masks_in_output}")
        if not self.IMPACT_AVAILABLE:
            logger.error("Impact Pack not available. Returning fallback outputs.")
            empty_img = torch.zeros_like(image)
            return (
                None,
                self.create_preview_image(image, [], (255, 255, 0, 255)),
                self.create_cropped_fragments_image(image, [], padding=10),
                None,
                None,
                None,
                empty_img,
                image,
            )

        # Check detector type mismatches
        if stage_1_enabled:
            if (stage_1_detector_type == "bbox" and stage_1_segm_detector is not None) or (
                stage_1_detector_type == "segm" and stage_1_bbox_detector is not None
            ):
                raise Exception(
                    f"Stage 1: Selected detector type '{stage_1_detector_type}' does not match the provided detector input."
                )
        if stage_2_enabled:
            if (stage_2_detector_type == "bbox" and stage_2_segm_detector is not None) or (
                stage_2_detector_type == "segm" and stage_2_bbox_detector is not None
            ):
                raise Exception(
                    f"Stage 2: Selected detector type '{stage_2_detector_type}' does not match the provided detector input."
                )
        if stage_3_enabled:
            if (stage_3_detector_type == "bbox" and stage_3_segm_detector is not None) or (
                stage_3_detector_type == "segm" and stage_3_bbox_detector is not None
            ):
                raise Exception(
                    f"Stage 3: Selected detector type '{stage_3_detector_type}' does not match the provided detector input."
                )

        # Determine detectors
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
            {
                "scale_mode": stage_1_scale_mode,
                "target_size": stage_1_target_size,
                "max_size": stage_1_max_size,
                "confidence": stage_1_confidence,
                "iou": stage_1_iou_threshold,
                "iou_threshold": iou_threshold,
                "dilation": stage_1_dilation,
                "classes": stage_1_classes,
                "crop_factor": stage_1_crop_factor,
                "drop_size": drop_size,
            },
            {
                "scale_mode": stage_2_scale_mode,
                "target_size": stage_2_target_size,
                "max_size": stage_2_max_size,
                "confidence": stage_2_confidence,
                "iou": stage_2_iou_threshold,
                "iou_threshold": iou_threshold,
                "dilation": stage_2_dilation,
                "classes": stage_2_classes,
                "crop_factor": stage_2_crop_factor,
                "drop_size": drop_size,
            },
            {
                "scale_mode": stage_3_scale_mode,
                "target_size": stage_3_target_size,
                "max_size": stage_3_max_size,
                "confidence": stage_3_confidence,
                "iou": stage_3_iou_threshold,
                "iou_threshold": iou_threshold,
                "dilation": stage_3_dilation,
                "classes": stage_3_classes,
                "crop_factor": stage_3_crop_factor,
                "drop_size": drop_size,
            },
        ]

        # Parse initial SEGS input
        initial_segs = None
        if segs_input is not None:
            if self.IMPACT_AVAILABLE and isinstance(segs_input, tuple) and len(segs_input) == 2:
                _, segs_list_impact = segs_input
                initial_segs = []
                for seg_impact in segs_list_impact:
                    if hasattr(seg_impact, "bbox"):
                        initial_segs.append(
                            {
                                "bbox": seg_impact.bbox,
                                "crop_region": getattr(seg_impact, "crop_region", seg_impact.bbox),
                                "label": getattr(seg_impact, "label", "object"),
                                "confidence": getattr(seg_impact, "confidence", 0.5),
                                "cropped_mask": getattr(seg_impact, "cropped_mask", None),
                                "orig_shape": segs_input[0],
                            }
                        )

        original_shape_hw = image.shape[1:3]
        original_image_np = self.tensor_to_np(image)

        # Process based on mode
        if mode == "sequential":
            results = self.process_sequential(
                image, initial_segs, enabled_stages, detectors_info, settings,
                include_masks_in_output, original_shape_hw, original_image_np,
                simplify_masks, simplify_kernel_size, simplify_iterations,
                stage_1_process_empty, stage_2_process_empty, stage_3_process_empty
            )
        else:  # parallel
            results = self.process_parallel(
                image, initial_segs, enabled_stages, detectors_info, settings,
                include_masks_in_output, original_shape_hw, original_image_np,
                simplify_masks, simplify_kernel_size, simplify_iterations,
                stage_1_process_empty, stage_2_process_empty, stage_3_process_empty
            )

        # Apply filters to all results
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

        preview_image = self.create_preview_image(image, results["combined"], (255, 255, 0, 255))
        cropped_fragments_image = self.create_cropped_fragments_image(image, results["combined"], padding=10)
        masked_fragments_image = self.create_masked_fragments_image(image, results["combined"], padding=10) if include_masks_in_output else torch.zeros_like(image)

        # Determine bypass image
        count_out = len(results["combined"]) if segs_output else 0
        if count_out == 0:
            num_enabled = sum(enabled_stages)
            num_with_process_empty = sum([
                1 for i in range(3) if enabled_stages[i] and [stage_1_process_empty, stage_2_process_empty, stage_3_process_empty][i]
            ])
            image_bypass = image if num_enabled == num_with_process_empty else torch.zeros_like(image)
        else:
            image_bypass = image

        logger.info(f"Output counts - Combined SEGS: {count_out}")
        return (
            segs_output,
            preview_image,
            cropped_fragments_image,
            segs_stage1,
            segs_stage2,
            segs_stage3,
            masked_fragments_image,
            image_bypass,
        )

# Register node
NODE_CLASS_MAPPINGS = {"CascadeDetectorAdvanced": CascadeDetector}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CascadeDetectorAdvanced": "🎯 Cascade Detector Advanced (FIXED - Mask Recalculation)"
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
