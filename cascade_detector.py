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
logger.info("CascadeDetector script loaded.")

# --- NEW: Import for creating full-size masks ---
# from impact.core import make_2d_mask # Assuming this is available via Impact Pack

class CascadeDetector:
    @classmethod
    def INPUT_TYPES(cls):
        detector_options = ["bbox", "segm"]
        # --- NEW: Define scale_mode options list ---
        scale_modes = ["bbox", "crop_region", "fixed"]
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["sequential", "parallel"], {"default": "sequential"}),
                # --- REMOVED: Global scale_mode ---
                # "scale_mode": (["bbox", "crop_region", "fixed"], {"default": "bbox"}),
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
                # --- NEW: include_masks_in_output switch ---
                "include_masks_in_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If True, attempts to recalculate masks for stages 2 and 3 to be compatible with output coordinates (high computational load) and generates masked fragments image. If False, masks from stages 2 and 3 are set to None to prevent errors, and masked fragments image is blank."
                    },
                ),
                # --- NEW: simplify_masks switch ---
                "simplify_masks": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If True, applies morphological closing to smooth and fill gaps in the recalculated masks (only active if include_masks_in_output is True)."
                    },
                ),
                # --- NEW: simplify_kernel_size parameter ---
                "simplify_kernel_size": (
                    "INT",
                    {"default": 5, "min": 1, "max": 21, "step": 2},
                ),
                # --- NEW: simplify_iterations parameter ---
                "simplify_iterations": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
            },
            "optional": {
                "segs_input": (
                    "SEGS",
                    {
                        "tooltip": "Used only in 'sequential' mode. Provides initial segments to start the cascade. Ignored in 'parallel' mode. Recommended to disable Stage 1 when using this."
                    },
                ),
                # --- GROUPED STAGE PARAMETERS ---
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
                        "multiline": False, # Keep single line for classes
                        "placeholder": "Class filters for Stage 1, comma-separated",
                    },
                ),
                # --- NEW: stage_1_crop_factor ---
                "stage_1_crop_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                # --- NEW: stage_1_scale_mode ---
                "stage_1_scale_mode": (scale_modes, {"default": "bbox"}), # NEW
                # --- NEW: stage_1_target_size, stage_1_max_size ---
                "stage_1_target_size": (
                    "INT",
                    {"default": 640, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "stage_1_max_size": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                # --- NEW: stage_1_process_empty ---
                "stage_1_process_empty": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If True and Stage 1 receives no input segments (e.g., from segs_input or previous stages), it will run its detector on the full input image. If False, it will output an empty list of segments."
                    },
                ),
                # --- NEW: Min Confidence and BBox Size for Final Output ---
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
                        "multiline": False, # Keep single line for classes
                        "placeholder": "Class filters for Stage 2, comma-separated",
                    },
                ),
                # --- NEW: stage_2_crop_factor ---
                "stage_2_crop_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                # --- NEW: stage_2_scale_mode ---
                "stage_2_scale_mode": (scale_modes, {"default": "bbox"}), # NEW
                # --- NEW: stage_2_target_size, stage_2_max_size ---
                "stage_2_target_size": (
                    "INT",
                    {"default": 640, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "stage_2_max_size": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                # --- NEW: stage_2_process_empty ---
                "stage_2_process_empty": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If True and Stage 2 receives no input segments from Stage 1, it will run its detector on the full input image. If False, it will output an empty list of segments."
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
                        "multiline": False, # Keep single line for classes
                        "placeholder": "Class filters for Stage 3, comma-separated",
                    },
                ),
                # --- NEW: stage_3_crop_factor ---
                "stage_3_crop_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                # --- NEW: stage_3_scale_mode ---
                "stage_3_scale_mode": (scale_modes, {"default": "bbox"}), # NEW
                # --- NEW: stage_3_target_size, stage_3_max_size ---
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
                        "tooltip": "If True and Stage 3 receives no input segments from Stage 2, it will run its detector on the full input image. If False, it will output an empty list of segments."
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
        "SEGS", # segs_output_all_stages
        "IMAGE", # preview_image
        "IMAGE", # cropped_fragments_image
        "SEGS", # stage1_segs
        "SEGS", # stage2_segs
        "SEGS", # stage3_segs
        "IMAGE", # masked_fragments_image (NEW)
        "IMAGE", # image_bypass
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
    Use 'segs_input' only in 'sequential' mode to start the cascade (recommended with Stage 1 disabled). 
    The main output for further processing (e.g., SEGSPaste, SEGSDetailer) is 'segs_output_all_stages', which combines results from all stages. 
    Includes checks for detector type mismatches, per-stage crop factors, filtering by confidence/BBox size, and an image bypass output if no detections are found. 
    The 'include_masks_in_output' toggle enables experimental mask recalculation for stages 2 and 3 (high load) and generates a combined image of masked fragments.
    The 'simplify_masks' toggle applies morphological closing to the recalculated masks to reduce complexity (requires 'include_masks_in_output').
    Per-stage 'scale_mode' options control how the image/region is scaled before detection on each stage.
    Per-stage 'target_size' and 'max_size' options allow independent control over the scaling for detection on each stage.
    The 'image_bypass' output returns the original input image if no detections are found after filtering, allowing downstream nodes to receive the unprocessed image in such cases.
    Per-stage 'process_empty' toggles (stage_1_process_empty, stage_2_process_empty, stage_3_process_empty) allow a stage to run its detector on the full input image if it receives no segments from the previous stage, enabling fallback detection.""" # NEW: Updated description

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
        # Sort by confidence descending
        segs.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        selected = []
        while segs:
            current = segs.pop(0)
            selected.append(current)
            # Filter out overlapping ones
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
        # Clamp to image bounds
        orig_x1 = max(0, min(image_shape_hw[1], orig_x1))
        orig_y1 = max(0, min(image_shape_hw[0], orig_y1))
        orig_x2 = max(0, min(image_shape_hw[1], orig_x2))
        orig_y2 = max(0, min(image_shape_hw[0], orig_y2))
        seg["bbox"] = (orig_x1, orig_y1, orig_x2, orig_y2)
        seg["crop_region"] = self.calculate_crop_region(seg["bbox"], image_shape_hw)
        return seg

    # --- NEW: Function to recalculate mask ---
    def recalculate_mask(self, original_image_np: np.ndarray, seg_result: Dict, original_image_shape_hw: Tuple[int, int], simplify_masks: bool, kernel_size: int, iterations: int):
        """
        Recalculates the cropped_mask for a given seg_result based on its new crop_region.
        This involves creating a full-size mask and then cropping it appropriately.
        Args:
            original_image_np: The original input image as a numpy array (H, W, C).
            seg_result: The SEG result dictionary containing bbox, crop_region, and potentially an old cropped_mask.
            original_image_shape_hw: The shape (H, W) of the original image.
            simplify_masks: Whether to apply morphological closing to the mask.
            kernel_size: Kernel size for the morphological operation.
            iterations: Number of times to apply the morphological operation.
        Returns:
            The updated seg_result dictionary with the recalculated cropped_mask.
        """
        # Get the new crop region for this specific seg_result
        crop_region = seg_result["crop_region"]
        x1_cr, y1_cr, x2_cr, y2_cr = crop_region
        # Ensure crop region is within bounds
        x1_cr = max(0, min(original_image_shape_hw[1], x1_cr))
        y1_cr = max(0, min(original_image_shape_hw[0], y1_cr))
        x2_cr = max(0, min(original_image_shape_hw[1], x2_cr))
        y2_cr = max(0, min(original_image_shape_hw[0], y2_cr))

        # Extract the corresponding image fragment from the original image
        fragment_image = self.crop_image(original_image_np, crop_region)
        fragment_h, fragment_w = fragment_image.shape[:2]

        # Create a full-size mask initialized to zeros
        full_mask = np.zeros(original_image_shape_hw, dtype=np.uint8)

        # Get the original bbox mask if available (from stage 1 or initial input)
        # If not available (e.g., from stage 2/3 detection), we cannot accurately recreate it from scratch.
        # However, if the detector provided a cropped_mask for the fragment, we can rescale it.
        old_cropped_mask = seg_result.get("cropped_mask")

        if old_cropped_mask is not None and old_cropped_mask.ndim >= 2:
             # Handle potential batch dimension if present (though unlikely from Impact Pack)
             if old_cropped_mask.ndim == 3 and old_cropped_mask.shape[0] == 1:
                 old_cropped_mask = old_cropped_mask[0]
             elif old_cropped_mask.ndim > 2:
                 # Assume first channel or take max/mean if multi-channel mask
                 old_cropped_mask = old_cropped_mask.max(axis=0) if old_cropped_mask.ndim == 3 else old_cropped_mask

             # Ensure old_cropped_mask is float32 or uint8 for cv2.resize
             if old_cropped_mask.dtype != np.uint8 and old_cropped_mask.dtype != np.float32:
                 # Assume it's float-like and normalize/convert to float32
                 old_cropped_mask = old_cropped_mask.astype(np.float32)
                 # Normalize to 0-1 if values are outside this range (e.g., 0-255 accidentally passed as float)
                 if old_cropped_mask.max() > 1.0:
                      old_cropped_mask = old_cropped_mask / 255.0

             # Resize the old mask to the size of the current fragment
             if old_cropped_mask.shape[:2] != (fragment_h, fragment_w): # Check height and width
                 # Use cv2 resize for masks (nearest neighbor is often preferred, but linear is okay for soft masks)
                 # Ensure old_cropped_mask is float32 or uint8
                 resized_mask = cv2.resize(old_cropped_mask, (fragment_w, fragment_h), interpolation=cv2.INTER_LINEAR)
                 # Threshold back to binary if necessary
                 # Convert to uint8 for thresholding if it's float
                 if resized_mask.dtype == np.float32:
                     resized_mask = (resized_mask > 0.5).astype(np.uint8) * 255 # Use 0.5 as threshold for float
                 elif resized_mask.dtype == np.uint8:
                     resized_mask = (resized_mask > 127).astype(np.uint8) * 255 # Use 127 as threshold for uint8
                 else:
                     # Should not happen if dtype was converted earlier, but just in case
                     logger.warning(f"Unexpected dtype {resized_mask.dtype} for resized mask. Attempting conversion.")
                     resized_mask = (resized_mask > 0.5).astype(np.uint8) * 255
             else:
                 resized_mask = old_cropped_mask
                 # Ensure it's uint8 for the |= operation
                 if resized_mask.dtype != np.uint8:
                      if resized_mask.dtype == np.float32:
                          resized_mask = (resized_mask > 0.5).astype(np.uint8) * 255
                      else:
                          resized_mask = (resized_mask > 0.5).astype(np.uint8) * 255 # Default safe conversion

             # --- NEW: Simplify the mask if requested ---
             if simplify_masks:
                 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                 # Apply morphological closing (dilate then erode) to close gaps and smooth edges
                 resized_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

             # Ensure the slice and the resized_mask have the same dtype and shape for the operation
             target_slice = full_mask[y1_cr:y2_cr, x1_cr:x2_cr]
             if target_slice.shape != resized_mask.shape:
                 logger.warning(f"Shape mismatch for mask paste: target_slice {target_slice.shape} vs resized_mask {resized_mask.shape}. Skipping mask paste for this segment.")
                 # Set cropped_mask to None as the operation cannot proceed safely
                 seg_result["cropped_mask"] = None
                 return seg_result

             # Ensure both arrays for |= are np.uint8
             if target_slice.dtype != np.uint8 or resized_mask.dtype != np.uint8:
                  logger.warning(f"Dtype mismatch for |= operation: target_slice {target_slice.dtype} vs resized_mask {resized_mask.dtype}. Converting both to uint8.")
                  target_slice = target_slice.astype(np.uint8)
                  resized_mask = resized_mask.astype(np.uint8)

             # Paste the resized mask onto the full mask at the crop region location using |=
             # Make sure to assign back to the original slice view
             full_mask[y1_cr:y2_cr, x1_cr:x2_cr] = target_slice | resized_mask # Explicit bitwise OR assignment

             # Now, extract the mask corresponding to the *new* bbox within the *new* crop_region
             # The new bbox coordinates are already absolute (from transform_coordinates)
             x1_bb, y1_bb, x2_bb, y2_bb = seg_result["bbox"]
             # Find the relative position of the bbox within the crop_region
             rel_x1 = max(0, x1_bb - x1_cr)
             rel_y1 = max(0, y1_bb - y1_cr)
             rel_x2 = min(fragment_w, x2_bb - x1_cr)
             rel_y2 = min(fragment_h, y2_bb - y1_cr)

             # Extract the mask part relevant to the new crop_region defined by the new bbox/crop_region
             # This becomes the new cropped_mask for the SEG object.
             # It should cover the area defined by crop_region, but the mask data comes from the relevant part
             # of the full_mask (or the fragment_mask applied to the full_mask).
             # The crop_region for the SEG object defines the *input* area for detailers.
             # The cropped_mask should be the mask *within* that crop_region.
             # So, we take the full_mask and slice it by crop_region.
             new_cropped_mask = full_mask[y1_cr:y2_cr, x1_cr:x2_cr].astype(np.float32) / 255.0
             seg_result["cropped_mask"] = new_cropped_mask
             # The crop_region remains the one calculated for the new bbox.
             # seg_result["crop_region"] = crop_region # Already set correctly
        else:
             # If there was no old mask, set the cropped_mask to None for safety.
             seg_result["cropped_mask"] = None

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
            f"Attempting detection with {detector_type} detector, confidence={confidence}, dilation={dilation}, det_type={type(detector)}"
        )
        if not self.IMPACT_AVAILABLE or detector is None:
            logger.warning(
                f"Impact Pack available (at init): {self.IMPACT_AVAILABLE}, Detector is None: {detector is None}"
            )
            logger.warning(
                "Impact Pack not available (at init) or detector is None. Cannot perform detection."
            )
            return []
        try:
            logger.debug(f"Calling detector.detect directly for {detector_type}")
            # Wrap the detector.detect call in try-except
            try:
                shape, segs_list_impact = detector.detect(
                    image_tensor,
                    confidence,
                    dilation,
                    crop_factor,
                    drop_size,
                    detailer_hook=None,
                )
            except Exception as e_internal:
                logger.error(
                    f"Internal error during {detector_type} detection call: {e_internal}"
                )
                import traceback
                traceback.print_exc()
                return []  # Return empty list if detection fails internally
            logger.debug(
                f"Detection successful via detector.detect, got {len(segs_list_impact)} segments from Impact Pack."
            )
            unified_segs = []
            for seg_impact in segs_list_impact:
                # Add a check for seg_impact itself
                if seg_impact is None:
                    logger.warning(
                        "Received None segment from detector.detect, skipping."
                    )
                    continue
                # Add checks for bbox and confidence
                if not hasattr(seg_impact, "bbox") or not hasattr(
                    seg_impact, "confidence"
                ):
                    logger.warning(
                        f"Segment object missing required attributes (bbox or confidence): {dir(seg_impact)}, skipping."
                    )
                    continue
                bbox = seg_impact.bbox
                if bbox is None:
                    logger.warning("Segment bbox is None, skipping.")
                    continue
                if isinstance(bbox, (list, tuple)):
                    bbox = tuple(bbox)
                elif hasattr(bbox, "tolist"):
                    bbox = tuple(bbox.tolist())
                else:
                    try:
                        bbox = tuple(bbox)
                    except TypeError:
                        logger.warning(
                            f"Unable to convert bbox to tuple: {type(bbox)}, skipping."
                        )
                        continue
                confidence_val = seg_impact.confidence
                if confidence_val is None:
                    logger.warning("Segment confidence is None, skipping.")
                    continue
                if hasattr(confidence_val, "item"):
                    confidence_val = confidence_val.item()
                elif isinstance(confidence_val, (list, tuple, np.ndarray)):
                    if len(confidence_val) > 0:
                        confidence_val = float(confidence_val[0])
                    else:
                        logger.warning("Segment confidence array is empty, skipping.")
                        continue
                else:
                    try:
                        confidence_val = float(confidence_val)
                    except TypeError:
                        logger.warning(
                            f"Unable to convert confidence to float: {type(confidence_val)}, skipping."
                        )
                        continue
                # Add check for cropped_mask
                cropped_mask = getattr(seg_impact, "cropped_mask", None)  # Allow None
                if (
                    cropped_mask is not None
                    and not isinstance(cropped_mask, np.ndarray)
                    and not isinstance(cropped_mask, torch.Tensor)
                ):
                    logger.warning(
                        f"Segment cropped_mask is unexpected type: {type(cropped_mask)}, allowing None."
                    )
                    cropped_mask = None

                unified_segs.append(
                    {
                        "bbox": bbox,
                        "crop_region": getattr(
                            seg_impact, "crop_region", bbox
                        ),  # Fallback to bbox
                        "label": getattr(seg_impact, "label", "object"),
                        "confidence": confidence_val,
                        "cropped_mask": cropped_mask,  # Can be None
                        "orig_shape": shape,
                    }
                )
            return unified_segs
        except AttributeError as ae:
            logger.error(f"Detector object does not have a 'detect' method: {ae}")
            import traceback
            traceback.print_exc()
            return []
        except Exception as e:
            logger.error(f"Unexpected error during {detector_type} detection: {e}")
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
            logger.warning("Unexpected image format for preview. Converting naively.")
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
            logger.debug(
                "No segments provided to create_cropped_fragments_image, returning a small blank canvas."
            )
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
        # canvas_draw = ImageDraw.Draw(canvas_img)
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
                current_y += row_height + padding # ADDED: spacing between rows
                row_height = 0
            canvas_img.paste(frag_pil, (current_x, current_y))
            # canvas_draw.rectangle( # REMOVED: No yellow border
            #     [current_x, current_y, current_x + frag_w - 1, current_y + frag_h - 1],
            #     outline=(255, 255, 0),
            #     width=10,
            # )
            current_x += frag_w + padding
            row_height = max(row_height, frag_h)

        final_canvas = canvas_img.crop(
            (
                0,
                0,
                min(canvas_size, current_x + padding), # ADDED: spacing after last column
                min(canvas_size, current_y + row_height + padding), # ADDED: spacing after last row
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


    # --- NEW: Function to create a combined image of masked fragments ---
    def create_masked_fragments_image(self, image: torch.Tensor, segs_list: List[Dict], padding: int = 10) -> torch.Tensor:
        """
        Creates a single image combining masked fragments from the segs_list.
        Args:
            image: The original input image (BCHW format).
            segs_list: List of seg dictionaries containing bbox, crop_region, cropped_mask.
            padding: Padding between fragments on the canvas.
        Returns:
            A tensor (BCHW format) representing the combined masked fragments image.
        """
        img_np = self.tensor_to_np(image) # HWC format
        h, w, c = img_np.shape
        if c == 4:
            bg_color = (0, 0, 0, 0)
        else:
            bg_color = (0, 0, 0)

        if not segs_list:
            logger.debug(
                "No segments provided to create_masked_fragments_image, returning a small blank canvas."
            )
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
        # canvas_draw = ImageDraw.Draw(canvas_img) # REMOVED: No yellow border for masked fragments either
        current_x = padding
        current_y = padding
        row_height = 0
        max_row_width = canvas_size - 2 * padding

        for i, seg in enumerate(segs_list):
            crop_region = seg["crop_region"]
            cropped_mask_np = seg.get("cropped_mask")

            # Crop the original image to the region
            frag_np = self.crop_image(img_np, crop_region) # HWC format

            if frag_np.size == 0 or cropped_mask_np is None:
                 logger.warning(f"Segment {i} has zero-sized fragment or no mask. Skipping.")
                 continue

            # Ensure mask is numpy and shape is (H, W)
            if isinstance(cropped_mask_np, torch.Tensor):
                 cropped_mask_np = cropped_mask_np.cpu().numpy()
            if cropped_mask_np.ndim == 3 and cropped_mask_np.shape[0] == 1:
                 cropped_mask_np = cropped_mask_np[0]
            elif cropped_mask_np.ndim != 2:
                 logger.warning(f"Segment {i} cropped_mask has unexpected shape {cropped_mask_np.shape}. Skipping.")
                 continue

            # Expand mask to all channels if necessary
            if frag_np.shape[2] == 3:
                expanded_mask = np.stack([cropped_mask_np]*3, axis=-1) # HWC
            elif frag_np.shape[2] == 4:
                expanded_mask = np.stack([cropped_mask_np]*4, axis=-1) # HWC
            else:
                logger.warning(f"Segment {i} cropped image has unexpected number of channels: {frag_np.shape[2]}. Skipping.")
                continue

            # Apply the mask to the fragment
            masked_frag_np = frag_np * expanded_mask # HWC format

            # Convert masked fragment to PIL Image
            frag_pil = Image.fromarray((masked_frag_np * 255).astype(np.uint8))

            frag_w, frag_h = frag_pil.size
            if current_x + frag_w > max_row_width:
                current_x = padding
                current_y += row_height + padding # ADDED: spacing between rows
                row_height = 0
            canvas_img.paste(frag_pil, (current_x, current_y))
            # canvas_draw.rectangle( # REMOVED: No yellow border
            #     [current_x, current_y, current_x + frag_w - 1, current_y + frag_h - 1],
            #     outline=(255, 255, 0), # Yellow border
            #     width=10,
            # )
            current_x += frag_w + padding # ADDED: spacing between columns
            row_height = max(row_height, frag_h)

        final_canvas = canvas_img.crop(
            (
                0,
                0,
                min(canvas_size, current_x + padding), # ADDED: spacing after last column
                min(canvas_size, current_y + row_height + padding), # ADDED: spacing after last row
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
            logger.error(
                "Impact Pack not available (at init). Cannot convert to SEGS format. Returning None."
            )
            return None
        try:
            segs_objects = []
            for seg in segs_list:
                # Ensure cropped_mask is correctly formatted (torch.Tensor, CHW, 0-1)
                cropped_mask_tensor = seg.get("cropped_mask")
                if cropped_mask_tensor is not None:
                    if not isinstance(cropped_mask_tensor, torch.Tensor):
                         cropped_mask_tensor = torch.from_numpy(cropped_mask_tensor)
                    if cropped_mask_tensor.ndim == 2:
                         cropped_mask_tensor = cropped_mask_tensor.unsqueeze(0) # Add channel dim if missing
                    if cropped_mask_tensor.ndim != 3:
                         logger.warning(f"Skipping seg, cropped_mask has incorrect dims: {cropped_mask_tensor.ndim}. Expected 2 or 3.")
                         continue
                    # Ensure values are 0-1
                    if cropped_mask_tensor.max() > 1.0:
                         cropped_mask_tensor = cropped_mask_tensor / 255.0
                else:
                    # If cropped_mask is None, pass None, Impact Pack might handle it or set default
                    pass

                seg_obj = self.SEG_IMPACT(
                    cropped_image=seg.get("cropped_image"), # Usually None after processing
                    cropped_mask=cropped_mask_tensor, # Pass the processed mask
                    confidence=seg.get("confidence", 0.5),
                    crop_region=seg.get("crop_region", seg["bbox"]),
                    bbox=seg["bbox"],
                    label=seg.get("label", "object"),
                    control_net_wrapper=None,
                )
                segs_objects.append(seg_obj)
            return (image_shape_wh, segs_objects)
        except Exception as e:
            logger.error(f"Failed to convert to SEGS format using Impact Pack: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_sequential(
        self,
        image: torch.Tensor,
        initial_segs: Optional[List[Dict]],
        enabled_stages: List[bool],
        detectors_info: List,
        settings: List[Dict], # Now includes stage_X_target_size, stage_X_max_size
        include_masks_in_output: bool, # NEW: Pass the flag
        original_image_shape_hw: Tuple[int, int], # NEW: Pass original shape
        original_image_np: np.ndarray, # NEW: Pass original image as np array
        simplify_masks: bool, # NEW: Pass the flag
        kernel_size: int,     # NEW: Pass the parameter
        iterations: int,      # NEW: Pass the parameter
        # NEW: Pass process_empty flags
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
                # NEW LOGIC: Check if input is empty and process_empty is True
                if not current_segs and stage_1_process_empty:
                    logger.info("Stage 1: Input segments are empty, running detector on full image due to process_empty=True")
                    img_np = self.tensor_to_np(image)
                    # Apply scaling based on settings[0]
                    if settings[0]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(
                            img_np, settings[0]["target_size"], settings[0]["max_size"]
                        )
                        img_tensor = self.np_to_tensor(scaled_img)
                        logger.debug(
                            f"Scaled image for Stage 1 (via process_empty): {scaled_img.shape} with scale {scale}"
                        )
                    else:
                        img_tensor = image
                        scale = 1.0
                    stage1_segs_raw = self.detect_with_model(
                        det,
                        det_type,
                        img_tensor,
                        settings[0]["confidence"],
                        settings[0]["dilation"],
                        settings[0]["crop_factor"],
                        settings[0]["drop_size"],
                    )
                    stage1_segs = []
                    for seg_dict in stage1_segs_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(
                                int(coord * (1 / scale)) for coord in seg_dict["bbox"]
                            )
                            orig_crop_region = self.calculate_crop_region(
                                orig_bbox, original_image_shape_hw, settings[0]["crop_factor"]
                            )
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        stage1_segs.append(seg_dict)
                    results["stage1"] = stage1_segs
                    current_segs = stage1_segs # Update current_segs for next stage
                else:
                    # Original logic: process current_segs (could be initial_segs or results from prev stage)
                    # This part runs if current_segs is not empty, OR if process_empty is False
                    img_np = self.tensor_to_np(image)
                    if settings[0]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(
                            img_np, settings[0]["target_size"], settings[0]["max_size"]
                        )
                        img_tensor = self.np_to_tensor(scaled_img)
                        logger.debug(
                            f"Scaled image for Stage 1 (normal): {scaled_img.shape} with scale {scale}"
                        )
                    else:
                        img_tensor = image
                        scale = 1.0
                    stage1_segs_raw = self.detect_with_model(
                        det,
                        det_type,
                        img_tensor,
                        settings[0]["confidence"],
                        settings[0]["dilation"],
                        settings[0]["crop_factor"],
                        settings[0]["drop_size"],
                    )
                    stage1_segs = []
                    for seg_dict in stage1_segs_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(
                                int(coord * (1 / scale)) for coord in seg_dict["bbox"]
                            )
                            orig_crop_region = self.calculate_crop_region(
                                orig_bbox, original_image_shape_hw, settings[0]["crop_factor"]
                            )
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        stage1_segs.append(seg_dict)
                    logger.debug(
                        f"Stage 1 raw detections: {len(stage1_segs_raw)}, post-scaled: {len(stage1_segs)}"
                    )
                    if settings[0]["classes"]:
                        stage1_segs = self.filter_by_classes(
                            stage1_segs, settings[0]["classes"]
                        )
                    logger.debug(f"Stage 1 after class filter: {len(stage1_segs)}")
                    stage1_segs = self.apply_nms(stage1_segs, settings[0]["iou"])
                    logger.debug(f"Stage 1 after NMS: {len(stage1_segs)}")
                    results["stage1"] = stage1_segs
                    current_segs = stage1_segs
            else:
                logger.info("Detector 1 is None, skipping Stage 1.")
        else:
            logger.info("Stage 1 disabled or no Detector 1 provided, skipping Stage 1.")

        # Stage 2
        if enabled_stages[1] and detectors_info[1] is not None:
             logger.info(f"Stage 2: Processing {len(current_segs)} input segments")
             det, det_type, name = detectors_info[1]
             if det is not None:
                 # NEW LOGIC: Check if input is empty and process_empty is True
                 if not current_segs and stage_2_process_empty:
                     logger.info("Stage 2: Input segments are empty, running detector on full image due to process_empty=True")
                     img_np = self.tensor_to_np(image)
                     # Apply scaling based on settings[1] for full image
                     if settings[1]["scale_mode"] == "fixed":
                         scaled_img, scale = self.resize_image(
                             img_np, settings[1]["target_size"], settings[1]["max_size"]
                         )
                         img_tensor = self.np_to_tensor(scaled_img)
                         logger.debug(
                             f"Scaled image for Stage 2 (via process_empty): {scaled_img.shape} with scale {scale}"
                         )
                     else:
                         img_tensor = image
                         scale = 1.0
                     detected_raw = self.detect_with_model(
                         det,
                         det_type,
                         img_tensor,
                         settings[1]["confidence"],
                         settings[1]["dilation"],
                         settings[1]["crop_factor"],
                         settings[1]["drop_size"],
                     )
                     stage2_segs = []
                     for seg_dict in detected_raw:
                         if scale != 1.0:
                             orig_bbox = tuple(
                                 int(coord * (1 / scale)) for coord in seg_dict["bbox"]
                             )
                             orig_crop_region = self.calculate_crop_region(
                                 orig_bbox, original_image_shape_hw, settings[1]["crop_factor"]
                             )
                             seg_dict["bbox"] = orig_bbox
                             seg_dict["crop_region"] = orig_crop_region
                             seg_dict["orig_shape"] = original_image_shape_hw
                         # Apply recalculate_mask if needed
                         if include_masks_in_output and seg_dict.get("cropped_mask") is not None:
                             seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                         else:
                             seg_dict["cropped_mask"] = None # Clear mask if not recalculating
                         stage2_segs.append(seg_dict)
                     results["stage2"] = stage2_segs
                     current_segs = stage2_segs # Update for next stage
                 elif current_segs: # Only run on segments if input is not empty
                     logger.info(f"Stage 2: Detecting on {len(current_segs)} crops")
                     stage2_segs = []
                     img_np = self.tensor_to_np(image)
                     for i, parent_seg in enumerate(current_segs):
                         logger.debug(
                             f"Processing crop {i + 1}/{len(current_segs)} for Stage 2"
                         )
                         crop_region = parent_seg.get("crop_region", parent_seg["bbox"])
                         cropped = self.crop_image(img_np, crop_region)
                         if cropped.size == 0:
                             logger.warning(
                                 f"Crop {i + 1} resulted in zero-sized image. Skipping."
                             )
                             continue
                         scale = 1.0
                         if settings[1]["scale_mode"] == "fixed":
                              # Scale the cropped fragment using stage 2's specific target/max sizes
                              scaled_crop, scale = self.resize_image(
                                   cropped, settings[1]["target_size"], settings[1]["max_size"]
                              )
                              cropped_tensor = self.np_to_tensor(scaled_crop)
                              logger.debug(
                                   f"Rescaled crop for Stage 2: {cropped.shape} -> {scaled_crop.shape} with scale {scale}"
                              )
                         elif settings[1]["scale_mode"] == "bbox":
                             bbox_in_crop = (
                                 parent_seg["bbox"][0] - crop_region[0],
                                 parent_seg["bbox"][1] - crop_region[1],
                                 parent_seg["bbox"][2] - crop_region[0],
                                 parent_seg["bbox"][3] - crop_region[1],
                             )
                             bbox_size = max(
                                 bbox_in_crop[2] - bbox_in_crop[0],
                                 bbox_in_crop[3] - bbox_in_crop[1],
                             )
                             if bbox_size > 0:
                                 scale = settings[1]["target_size"] / bbox_size
                                 max_crop_dim = max(cropped.shape[:2])
                                 if scale * max_crop_dim > settings[1]["max_size"]:
                                     scale = settings[1]["max_size"] / max_crop_dim
                                 new_h = int(cropped.shape[0] * scale)
                                 new_w = int(cropped.shape[1] * scale)
                                 if new_h > 0 and new_w > 0:
                                     scaled_crop = cv2.resize(
                                         cropped,
                                         (new_w, new_h),
                                         interpolation=cv2.INTER_LINEAR,
                                     )
                                     cropped_tensor = self.np_to_tensor(scaled_crop)
                                     logger.debug(
                                         f"Rescaled crop for Stage 2: {cropped.shape} -> {scaled_crop.shape} with scale {scale}"
                                     )
                                 else:
                                     logger.warning(
                                         f"Calculated scale {scale} led to invalid dimensions. Using original crop."
                                     )
                                     cropped_tensor = self.np_to_tensor(cropped)
                                     scale = 1.0
                             else:
                                 logger.warning(
                                     f"BBox size in crop {i + 1} is zero. Using original crop."
                                 )
                                 cropped_tensor = self.np_to_tensor(cropped)
                                 scale = 1.0
                         else: # crop_region or other modes if extended
                             cropped_tensor = self.np_to_tensor(cropped)
                             scale = 1.0

                         detected_raw = self.detect_with_model(
                             det,
                             det_type,
                             cropped_tensor,
                             settings[1]["confidence"],
                             settings[1]["dilation"],
                             settings[1]["crop_factor"],
                             settings[1]["drop_size"],
                         )
                         for seg_dict in detected_raw:
                             if settings[1]["classes"]:
                                 if seg_dict.get("label", "").lower() not in [
                                     c.strip().lower()
                                     for c in settings[1]["classes"].split(",")
                                     if c.strip()
                                 ]:
                                     continue
                             crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
                             seg_x1_scaled, seg_y1_scaled, seg_x2_scaled, seg_y2_scaled = (
                                 seg_dict["bbox"]
                             )
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
                                 seg_dict["bbox"],
                                 original_image_shape_hw,
                                 settings[1]["crop_factor"],
                             )
                             seg_dict["orig_shape"] = original_image_shape_hw

                             # --- NEW: Recalculate mask if include_masks_in_output is True ---
                             if include_masks_in_output and seg_dict.get("cropped_mask") is not None:
                                  seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                             else:
                                  seg_dict["cropped_mask"] = None # Clear mask if not recalculating

                             stage2_segs.append(seg_dict)
                     logger.debug(f"Stage 2 total detections before NMS: {len(stage2_segs)}")
                     stage2_segs = self.apply_nms(stage2_segs, settings[1]["iou"])
                     logger.debug(f"Stage 2 after NMS: {len(stage2_segs)}")
                     results["stage2"] = stage2_segs
                     current_segs = stage2_segs
                 else: # Input was empty and process_empty was False
                     logger.info("Stage 2: Input segments are empty and process_empty=False, skipping detection. Outputting empty list.")
                     results["stage2"] = []
                     # current_segs remains empty for stage 3
             else:
                 logger.info("Detector 2 is None, skipping Stage 2.")
        else:
             logger.info("Stage 2 disabled or no Detector 2 provided, skipping Stage 2.")

        # Stage 3
        if enabled_stages[2] and detectors_info[2] is not None:
             logger.info(f"Stage 3: Processing {len(current_segs)} input segments")
             det, det_type, name = detectors_info[2]
             if det is not None:
                 # NEW LOGIC: Check if input is empty and process_empty is True
                 if not current_segs and stage_3_process_empty:
                     logger.info("Stage 3: Input segments are empty, running detector on full image due to process_empty=True")
                     img_np = self.tensor_to_np(image)
                     # Apply scaling based on settings[2] for full image
                     if settings[2]["scale_mode"] == "fixed":
                         scaled_img, scale = self.resize_image(
                             img_np, settings[2]["target_size"], settings[2]["max_size"]
                         )
                         img_tensor = self.np_to_tensor(scaled_img)
                         logger.debug(
                             f"Scaled image for Stage 3 (via process_empty): {scaled_img.shape} with scale {scale}"
                         )
                     else:
                         img_tensor = image
                         scale = 1.0
                     detected_raw = self.detect_with_model(
                         det,
                         det_type,
                         img_tensor,
                         settings[2]["confidence"],
                         settings[2]["dilation"],
                         settings[2]["crop_factor"],
                         settings[2]["drop_size"],
                     )
                     stage3_segs = []
                     for seg_dict in detected_raw:
                         if scale != 1.0:
                             orig_bbox = tuple(
                                 int(coord * (1 / scale)) for coord in seg_dict["bbox"]
                             )
                             orig_crop_region = self.calculate_crop_region(
                                 orig_bbox, original_image_shape_hw, settings[2]["crop_factor"]
                             )
                             seg_dict["bbox"] = orig_bbox
                             seg_dict["crop_region"] = orig_crop_region
                             seg_dict["orig_shape"] = original_image_shape_hw
                         # Apply recalculate_mask if needed
                         if include_masks_in_output and seg_dict.get("cropped_mask") is not None:
                             seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                         else:
                             seg_dict["cropped_mask"] = None # Clear mask if not recalculating
                         stage3_segs.append(seg_dict)
                     results["stage3"] = stage3_segs
                     current_segs = stage3_segs # Update for combined (though combined uses all_segs)
                 elif current_segs: # Only run on segments if input is not empty
                     logger.info(f"Stage 3: Detecting on {len(current_segs)} crops")
                     stage3_segs = []
                     img_np = self.tensor_to_np(image)
                     for i, parent_seg in enumerate(current_segs):
                         logger.debug(
                             f"Processing crop {i + 1}/{len(current_segs)} for Stage 3"
                         )
                         crop_region = parent_seg.get("crop_region", parent_seg["bbox"])
                         cropped = self.crop_image(img_np, crop_region)
                         if cropped.size == 0:
                             logger.warning(
                                 f"Crop {i + 1} resulted in zero-sized image. Skipping."
                             )
                             continue
                         # --- UPDATED: Use settings[2]["target_size"], settings[2]["max_size"] for scaling ---
                         if settings[2]["scale_mode"] == "fixed":
                              # Scale the cropped fragment using stage 3's specific target/max sizes
                              scaled_crop, scale = self.resize_image(
                                   cropped, settings[2]["target_size"], settings[2]["max_size"]
                              )
                              cropped_tensor = self.np_to_tensor(scaled_crop)
                              logger.debug(
                                   f"Rescaled crop for Stage 3: {cropped.shape} -> {scaled_crop.shape} with scale {scale}"
                              )
                         else: # bbox or crop_region mode not typically used for stage 3 processing logic in original code
                              cropped_tensor = self.np_to_tensor(cropped)
                              scale = 1.0 # No scaling assumed for stage 3 processing logic in original code

                         detected_raw = self.detect_with_model(
                             det,
                             det_type,
                             cropped_tensor,
                             settings[2]["confidence"],
                             settings[2]["dilation"],
                             settings[2]["crop_factor"],
                             settings[2]["drop_size"],
                         )
                         for seg_dict in detected_raw:
                             if settings[2]["classes"]:
                                 if seg_dict.get("label", "").lower() not in [
                                     c.strip().lower()
                                     for c in settings[2]["classes"].split(",")
                                     if c.strip()
                                 ]:
                                     continue
                             crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
                             seg_x1_scaled, seg_y1_scaled, seg_x2_scaled, seg_y2_scaled = (
                                 seg_dict["bbox"]
                             )
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
                                 seg_dict["bbox"],
                                 original_image_shape_hw,
                                 settings[2]["crop_factor"],
                             )
                             seg_dict["orig_shape"] = original_image_shape_hw

                             # --- NEW: Recalculate mask if include_masks_in_output is True ---
                             if include_masks_in_output and seg_dict.get("cropped_mask") is not None:
                                  seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                             else:
                                  seg_dict["cropped_mask"] = None # Clear mask if not recalculating

                             stage3_segs.append(seg_dict)
                     logger.debug(f"Stage 3 total detections before NMS: {len(stage3_segs)}")
                     stage3_segs = self.apply_nms(stage3_segs, settings[2]["iou"])
                     logger.debug(f"Stage 3 after NMS: {len(stage3_segs)}")
                     results["stage3"] = stage3_segs
                     current_segs = stage3_segs
                 else: # Input was empty and process_empty was False
                     logger.info("Stage 3: Input segments are empty and process_empty=False, skipping detection. Outputting empty list.")
                     results["stage3"] = []
                     # current_segs remains empty
             else:
                 logger.info("Detector 3 is None, skipping Stage 3.")
        else:
             logger.info("Stage 3 disabled or no Detector 3 provided, skipping Stage 3.")

        # Combine all stages for final output
        all_segs = []
        # --- NEW: Apply mask recalculation or clearing to stage1 results too ---
        for seg in results["stage1"]:
            if include_masks_in_output and seg.get("cropped_mask") is not None:
                 # Stage 1 masks are already potentially aligned, but let's ensure consistency
                 # They were created on the full/scaled image, so the crop_region might be large.
                 # The recalculate_mask function should still work, ensuring the crop_region and cropped_mask align.
                 # However, if the original mask was for the *entire detection area* on a scaled-down image,
                 # this might still cause issues if the final crop_region is much smaller than the original detection context.
                 # For now, assume stage1 masks are fine or apply the same logic if needed.
                 # The safest bet for stage1 is often just to pass the mask as-is if it exists and is valid.
                 # But to maintain consistency with the output format expectation (mask matching crop_region),
                 # we might need to apply the same logic. Let's see...
                 # If stage1 mask exists and is full-image size, it won't match a smaller crop_region later.
                 # If stage1 mask exists and is part of a larger crop_region, it might be okay.
                 # The safest is to *always* ensure that the final cropped_mask matches the crop_region it's paired with.
                 # So, let's apply the recalculation to stage1 too if the flag is on.
                 # This might involve taking the existing stage1 mask and clipping it to the calculated crop_region.
                 # This is tricky because stage1 mask might be larger than the calculated crop_region.
                 # The `recalculate_mask` function handles this by creating a full mask first.
                 # For stage 1, `old_cropped_mask` would be the one from the detector.
                 # The `crop_region` is the one calculated for the stage1 bbox.
                 # So, `recalculate_mask` will put the stage1 mask into the full mask, then clip it according to its own crop_region.
                 # This should make it consistent.
                 seg = self.recalculate_mask(original_image_np, seg, original_image_shape_hw, simplify_masks, kernel_size, iterations)
            else:
                 seg["cropped_mask"] = None # Clear mask if not recalculating
            all_segs.append(seg)

        for seg in results["stage2"]:
            # Already handled in the loop above or via process_empty
            all_segs.append(seg)

        for seg in results["stage3"]:
            # Already handled in the loop above or via process_empty
            all_segs.append(seg)

        results["combined"] = self.apply_nms(all_segs, settings[0]["iou_threshold"])
        logger.info(
            f"Sequential processing finished. Combined detections: {len(results['combined'])}"
        )
        return results

    def process_parallel(
        self,
        image: torch.Tensor,
        initial_segs: Optional[List[Dict]], # Not used in parallel mode, but kept for signature consistency
        enabled_stages: List[bool],
        detectors_info: List,
        settings: List[Dict], # Now includes stage_X_target_size, stage_X_max_size
        include_masks_in_output: bool, # NEW: Pass the flag
        original_image_shape_hw: Tuple[int, int], # NEW: Pass original shape
        original_image_np: np.ndarray, # NEW: Pass original image as np array
        simplify_masks: bool, # NEW: Pass the flag
        kernel_size: int,     # NEW: Pass the parameter
        iterations: int,      # NEW: Pass the parameter
        # NEW: Pass process_empty flags (not used in parallel, but for consistency in process())
        stage_1_process_empty: bool, # Unused
        stage_2_process_empty: bool, # Unused
        stage_3_process_empty: bool, # Unused
    ) -> Dict[str, List[Dict]]:
        results = {"stage1": [], "stage2": [], "stage3": [], "combined": []}
        img_np = self.tensor_to_np(image)

        for i in range(3):
            if enabled_stages[i] and detectors_info[i] is not None:
                det, det_type, name = detectors_info[i]
                if det is not None:
                    logger.info(
                        f"Parallel processing with model {i + 1} ({name}, type: {det_type})"
                    )
                    # --- UPDATED: Use settings[i]["target_size"], settings[i]["max_size"] for scaling ---
                    if settings[i]["scale_mode"] == "fixed":
                        scaled_img, scale = self.resize_image(
                            img_np,
                            settings[i]["target_size"],
                            settings[i]["max_size"],
                        )
                        img_tensor = self.np_to_tensor(scaled_img)
                        logger.debug(
                            f"Scaled image for Parallel Model {i + 1}: {scaled_img.shape} with scale {scale}"
                        )
                    else:
                        img_tensor = image
                        scale = 1.0
                    detected_raw = self.detect_with_model(
                        det,
                        det_type,
                        img_tensor,
                        settings[i]["confidence"],
                        settings[i]["dilation"],
                        settings[i]["crop_factor"],
                        settings[i]["drop_size"],
                    )
                    detected = []
                    for seg_dict in detected_raw:
                        if scale != 1.0:
                            orig_bbox = tuple(
                                int(coord * (1 / scale))
                                for coord in seg_dict["bbox"]
                            )
                            orig_crop_region = self.calculate_crop_region(
                                orig_bbox,
                                original_image_shape_hw,
                                settings[i]["crop_factor"],
                            )
                            seg_dict["bbox"] = orig_bbox
                            seg_dict["crop_region"] = orig_crop_region
                            seg_dict["orig_shape"] = original_image_shape_hw
                        # --- NEW: Recalculate mask if include_masks_in_output is True ---
                        # In parallel mode, all detections happen on the full/scaled image initially.
                        # So, the crop_region corresponds directly to the bbox area from the detector.
                        # The `cropped_mask` from the detector should already be for its specific crop_region.
                        # However, the `recalculate_mask` function assumes it needs to take an old mask
                        # and fit it into a potentially different/new crop_region.
                        # For parallel stage 1, 2, 3, the `crop_region` is calculated *after* detection
                        # based on the detected `bbox`. The original `cropped_mask` was likely calculated
                        # based on the *input* to the detector (e.g., full image for stage 1 parallel).
                        # Therefore, for parallel mode, the `cropped_mask` from the detector might already be correct
                        # relative to the image it was detected on, but its coordinates need to map back to the *original* full image's crop region.
                        # The `recalculate_mask` function, as written, takes the *original full image* and a *seg_result*
                        # which contains the *final absolute bbox and crop_region*. It then tries to find the mask for that final crop_region.
                        # For parallel stage 1: `seg_result["bbox"]` is absolute, `crop_region` is calculated for that bbox on the original image.
                        #                      `cropped_mask` from detector was for the *scaled* input crop region.
                        #                      `recalculate_mask` needs to map the scaled mask back.
                        # For parallel stage 2/3: Similar issue, mask was for a cropped fragment, now needs to be mapped to an absolute crop_region.
                        # This means the `recalculate_mask` function should work similarly for parallel mode.
                        # The key difference is *when* the `crop_region` for the seg_result is calculated.
                        # In sequential, it's calculated *after* transforming coordinates from the fragment detection.
                        # In parallel, it's calculated *after* scaling/detection on the whole image part.
                        # Let's apply the recalculation to all parallel stages if the flag is true.
                        if include_masks_in_output and seg_dict.get("cropped_mask") is not None:
                             seg_dict = self.recalculate_mask(original_image_np, seg_dict, original_image_shape_hw, simplify_masks, kernel_size, iterations)
                        else:
                             seg_dict["cropped_mask"] = None # Clear mask if not recalculating
                        detected.append(seg_dict)

                    logger.debug(
                        f"Parallel Model {i + 1} raw detections: {len(detected_raw)}, post-scaled: {len(detected)}"
                    )
                    if settings[i]["classes"]:
                        detected = self.filter_by_classes(
                            detected, settings[i]["classes"]
                        )
                    logger.debug(
                        f"Parallel Model {i + 1} after class filter: {len(detected)}"
                    )
                    detected = self.apply_nms(detected, settings[i]["iou"])
                    logger.debug(
                        f"Parallel Model {i + 1} after NMS: {len(detected)}"
                    )
                    if i == 0:
                        results["stage1"] = detected
                    elif i == 1:
                        results["stage2"] = detected
                    elif i == 2:
                        results["stage3"] = detected
                else:
                    logger.info(
                        f"Detector {i+1} is None, skipping Stage {i+1}."
                    )
            else:
                logger.info(f"Stage {i + 1} disabled or detector is None, skipping.")

        # Combine results for parallel mode
        all_segs = results["stage1"] + results["stage2"] + results["stage3"]
        results["combined"] = self.apply_nms(all_segs, settings[0]["iou_threshold"])
        logger.info(
            f"Parallel processing finished. Combined detections: {len(results['combined'])}"
        )
        return results


    def process(
        self,
        image: torch.Tensor,
        mode: str,
        # --- REMOVED: Global scale_mode parameter ---
        # scale_mode: str,
        target_size: int,
        max_size: int,
        iou_threshold: float,
        include_masks_in_output: bool, # NEW: Get the flag
        simplify_masks: bool, # NEW: Get the flag
        simplify_kernel_size: int, # NEW: Get the parameter
        simplify_iterations: int, # NEW: Get the parameter
        segs_input=None,
        stage_1_enabled=True,
        stage_1_detector_type="bbox",
        stage_1_bbox_detector=None,
        stage_1_segm_detector=None,
        stage_1_confidence=0.25,
        stage_1_iou_threshold=0.45,
        stage_1_dilation=0,
        stage_1_classes="",
        # --- NEW: stage_1_crop_factor ---
        stage_1_crop_factor=3.0,
        # --- NEW: stage_1_scale_mode ---
        stage_1_scale_mode="bbox", # NEW
        # --- NEW: stage_1_target_size, stage_1_max_size ---
        stage_1_target_size=640, # NEW
        stage_1_max_size=1024,   # NEW
        # --- NEW: stage_1_process_empty ---
        stage_1_process_empty=False, # NEW
        # --- NEW: min_confidence, min_bbox_width, min_bbox_height ---
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
        # --- NEW: stage_2_crop_factor ---
        stage_2_crop_factor=3.0,
        # --- NEW: stage_2_scale_mode ---
        stage_2_scale_mode="bbox", # NEW
        # --- NEW: stage_2_target_size, stage_2_max_size ---
        stage_2_target_size=640, # NEW
        stage_2_max_size=1024,   # NEW
        # --- NEW: stage_2_process_empty ---
        stage_2_process_empty=False, # NEW
        stage_3_enabled=True,
        stage_3_detector_type="bbox",
        stage_3_bbox_detector=None,
        stage_3_segm_detector=None,
        stage_3_confidence=0.25,
        stage_3_iou_threshold=0.45,
        stage_3_dilation=0,
        stage_3_classes="",
        # --- NEW: stage_3_crop_factor ---
        stage_3_crop_factor=3.0,
        # --- NEW: stage_3_scale_mode ---
        stage_3_scale_mode="bbox", # NEW
        # --- NEW: stage_3_target_size, stage_3_max_size ---
        stage_3_target_size=640, # NEW
        stage_3_max_size=1024,   # NEW
        # --- NEW: stage_3_process_empty ---
        stage_3_process_empty=False, # NEW
        crop_factor=3.0,  # Kept for backward compatibility if needed elsewhere, but not used in settings
        drop_size=1,
        extra_pnginfo=None, # Hidden input
        prompt=None,        # Hidden input
    ):
        logger.info(f"Starting Cascade Detector in mode: {mode}, include_masks: {include_masks_in_output}, simplify_masks: {simplify_masks}")

        if not self.IMPACT_AVAILABLE:
            logger.error(
                "Impact Pack is not available at runtime (checked at init). Cannot perform detection or convert to SEGS. Returning only preview image and bypass image."
            )
            empty_img = torch.zeros_like(image)
            # NEW: Return an empty image for masked_fragments too
            return (
                None,
                self.create_preview_image(image, [], (255, 255, 0, 255)),
                self.create_cropped_fragments_image(image, [], padding=10),
                None,
                None,
                None,
                empty_img, # masked_fragments_image
                image, # NEW: Return original image as bypass if Impact Pack unavailable
            )  # Return input image as bypass

        # --- NEW: Check Detector Type Mismatches ---
        if stage_1_enabled:
            if (
                stage_1_detector_type == "bbox" and stage_1_segm_detector is not None
            ) or (
                stage_1_detector_type == "segm" and stage_1_bbox_detector is not None
            ):
                raise Exception(
                    f"Stage 1: Selected detector type '{stage_1_detector_type}' does not match the provided detector input. Please check the connection."
                )
        if stage_2_enabled:
            if (
                stage_2_detector_type == "bbox" and stage_2_segm_detector is not None
            ) or (
                stage_2_detector_type == "segm" and stage_2_bbox_detector is not None
            ):
                raise Exception(
                    f"Stage 2: Selected detector type '{stage_2_detector_type}' does not match the provided detector input. Please check the connection."
                )
        if stage_3_enabled:
            if (
                stage_3_detector_type == "bbox" and stage_3_segm_detector is not None
            ) or (
                stage_3_detector_type == "segm" and stage_3_bbox_detector is not None
            ):
                raise Exception(
                    f"Stage 3: Selected detector type '{stage_3_detector_type}' does not match the provided detector input. Please check the connection."
                )

        # Determine detectors based on type
        det_1 = (
            stage_1_bbox_detector
            if stage_1_detector_type == "bbox"
            else stage_1_segm_detector
        )
        det_2 = (
            stage_2_bbox_detector
            if stage_2_detector_type == "bbox"
            else stage_2_segm_detector
        )
        det_3 = (
            stage_3_bbox_detector
            if stage_3_detector_type == "bbox"
            else stage_3_segm_detector
        )
        enabled_stages = [stage_1_enabled, stage_2_enabled, stage_3_enabled]
        detectors_info = [
            (det_1, stage_1_detector_type, f"stage_1_{stage_1_detector_type}_detector")
            if det_1 is not None
            else None,
            (det_2, stage_2_detector_type, f"stage_2_{stage_2_detector_type}_detector")
            if det_2 is not None
            else None,
            (det_3, stage_3_detector_type, f"stage_3_{stage_3_detector_type}_detector")
            if det_3 is not None
            else None,
        ]
        # --- UPDATED: Include stage_X_crop_factor AND stage_X_scale_mode AND stage_X_target_size, stage_X_max_size in settings ---
        settings = [
            {
                "scale_mode": stage_1_scale_mode, # NEW: Use individual scale mode
                "target_size": stage_1_target_size, # NEW: Use individual target size
                "max_size": stage_1_max_size,     # NEW: Use individual max size
                "confidence": stage_1_confidence,
                "iou": stage_1_iou_threshold,
                "iou_threshold": iou_threshold,
                "dilation": stage_1_dilation,
                "classes": stage_1_classes,
                # Use the new per-stage crop factor
                "crop_factor": stage_1_crop_factor,
                "drop_size": drop_size,
            },
            {
                "scale_mode": stage_2_scale_mode, # NEW: Use individual scale mode
                "target_size": stage_2_target_size, # NEW: Use individual target size
                "max_size": stage_2_max_size,     # NEW: Use individual max size
                "confidence": stage_2_confidence,
                "iou": stage_2_iou_threshold,
                "iou_threshold": iou_threshold,
                "dilation": stage_2_dilation,
                "classes": stage_2_classes,
                # Use the new per-stage crop factor
                "crop_factor": stage_2_crop_factor,
                "drop_size": drop_size,
            },
            {
                "scale_mode": stage_3_scale_mode, # NEW: Use individual scale mode
                "target_size": stage_3_target_size, # NEW: Use individual target size
                "max_size": stage_3_max_size,     # NEW: Use individual max size
                "confidence": stage_3_confidence,
                "iou": stage_3_iou_threshold,
                "iou_threshold": iou_threshold,
                "dilation": stage_3_dilation,
                "classes": stage_3_classes,
                # Use the new per-stage crop factor
                "crop_factor": stage_3_crop_factor,
                "drop_size": drop_size,
            },
        ]

        initial_segs = None
        if segs_input is not None:
            if (
                self.IMPACT_AVAILABLE
                and isinstance(segs_input, tuple)
                and len(segs_input) == 2
            ):
                _, segs_list_impact = segs_input
                initial_segs = []
                for seg_impact in segs_list_impact:
                    if hasattr(seg_impact, "bbox"):
                        initial_segs.append(
                            {
                                "bbox": seg_impact.bbox,
                                "crop_region": getattr(
                                    seg_impact, "crop_region", seg_impact.bbox
                                ),
                                "label": getattr(seg_impact, "label", "object"),
                                "confidence": getattr(seg_impact, "confidence", 0.5),
                                "cropped_mask": getattr(
                                    seg_impact, "cropped_mask", None
                                ), # Pass original mask
                                "orig_shape": segs_input[0],
                            }
                        )
                    else:
                        logger.warning(
                            f"Input SEGS item is not an SEG object: {type(seg_impact)}"
                        )
            else:
                logger.warning(
                    "Input SEGS format is invalid or Impact Pack unavailable (checked at init) during input check."
                )

        original_shape_hw = image.shape[1:3]
        original_shape_wh = (original_shape_hw[1], original_shape_hw[0])
        original_image_np = self.tensor_to_np(image) # NEW: Convert once here

        if mode == "sequential":
            results = self.process_sequential(
                image, initial_segs, enabled_stages, detectors_info, settings,
                include_masks_in_output, # NEW: Pass flag
                original_shape_hw, # NEW: Pass shape
                original_image_np, # NEW: Pass image array
                simplify_masks, simplify_kernel_size, simplify_iterations, # NEW: Pass simplify params
                stage_1_process_empty, stage_2_process_empty, stage_3_process_empty # NEW: Pass process_empty flags
            )
        else:  # parallel
            results = self.process_parallel(
                image, initial_segs, enabled_stages, detectors_info, settings,
                include_masks_in_output, # NEW: Pass flag
                original_shape_hw, # NEW: Pass shape
                original_image_np, # NEW: Pass image array
                simplify_masks, simplify_kernel_size, simplify_iterations, # NEW: Pass simplify params
                stage_1_process_empty, stage_2_process_empty, stage_3_process_empty # NEW: Pass process_empty flags (unused in parallel)
            )

        preview_image = self.create_preview_image(
            image,
            results["combined"],
            (255, 255, 0, 255),  # Yellow for combined
        )
        image_shape_wh = (image.shape[2], image.shape[1])

        # --- NEW: Apply Filters to ALL results (stage1, stage2, stage3, combined) before conversion ---
        def apply_filters(segs_list):
            filtered = []
            for seg in segs_list:
                # Check confidence
                if seg.get("confidence", 0) < min_confidence:
                    continue
                # Check bbox size
                x1, y1, x2, y2 = seg["bbox"]
                width = x2 - x1
                height = y2 - y1
                if width < min_bbox_width or height < min_bbox_height:
                    continue
                # Add to filtered list
                filtered.append(seg)
            return filtered

        results["stage1"] = apply_filters(results["stage1"])
        results["stage2"] = apply_filters(results["stage2"])
        results["stage3"] = apply_filters(results["stage3"])
        results["combined"] = apply_filters(results["combined"]) # Apply to final output

        logger.debug(f"Filtered stage1 results: {len(results['stage1'])}")
        logger.debug(f"Filtered stage2 results: {len(results['stage2'])}")
        logger.debug(f"Filtered stage3 results: {len(results['stage3'])}")
        logger.debug(f"Filtered combined results: {len(results['combined'])}")


        segs_output = self.convert_to_segs_format(results["combined"], image_shape_wh)
        segs_stage1 = self.convert_to_segs_format(results["stage1"], image_shape_wh)
        segs_stage2 = self.convert_to_segs_format(results["stage2"], image_shape_wh)
        segs_stage3 = self.convert_to_segs_format(results["stage3"], image_shape_wh)

        cropped_fragments_image = self.create_cropped_fragments_image(
            image, results["combined"], padding=10
        )

        # --- NEW: Generate masked fragments image ---
        if include_masks_in_output:
             # Use the combined results which should have recalculated masks if the flag was True
             masked_fragments_image = self.create_masked_fragments_image(image, results["combined"], padding=10)
        else:
             # Return a blank image if masks are not included
             masked_fragments_image = torch.zeros_like(image)

        count_out = len(results["combined"]) if segs_output else 0
        logger.info(f"Output counts - Combined SEGS: {count_out}")

        # --- NEW: Determine image_bypass based on detection results and process_empty flags ---
        # Count enabled stages
        num_enabled_stages = sum(enabled_stages)
        # Count enabled stages where process_empty is True
        process_empty_flags = [stage_1_process_empty, stage_2_process_empty, stage_3_process_empty]
        num_enabled_stages_with_process_empty = sum(
            1 for i in range(3) if enabled_stages[i] and process_empty_flags[i]
        )

        if count_out == 0:
            # No detections found after filtering
            # Check if ALL enabled stages were configured with process_empty=True
            if num_enabled_stages == num_enabled_stages_with_process_empty:
                 # All enabled stages were set to attempt detection even on empty input
                 # Since no detections were found, the entire cascade failed.
                 # Return the original image as a signal for further processing.
                 image_bypass = image
                 logger.info(
                     "No detections found after filtering, and all enabled stages had process_empty=True. "
                     "Returning original input image as bypass for alternative processing."
                 )
            else:
                 # Some enabled stages were NOT configured with process_empty=True
                 # The lack of detections might be expected behavior for those stages if their input was empty.
                 # Return a blank image to indicate no detections and no fallback expected from this node's configuration.
                 image_bypass = torch.zeros_like(image)
                 logger.info(
                     "No detections found after filtering, and not all enabled stages had process_empty=True. "
                     "Returning blank image as bypass."
                 )
        else:
            # Detections were found, the main SEGS path is active.
            # The bypass image is less relevant here but can still hold the original for consistency.
            # However, the primary output (segs_output) should be used.
            # We can still return the original image as a potential fallback if segs_output is ignored later.
            # Or, return blank to indicate detections exist via the main path.
            # Let's return the original image as a general fallback, but log differently.
            image_bypass = image # Or torch.zeros_like(image) if you prefer blank when detections exist
            logger.info(
                 "Detections found. Primary SEGS output is available. Bypass holds original image as potential fallback if main path is unused."
            )

        # Return all outputs including new preview images and image_bypass
        # Return the original image if no detections were found after filtering
        # NEW: Include masked_fragments_image, remove stageX_previews
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

# Регистрация ноды
NODE_CLASS_MAPPINGS = {"CascadeDetectorAdvanced": CascadeDetector}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CascadeDetectorAdvanced": "🎯 Cascade Detector Advanced (Bbox/Segm, Staged, Enhanced)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
