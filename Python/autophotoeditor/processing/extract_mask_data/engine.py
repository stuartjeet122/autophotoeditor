from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from autophotoeditor.core.image_io import decode_color

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("human_mask_extractor")


# ============================================================
# LABELS
# ============================================================

FASHN_LABELS = {
    0: "background",
    1: "face",
    2: "hair",
    3: "top",
    4: "dress",
    5: "skirt",
    6: "pants",
    7: "belt",
    8: "bag",
    9: "hat",
    10: "scarf",
    11: "glasses",
    12: "arms",
    13: "hands",
    14: "legs",
    15: "feet",
    16: "torso",
    17: "jewelry",
}


FACE_LABELS = {
    0: "background",
    1: "skin",
    2: "nose",
    3: "eye_g",
    4: "l_eye",
    5: "r_eye",
    6: "l_brow",
    7: "r_brow",
    8: "l_ear",
    9: "r_ear",
    10: "mouth",
    11: "u_lip",
    12: "l_lip",
    13: "hair",
    14: "hat",
    15: "ear_r",
    16: "neck_l",
    17: "neck",
    18: "cloth",
}


# ============================================================
# MASK NAMES
# ============================================================

MASK_NAMES = [
    "person",
    "face",
    "skin",
    "skin_face",
    "skin_neck",
    "hair",
    "eyes",
    "brows",
    "ears",
    "nose",
    "mouth",
    "lips",
    "neck",
    "clothing",
    "top",
    "dress",
    "skirt",
    "pants",
    "belt",
    "arms",
    "hands",
    "legs",
    "feet",
    "torso",
    "glasses",
    "hat",
    "scarf",
    "bag",
    "jewelry",
    "body",
    "background",
    "non_person",
]


# ============================================================
# CLASS-SPECIFIC THRESHOLDS
# ============================================================

MASK_THRESHOLDS = {
    "face": 0.40,
    "skin": 0.38,
    "skin_face": 0.38,
    "skin_neck": 0.35,

    "hair": 0.35,

    "top": 0.38,
    "dress": 0.38,
    "skirt": 0.38,
    "pants": 0.38,
    "belt": 0.35,

    "arms": 0.30,
    "hands": 0.28,
    "legs": 0.30,
    "feet": 0.28,

    "torso": 0.35,

    "glasses": 0.25,
    "hat": 0.30,
    "scarf": 0.30,
    "bag": 0.30,
    "jewelry": 0.22,

    "eyes": 0.25,
    "brows": 0.25,
    "ears": 0.28,
    "nose": 0.28,
    "mouth": 0.28,
    "lips": 0.25,
    "neck": 0.32,

    "clothing": 0.35,
    "body": 0.32,
}


# ============================================================
# MORPHOLOGY POLICY
# ============================================================

MORPHOLOGY_KERNELS = {
    "face": 3,
    "skin": 3,
    "skin_face": 3,
    "skin_neck": 3,
    "hair": 3,

    "clothing": 5,
    "top": 5,
    "dress": 5,
    "skirt": 5,
    "pants": 5,
    "belt": 3,

    "arms": 3,
    "hands": 2,
    "legs": 3,
    "feet": 2,
    "torso": 3,

    # Tiny/detail regions should generally not be closed.
    "eyes": 0,
    "brows": 0,
    "ears": 2,
    "nose": 0,
    "mouth": 0,
    "lips": 0,
    "glasses": 0,
    "hat": 2,
    "scarf": 3,
    "bag": 3,
    "jewelry": 0,
    "neck": 3,
}


# ============================================================
# LOGGING
# ============================================================

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def fail(message: str) -> None:
    logger.error(message)
    raise SystemExit(1)


# ============================================================
# DEVICE
# ============================================================

def choose_device(requested: str) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            fail("CUDA was requested but is not available.")
        return "cuda"

    if requested == "cpu":
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# GENERAL UTILITIES
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clip_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))

    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    return x1, y1, x2, y2


def expand_box(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    padding: float,
) -> Tuple[int, int, int, int]:

    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    px = int(round(bw * padding))
    py = int(round(bh * padding))

    return clip_box(
        x1 - px,
        y1 - py,
        x2 + px,
        y2 + py,
        width,
        height,
    )


def bbox_from_mask(
    mask: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def mask_coverage(mask: np.ndarray) -> float:
    total = mask.shape[0] * mask.shape[1]

    if total <= 0:
        return 0.0

    return float(np.count_nonzero(mask) / total)


def resize_mask(
    mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:

    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def resize_probability(
    probability: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:

    return cv2.resize(
        probability.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


# ============================================================
# MORPHOLOGY
# ============================================================

def morph_close(
    mask: np.ndarray,
    kernel_size: int,
) -> np.ndarray:

    if kernel_size <= 1:
        return mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel,
    )


def morph_open(
    mask: np.ndarray,
    kernel_size: int,
) -> np.ndarray:

    if kernel_size <= 1:
        return mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel,
    )


def fill_holes(mask: np.ndarray) -> np.ndarray:

    mask = (mask > 0).astype(np.uint8)

    if np.count_nonzero(mask) == 0:
        return mask

    flood = mask.copy()

    h, w = mask.shape

    flood_mask = np.zeros(
        (h + 2, w + 2),
        dtype=np.uint8,
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        1,
    )

    holes = 1 - flood

    result = mask | holes

    return result.astype(np.uint8)


def remove_small_components(
    mask: np.ndarray,
    min_area: int,
) -> np.ndarray:

    binary = (mask > 0).astype(np.uint8)

    if min_area <= 0:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return binary

    result = np.zeros_like(binary)

    for label in range(1, num_labels):

        area = int(stats[label, cv2.CC_STAT_AREA])

        if area >= min_area:
            result[labels == label] = 1

    return result


def refine_mask(
    mask: np.ndarray,
    name: str,
    min_component_area: int,
    enable_hole_fill: bool,
) -> np.ndarray:

    result = (mask > 0).astype(np.uint8)

    kernel_size = MORPHOLOGY_KERNELS.get(
        name,
        0,
    )

    if kernel_size > 1:
        result = morph_close(
            result,
            kernel_size,
        )

    if name not in {
        "eyes",
        "brows",
        "nose",
        "mouth",
        "lips",
        "glasses",
        "jewelry",
    }:
        result = remove_small_components(
            result,
            min_component_area,
        )

    if enable_hole_fill and name in {
        "face",
        "skin",
        "skin_face",
        "skin_neck",
        "clothing",
        "top",
        "dress",
        "skirt",
        "pants",
        "torso",
    }:
        result = fill_holes(result)

    return result


# ============================================================
# SOFT MASK
# ============================================================

def create_soft_mask(
    probability: np.ndarray,
    hard_mask: np.ndarray,
    feather_sigma: float,
) -> np.ndarray:

    probability = np.clip(
        probability.astype(np.float32),
        0.0,
        1.0,
    )

    hard = (
        hard_mask.astype(np.float32)
    )

    # Keep probability information inside the detected area.
    soft = probability * hard

    if feather_sigma > 0:

        blurred = cv2.GaussianBlur(
            soft,
            (0, 0),
            feather_sigma,
        )

        # Keep interior probability strong while allowing
        # a smooth transition around boundaries.
        soft = np.maximum(
            soft * 0.85,
            blurred * 0.70,
        )

    return np.clip(
        soft,
        0.0,
        1.0,
    ).astype(np.float32)


def hard_to_soft_mask(
    mask: np.ndarray,
    sigma: float,
) -> np.ndarray:

    value = mask.astype(np.float32)

    if sigma <= 0:
        return value

    blurred = cv2.GaussianBlur(
        value,
        (0, 0),
        sigma,
    )

    return np.clip(
        blurred,
        0.0,
        1.0,
    )


# ============================================================
# RLE
# ============================================================

def rle_encode(mask: np.ndarray) -> List[int]:
    """
    COCO-style uncompressed RLE.
    Column-major order.
    """

    binary = (
        mask > 0
    ).astype(np.uint8)

    pixels = binary.T.flatten()

    padded = np.concatenate(
        [
            np.array([0], dtype=np.uint8),
            pixels,
            np.array([0], dtype=np.uint8),
        ]
    )

    changes = np.where(
        padded[1:] != padded[:-1]
    )[0]

    runs = changes.copy()

    runs[1::2] -= runs[::2]

    return runs.tolist()


# ============================================================
# GLOBAL MASK CONTAINER
# ============================================================

def create_mask_dictionary(
    height: int,
    width: int,
) -> Dict[str, np.ndarray]:

    return {
        name: np.zeros(
            (height, width),
            dtype=np.uint8,
        )
        for name in MASK_NAMES
    }


def create_soft_mask_dictionary(
    height: int,
    width: int,
) -> Dict[str, np.ndarray]:

    return {
        name: np.zeros(
            (height, width),
            dtype=np.float32,
        )
        for name in MASK_NAMES
    }


# ============================================================
# YOLO PERSON DETECTOR
# ============================================================

class PersonDetector:

    def __init__(
        self,
        model_name: str,
        device: str,
        imgsz: int,
        confidence: float,
        iou: float,
    ):

        if YOLO is None:
            raise RuntimeError(
                "Ultralytics is not installed.\n"
                "Install with:\n"
                "pip install ultralytics"
            )

        logger.info(
            "Loading person segmentation model: %s",
            model_name,
        )

        self.model = YOLO(model_name)

        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou

    def detect(
        self,
        image: np.ndarray,
    ) -> List[Dict[str, Any]]:

        height, width = image.shape[:2]

        logger.info(
            "Running person segmentation at imgsz=%d...",
            self.imgsz,
        )

        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            classes=[0],
            device=self.device,
            retina_masks=True,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        if result.masks is None:
            logger.warning(
                "People detected but segmentation masks are unavailable."
            )
            return []

        boxes = result.boxes
        masks = result.masks.data

        people = []

        for i in range(len(boxes)):

            box = (
                boxes.xyxy[i]
                .detach()
                .cpu()
                .numpy()
            )

            confidence = float(
                boxes.conf[i]
                .detach()
                .cpu()
                .item()
            )

            x1, y1, x2, y2 = box

            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            x1, y1, x2, y2 = clip_box(
                x1,
                y1,
                x2,
                y2,
                width,
                height,
            )

            mask = (
                masks[i]
                .detach()
                .cpu()
                .numpy()
            )

            mask = resize_mask(
                mask,
                width,
                height,
            )

            mask = (
                mask >= 0.50
            ).astype(np.uint8)

            # Person masks get mild cleanup only.
            mask = morph_close(
                mask,
                3,
            )

            mask = remove_small_components(
                mask,
                max(64, int(width * height * 0.00001)),
            )

            people.append(
                {
                    "id": i + 1,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "mask": mask,
                }
            )

        logger.info(
            "Detected %d person instance(s).",
            len(people),
        )

        return people


# ============================================================
# SEGFORMER BASE
# ============================================================

class SegformerParserBase:

    def __init__(
        self,
        model_name: str,
        device: str,
        use_half: bool,
    ):

        self.device = device
        self.use_half = (
            use_half
            and device == "cuda"
            and torch.cuda.is_available()
        )

        self.processor = (
            SegformerImageProcessor
            .from_pretrained(model_name)
        )

        self.model = (
            SegformerForSemanticSegmentation
            .from_pretrained(model_name)
        )

        self.model.to(device)
        self.model.eval()

        self.num_classes = (
            self.model.config.num_labels
        )

    @torch.inference_mode()
    def _predict(
        self,
        crop_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        rgb = cv2.cvtColor(
            crop_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(rgb)

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        if self.use_half:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs = self.model(
                    **inputs
                )

        else:

            outputs = self.model(
                **inputs
            )

        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits,
            size=(
                crop_bgr.shape[0],
                crop_bgr.shape[1],
            ),
            mode="bilinear",
            align_corners=False,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        labels = torch.argmax(
            probabilities,
            dim=0,
        )

        labels = (
            labels
            .detach()
            .cpu()
            .numpy()
        )

        probabilities = (
            probabilities
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        return labels, probabilities


# ============================================================
# FASHN HUMAN PARSER
# ============================================================

class FashnHumanParser(
    SegformerParserBase
):

    def __init__(
        self,
        model_name: str,
        device: str,
        use_half: bool,
    ):

        logger.info(
            "Loading FASHN human parser: %s",
            model_name,
        )

        super().__init__(
            model_name=model_name,
            device=device,
            use_half=use_half,
        )

        logger.info(
            "FASHN parser loaded with %d classes.",
            self.num_classes,
        )

    def predict(
        self,
        crop_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        return self._predict(
            crop_bgr
        )


# ============================================================
# FACE PARSER
# ============================================================

class FaceParser(
    SegformerParserBase
):

    def __init__(
        self,
        model_name: str,
        device: str,
        use_half: bool,
    ):

        logger.info(
            "Loading face parser: %s",
            model_name,
        )

        super().__init__(
            model_name=model_name,
            device=device,
            use_half=use_half,
        )

        logger.info(
            "Face parser loaded with %d classes.",
            self.num_classes,
        )

    def predict(
        self,
        crop_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        return self._predict(
            crop_bgr
        )


# ============================================================
# FACE REGION
# ============================================================

def find_face_region(
    labels: np.ndarray,
    probabilities: np.ndarray,
    confidence: float,
) -> Optional[np.ndarray]:

    if probabilities.ndim != 3:
        return None

    if probabilities.shape[0] <= 1:
        return None

    face_probability = probabilities[1]

    candidate = (
        face_probability >= confidence
    ).astype(np.uint8)

    # If confidence threshold is too strict,
    # use the label map.
    if np.count_nonzero(candidate) < 20:

        candidate = (
            labels == 1
        ).astype(np.uint8)

    if np.count_nonzero(candidate) < 20:
        return None

    # Keep meaningful connected components.
    num_labels, component_labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return candidate

    best_label = None
    best_score = -1.0

    h, w = candidate.shape

    image_center_x = w * 0.5
    image_center_y = h * 0.28

    for label in range(1, num_labels):

        area = float(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < 20:
            continue

        cx, cy = centroids[label]

        distance = (
            ((cx - image_center_x) / max(w, 1)) ** 2
            +
            ((cy - image_center_y) / max(h, 1)) ** 2
        )

        score = (
            area / max(h * w, 1)
            -
            distance * 0.02
        )

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None

    result = (
        component_labels == best_label
    ).astype(np.uint8)

    return result


# ============================================================
# FACE BOX
# ============================================================

def make_face_box(
    local_face_mask: np.ndarray,
    crop_shape: Tuple[int, int, int],
    base_padding: float,
) -> Optional[Tuple[int, int, int, int]]:

    bbox = bbox_from_mask(
        local_face_mask
    )

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    crop_h, crop_w = crop_shape[:2]

    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)

    face_fraction = (
        max(fw, fh)
        /
        max(crop_w, crop_h, 1)
    )

    # Smaller faces need proportionally more context.
    adaptive_padding = (
        base_padding
        +
        max(
            0.0,
            0.20 - face_fraction,
        )
    )

    adaptive_padding = float(
        np.clip(
            adaptive_padding,
            0.15,
            0.55,
        )
    )

    px = int(
        round(fw * adaptive_padding)
    )

    py = int(
        round(fh * adaptive_padding)
    )

    x1 -= px
    y1 -= py
    x2 += px
    y2 += py

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(crop_w, x2)
    y2 = min(crop_h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# PASTE GLOBAL
# ============================================================

def paste_probability_mask(
    probability: np.ndarray,
    box: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> np.ndarray:

    height, width = image_shape

    x1, y1, x2, y2 = box

    crop_width = x2 - x1
    crop_height = y2 - y1

    resized = resize_probability(
        probability,
        crop_width,
        crop_height,
    )

    output = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    output[
        y1:y2,
        x1:x2
    ] = resized

    return output


def paste_binary_mask(
    mask: np.ndarray,
    box: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> np.ndarray:

    height, width = image_shape

    x1, y1, x2, y2 = box

    crop_width = x2 - x1
    crop_height = y2 - y1

    resized = resize_mask(
        mask,
        crop_width,
        crop_height,
    )

    output = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    output[
        y1:y2,
        x1:x2
    ] = (
        resized >= 128
    ).astype(np.uint8)

    return output


# ============================================================
# FASHN MASK MAPPING
# ============================================================

FASHN_MAPPING = {
    "face": [1],
    "hair": [2],
    "top": [3],
    "dress": [4],
    "skirt": [5],
    "pants": [6],
    "belt": [7],
    "bag": [8],
    "hat": [9],
    "scarf": [10],
    "glasses": [11],
    "arms": [12],
    "hands": [13],
    "legs": [14],
    "feet": [15],
    "torso": [16],
    "jewelry": [17],
}


# ============================================================
# FUSE FASHN
# ============================================================

def fuse_fashn_masks(
    hard_masks: Dict[str, np.ndarray],
    soft_masks: Dict[str, np.ndarray],
    probabilities: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    person_mask: np.ndarray,
) -> None:

    image_shape = person_mask.shape

    for name, classes in FASHN_MAPPING.items():

        valid_classes = [
            c
            for c in classes
            if c < probabilities.shape[0]
        ]

        if not valid_classes:
            continue

        probability = np.max(
            probabilities[valid_classes],
            axis=0,
        )

        threshold = MASK_THRESHOLDS.get(
            name,
            0.35,
        )

        global_probability = paste_probability_mask(
            probability,
            crop_box,
            image_shape,
        )

        hard = (
            global_probability >= threshold
        ).astype(np.uint8)

        hard &= person_mask

        hard = refine_mask(
            hard,
            name=name,
            min_component_area=32,
            enable_hole_fill=True,
        )

        hard &= person_mask

        soft = create_soft_mask(
            global_probability,
            hard,
            feather_sigma=1.25,
        )

        soft *= person_mask.astype(
            np.float32
        )

        # Binary masks are unioned.
        hard_masks[name] |= hard

        # Soft masks use maximum confidence.
        soft_masks[name] = np.maximum(
            soft_masks[name],
            soft,
        )


# ============================================================
# FACE FUSION
# ============================================================

def fuse_face_masks(
    hard_masks: Dict[str, np.ndarray],
    soft_masks: Dict[str, np.ndarray],
    probabilities: np.ndarray,
    face_box: Tuple[int, int, int, int],
    person_mask: np.ndarray,
) -> None:

    image_shape = person_mask.shape

    mapping = {
        # Face parser:
        # 1 = skin
        # 2 = nose
        # 3 = glasses
        # 4/5 = eyes
        # 6/7 = brows
        # 8/9/15 = ears
        # 10 = mouth
        # 11/12 = lips
        # 13 = hair
        # 14 = hat
        # 16/17 = neck
        # 18 = cloth

        "skin_face": [1],
        "nose": [2],
        "glasses": [3],
        "eyes": [4, 5],
        "brows": [6, 7],
        "ears": [8, 9, 15],
        "mouth": [10],
        "lips": [11, 12],
        "hair": [13],
        "hat": [14],
        "skin_neck": [16, 17],
        "clothing": [18],
        "neck": [17],
    }

    for name, classes in mapping.items():

        valid_classes = [
            c
            for c in classes
            if c < probabilities.shape[0]
        ]

        if not valid_classes:
            continue

        probability = np.max(
            probabilities[valid_classes],
            axis=0,
        )

        global_probability = paste_probability_mask(
            probability,
            face_box,
            image_shape,
        )

        threshold = MASK_THRESHOLDS.get(
            name,
            0.30,
        )

        hard = (
            global_probability >= threshold
        ).astype(np.uint8)

        hard &= person_mask

        # Face details need very little morphology.
        min_area = 4

        if name in {
            "skin_face",
            "skin_neck",
            "neck",
        }:
            min_area = 16

        hard = refine_mask(
            hard,
            name=name,
            min_component_area=min_area,
            enable_hole_fill=(
                name in {
                    "skin_face",
                    "skin_neck",
                    "neck",
                }
            ),
        )

        hard &= person_mask

        soft = create_soft_mask(
            global_probability,
            hard,
            feather_sigma=0.8,
        )

        soft *= person_mask.astype(
            np.float32
        )

        hard_masks[name] |= hard

        soft_masks[name] = np.maximum(
            soft_masks[name],
            soft,
        )


# ============================================================
# DERIVED MASKS
# ============================================================

def build_derived_masks(
    hard_masks: Dict[str, np.ndarray],
    soft_masks: Dict[str, np.ndarray],
    person_mask: np.ndarray,
) -> None:

    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    face_parts = [
        "skin_face",
        "eyes",
        "brows",
        "ears",
        "nose",
        "mouth",
        "lips",
    ]

    face = np.zeros_like(
        person_mask
    )

    face_soft = np.zeros_like(
        soft_masks["face"]
    )

    for name in face_parts:
        face |= hard_masks[name]
        face_soft = np.maximum(
            face_soft,
            soft_masks[name],
        )

    face &= person_mask

    hard_masks["face"] = face

    soft_masks["face"] = (
        face_soft
        *
        person_mask.astype(np.float32)
    )

    # --------------------------------------------------------
    # SKIN
    # --------------------------------------------------------

    skin = (
        hard_masks["skin_face"]
        |
        hard_masks["skin_neck"]
    )

    skin &= person_mask

    skin_soft = np.maximum(
        soft_masks["skin_face"],
        soft_masks["skin_neck"],
    )

    hard_masks["skin"] = skin

    soft_masks["skin"] = (
        skin_soft
        *
        person_mask.astype(np.float32)
    )

    # --------------------------------------------------------
    # CLOTHING
    # --------------------------------------------------------

    clothing_parts = [
        "top",
        "dress",
        "skirt",
        "pants",
        "belt",
        "scarf",
        "clothing",
    ]

    clothing = np.zeros_like(
        person_mask
    )

    clothing_soft = np.zeros_like(
        soft_masks["clothing"]
    )

    for name in clothing_parts:
        clothing |= hard_masks[name]
        clothing_soft = np.maximum(
            clothing_soft,
            soft_masks[name],
        )

    clothing &= person_mask

    hard_masks["clothing"] = clothing

    soft_masks["clothing"] = (
        clothing_soft
        *
        person_mask.astype(np.float32)
    )

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    body_parts = [
        "face",
        "hair",
        "clothing",
        "arms",
        "hands",
        "legs",
        "feet",
        "torso",
        "neck",
    ]

    body = np.zeros_like(
        person_mask
    )

    body_soft = np.zeros_like(
        soft_masks["body"]
    )

    for name in body_parts:
        body |= hard_masks[name]
        body_soft = np.maximum(
            body_soft,
            soft_masks[name],
        )

    body &= person_mask

    hard_masks["body"] = body

    soft_masks["body"] = (
        body_soft
        *
        person_mask.astype(np.float32)
    )

    # --------------------------------------------------------
    # BACKGROUND / NON-PERSON
    # --------------------------------------------------------

    hard_masks["background"] = (
        person_mask == 0
    ).astype(np.uint8)

    hard_masks["non_person"] = (
        person_mask == 0
    ).astype(np.uint8)

    soft_masks["background"] = (
        hard_masks["background"]
        .astype(np.float32)
    )

    soft_masks["non_person"] = (
        hard_masks["non_person"]
        .astype(np.float32)
    )


# ============================================================
# PERSON-SPECIFIC PROCESSING
# ============================================================

def process_person(
    image: np.ndarray,
    person: Dict[str, Any],
    human_parser: FashnHumanParser,
    face_parser: Optional[FaceParser],
    face_threshold: float,
    face_padding: float,
    global_shape: Tuple[int, int],
    feather_sigma: float,
) -> Tuple[
    Dict[str, Any],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:

    height, width = global_shape

    person_id = person["id"]

    person_mask = person["mask"]

    original_box = person["bbox"]

    crop_box = expand_box(
        original_box,
        width,
        height,
        padding=0.08,
    )

    x1, y1, x2, y2 = crop_box

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        raise RuntimeError(
            f"Empty crop for person {person_id}"
        )

    logger.info(
        "Person %d: parsing crop %dx%d",
        person_id,
        crop.shape[1],
        crop.shape[0],
    )

    # --------------------------------------------------------
    # LOCAL MASKS
    # --------------------------------------------------------

    local_hard = create_mask_dictionary(
        crop.shape[0],
        crop.shape[1],
    )

    local_soft = create_soft_mask_dictionary(
        crop.shape[0],
        crop.shape[1],
    )

    local_person_mask = person_mask[
        y1:y2,
        x1:x2,
    ]

    # --------------------------------------------------------
    # FASHN
    # --------------------------------------------------------

    fashn_labels, fashn_probabilities = (
        human_parser.predict(crop)
    )

    # --------------------------------------------------------
    # FASHN LOCAL FUSION
    # --------------------------------------------------------

    for name, classes in FASHN_MAPPING.items():

        valid_classes = [
            c
            for c in classes
            if c < fashn_probabilities.shape[0]
        ]

        if not valid_classes:
            continue

        probability = np.max(
            fashn_probabilities[valid_classes],
            axis=0,
        )

        threshold = MASK_THRESHOLDS.get(
            name,
            0.35,
        )

        hard = (
            probability >= threshold
        ).astype(np.uint8)

        hard &= local_person_mask

        min_area = 32

        if name in {
            "hands",
            "feet",
            "jewelry",
            "glasses",
        }:
            min_area = 4

        hard = refine_mask(
            hard,
            name=name,
            min_component_area=min_area,
            enable_hole_fill=True,
        )

        hard &= local_person_mask

        soft = create_soft_mask(
            probability,
            hard,
            feather_sigma=feather_sigma,
        )

        soft *= local_person_mask.astype(
            np.float32
        )

        local_hard[name] = hard

        local_soft[name] = soft

    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    local_face_mask = find_face_region(
        fashn_labels,
        fashn_probabilities,
        confidence=face_threshold,
    )

    face_box_local = None
    face_box_global = None

    if local_face_mask is not None:

        face_box_local = make_face_box(
            local_face_mask,
            crop.shape,
            base_padding=face_padding,
        )

    if (
        face_box_local is not None
        and face_parser is not None
    ):

        fx1, fy1, fx2, fy2 = face_box_local

        face_crop = crop[
            fy1:fy2,
            fx1:fx2,
        ]

        if face_crop.size > 0:

            logger.debug(
                "Person %d: face crop %dx%d",
                person_id,
                face_crop.shape[1],
                face_crop.shape[0],
            )

            (
                face_labels,
                face_probabilities,
            ) = face_parser.predict(
                face_crop
            )

            face_box_global = (
                x1 + fx1,
                y1 + fy1,
                x1 + fx2,
                y1 + fy2,
            )

            # ------------------------------------------------
            # FACE FUSION LOCAL
            # ------------------------------------------------

            face_mapping = {
                "skin_face": [1],
                "nose": [2],
                "glasses": [3],
                "eyes": [4, 5],
                "brows": [6, 7],
                "ears": [8, 9, 15],
                "mouth": [10],
                "lips": [11, 12],
                "hair": [13],
                "hat": [14],
                "skin_neck": [16, 17],
                "neck": [17],
                "clothing": [18],
            }

            face_person_mask = local_person_mask[
                fy1:fy2,
                fx1:fx2,
            ]

            for name, classes in face_mapping.items():

                valid_classes = [
                    c
                    for c in classes
                    if c < face_probabilities.shape[0]
                ]

                if not valid_classes:
                    continue

                probability = np.max(
                    face_probabilities[
                        valid_classes
                    ],
                    axis=0,
                )

                threshold = MASK_THRESHOLDS.get(
                    name,
                    0.30,
                )

                hard = (
                    probability >= threshold
                ).astype(np.uint8)

                hard &= face_person_mask

                min_area = (
                    4
                    if name not in {
                        "skin_face",
                        "skin_neck",
                        "neck",
                    }
                    else 12
                )

                hard = refine_mask(
                    hard,
                    name=name,
                    min_component_area=min_area,
                    enable_hole_fill=(
                        name in {
                            "skin_face",
                            "skin_neck",
                            "neck",
                        }
                    ),
                )

                hard &= face_person_mask

                soft = create_soft_mask(
                    probability,
                    hard,
                    feather_sigma=0.8,
                )

                soft *= face_person_mask.astype(
                    np.float32
                )

                # Paste face-parser output into local crop.
                full_hard = np.zeros_like(
                    local_person_mask
                )

                full_soft = np.zeros_like(
                    local_soft[name]
                )

                full_hard[
                    fy1:fy2,
                    fx1:fx2
                ] = hard

                full_soft[
                    fy1:fy2,
                    fx1:fx2
                ] = soft

                local_hard[name] |= full_hard

                local_soft[name] = np.maximum(
                    local_soft[name],
                    full_soft,
                )

    # --------------------------------------------------------
    # DERIVED LOCAL MASKS
    # --------------------------------------------------------

    build_derived_masks(
        local_hard,
        local_soft,
        local_person_mask,
    )

    # --------------------------------------------------------
    # FORCE PERSON
    # --------------------------------------------------------

    local_hard["person"] = (
        local_person_mask.copy()
    )

    local_soft["person"] = (
        local_person_mask.astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    local_hard["background"] = (
        local_person_mask == 0
    ).astype(np.uint8)

    local_hard["non_person"] = (
        local_person_mask == 0
    ).astype(np.uint8)

    local_soft["background"] = (
        local_hard["background"]
        .astype(np.float32)
    )

    local_soft["non_person"] = (
        local_hard["non_person"]
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # PASTE LOCAL MASKS INTO GLOBAL MASKS
    # --------------------------------------------------------

    global_hard = create_mask_dictionary(
        height,
        width,
    )

    global_soft = create_soft_mask_dictionary(
        height,
        width,
    )

    global_hard["person"] = person_mask.copy()

    global_soft["person"] = (
        person_mask.astype(np.float32)
    )

    for name in MASK_NAMES:

        if name == "person":
            continue

        if name in {
            "background",
            "non_person",
        }:
            continue

        local_mask = local_hard[name]

        local_probability = local_soft[name]

        global_hard[name][
            y1:y2,
            x1:x2
        ] = local_mask

        global_soft[name][
            y1:y2,
            x1:x2
        ] = local_probability

        global_hard[name] &= person_mask

        global_soft[name] *= (
            person_mask.astype(np.float32)
        )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {
        "id": int(person_id),

        "confidence": float(
            person["confidence"]
        ),

        "bbox": {
            "x1": int(original_box[0]),
            "y1": int(original_box[1]),
            "x2": int(original_box[2]),
            "y2": int(original_box[3]),
            "width": int(
                original_box[2]
                -
                original_box[0]
            ),
            "height": int(
                original_box[3]
                -
                original_box[1]
            ),
        },

        "crop_bbox": {
            "x1": int(crop_box[0]),
            "y1": int(crop_box[1]),
            "x2": int(crop_box[2]),
            "y2": int(crop_box[3]),
        },

        "face_bbox": (
            {
                "x1": int(face_box_global[0]),
                "y1": int(face_box_global[1]),
                "x2": int(face_box_global[2]),
                "y2": int(face_box_global[3]),
            }
            if face_box_global is not None
            else None
        ),

        "mask_area": mask_area(
            person_mask
        ),

        "image_coverage": mask_coverage(
            person_mask
        ),
    }

    return (
        metadata,
        global_hard,
        global_soft,
    )


# ============================================================
# MASK STATISTICS
# ============================================================

def mask_statistics(
    hard: np.ndarray,
    soft: Optional[np.ndarray],
) -> Dict[str, Any]:

    bbox = bbox_from_mask(
        hard
    )

    result = {
        "pixel_count": mask_area(
            hard
        ),

        "coverage": mask_coverage(
            hard
        ),

        "bbox": (
            {
                "x1": bbox[0],
                "y1": bbox[1],
                "x2": bbox[2],
                "y2": bbox[3],
                "width": bbox[2] - bbox[0],
                "height": bbox[3] - bbox[1],
            }
            if bbox is not None
            else None
        ),
    }

    if soft is not None:

        values = soft[
            hard > 0
        ]

        if values.size:

            result.update(
                {
                    "mean_confidence": float(
                        np.mean(values)
                    ),
                    "median_confidence": float(
                        np.median(values)
                    ),
                    "max_confidence": float(
                        np.max(values)
                    ),
                }
            )

        else:

            result.update(
                {
                    "mean_confidence": 0.0,
                    "median_confidence": 0.0,
                    "max_confidence": 0.0,
                }
            )

    return result


# ============================================================
# SAVE BINARY MASKS
# ============================================================

def save_binary_masks(
    masks: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:

    ensure_dir(
        output_dir
    )

    for name, mask in masks.items():

        output = (
            mask.astype(np.uint8)
            *
            255
        )

        path = (
            output_dir
            /
            f"{name}.png"
        )

        cv2.imwrite(
            str(path),
            output,
        )


# ============================================================
# SAVE SOFT MASKS
# ============================================================

def save_soft_masks(
    masks: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:

    ensure_dir(
        output_dir
    )

    for name, mask in masks.items():

        output = (
            np.clip(
                mask,
                0.0,
                1.0,
            )
            *
            255.0
        ).astype(np.uint8)

        path = (
            output_dir
            /
            f"{name}.png"
        )

        cv2.imwrite(
            str(path),
            output,
        )


# ============================================================
# VISUAL PREVIEW
# ============================================================

def create_visual_preview(
    image: np.ndarray,
    masks: Dict[str, np.ndarray],
    opacity: float,
) -> np.ndarray:

    preview = image.astype(
        np.float32
    )

    colors = {
        "person": (255, 255, 0),
        "face": (0, 180, 255),
        "skin": (80, 170, 255),
        "skin_face": (80, 170, 255),
        "skin_neck": (80, 140, 255),
        "hair": (80, 50, 20),

        "eyes": (255, 0, 255),
        "brows": (120, 60, 30),
        "ears": (0, 120, 255),
        "nose": (0, 220, 255),
        "mouth": (180, 50, 180),
        "lips": (180, 50, 180),

        "clothing": (255, 80, 0),
        "top": (255, 100, 0),
        "dress": (255, 120, 0),
        "skirt": (255, 140, 0),
        "pants": (255, 80, 40),

        "arms": (0, 255, 120),
        "hands": (0, 255, 200),
        "legs": (180, 100, 255),
        "feet": (100, 255, 255),

        "glasses": (255, 255, 255),
        "hat": (255, 100, 100),
        "bag": (100, 255, 100),
        "jewelry": (255, 200, 0),

        "neck": (255, 150, 50),
        "torso": (255, 80, 80),
        "background": (0, 0, 0),
    }

    priority = [
        "person",

        "clothing",
        "top",
        "dress",
        "skirt",
        "pants",
        "arms",
        "hands",
        "legs",
        "feet",
        "torso",

        "hair",

        "face",
        "skin",
        "skin_face",
        "skin_neck",
        "neck",

        "eyes",
        "brows",
        "ears",
        "nose",
        "mouth",
        "lips",

        "glasses",
        "hat",
        "bag",
        "jewelry",
    ]

    for name in priority:

        if name not in masks:
            continue

        mask = masks[name]

        if np.count_nonzero(mask) == 0:
            continue

        color = np.array(
            colors.get(
                name,
                (255, 255, 255),
            ),
            dtype=np.float32,
        )

        alpha = (
            mask.astype(np.float32)
            *
            opacity
        )

        alpha = alpha[:, :, None]

        preview = (
            preview * (1.0 - alpha)
            +
            color * alpha
        )

    return np.clip(
        preview,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# ATOMIC JSON
# ============================================================

def save_json_atomic(
    data: Dict[str, Any],
    path: Path,
    pretty: bool,
) -> None:

    ensure_dir(
        path.parent
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=(
                2
                if pretty
                else None
            ),
        )

    temporary.replace(
        path
    )


# ============================================================
# JSON MASK DATA
# ============================================================

def make_json_masks(
    hard_masks: Dict[str, np.ndarray],
    soft_masks: Dict[str, np.ndarray],
    include_rle: bool,
    binary_dir: Optional[Path],
    soft_dir: Optional[Path],
) -> Dict[str, Any]:

    output = {}

    for name, mask in hard_masks.items():

        item = mask_statistics(
            mask,
            soft_masks.get(name),
        )

        if binary_dir is not None:

            item["binary_file"] = str(
                binary_dir
                /
                f"{name}.png"
            )

        if soft_dir is not None:

            item["soft_file"] = str(
                soft_dir
                /
                f"{name}.png"
            )

        if include_rle:

            item["rle"] = rle_encode(
                mask
            )

        output[name] = item

    return output


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    input_path: Path,
    output_json: Path,

    person_model: str,
    human_model: str,
    face_model: str,

    device: str,

    person_imgsz: int,
    person_conf: float,
    person_iou: float,

    face_threshold: float,
    face_padding: float,

    visual_path: Optional[Path],
    visual_dir: Optional[Path],

    binary_dir: Optional[Path],
    soft_dir: Optional[Path],
    person_masks_dir: Optional[Path],

    mask_opacity: float,
    jpeg_quality: int,

    pretty: bool,
    no_rle: bool,

    save_person_previews: bool,

    use_half: bool = False,
    feather_sigma: float = 1.5,
) -> None:

    # ========================================================
    # INPUT
    # ========================================================

    if not input_path.exists():

        fail(
            f"Input image does not exist:\n"
            f"{input_path}"
        )

    logger.info(
        "Loading image: %s",
        input_path,
    )

    try:

        image = decode_color(
            input_path
        )

    except Exception:

        fail(
            f"Could not read image:\n"
            f"{input_path}"
        )

    if image is None:
        fail(
            "Image decoder returned None."
        )

    if image.ndim != 3:
        fail(
            "Expected a 3-channel color image."
        )

    height, width = image.shape[:2]

    logger.info(
        "Image: %dx%d",
        width,
        height,
    )

    # ========================================================
    # PERSON DETECTOR
    # ========================================================

    logger.info(
        "Device: %s",
        device,
    )

    person_detector = PersonDetector(
        model_name=person_model,
        device=device,
        imgsz=person_imgsz,
        confidence=person_conf,
        iou=person_iou,
    )

    # ========================================================
    # PERSON DETECTION
    # ========================================================

    people = person_detector.detect(
        image
    )

    if not people:

        logger.warning(
            "No people were detected."
        )

        hard_masks = create_mask_dictionary(
            height,
            width,
        )

        soft_masks = create_soft_mask_dictionary(
            height,
            width,
        )

        hard_masks["background"][:] = 1
        hard_masks["non_person"][:] = 1

        soft_masks["background"][:] = 1.0
        soft_masks["non_person"][:] = 1.0

        data = {
            "version": "4.0",

            "image": {
                "path": str(input_path),
                "width": width,
                "height": height,
                "channels": 3,
                "color_format": "BGR",
            },

            "device": device,

            "models": {
                "person": person_model,
                "human_parser": None,
                "face_parser": None,
            },

            "people_count": 0,

            "people": [],

            "masks": make_json_masks(
                hard_masks,
                soft_masks,
                include_rle=not no_rle,
                binary_dir=binary_dir,
                soft_dir=soft_dir,
            ),
        }

        if binary_dir is not None:
            save_binary_masks(
                hard_masks,
                binary_dir,
            )

        if soft_dir is not None:
            save_soft_masks(
                soft_masks,
                soft_dir,
            )

        save_json_atomic(
            data,
            output_json,
            pretty,
        )

        return

    # ========================================================
    # LOAD HUMAN PARSER ONLY WHEN NEEDED
    # ========================================================

    human_parser = FashnHumanParser(
        model_name=human_model,
        device=device,
        use_half=use_half,
    )

    # ========================================================
    # LOAD FACE PARSER
    # ========================================================

    face_parser: Optional[FaceParser] = None

    if face_model:

        face_parser = FaceParser(
            model_name=face_model,
            device=device,
            use_half=use_half,
        )

    # ========================================================
    # GLOBAL MASKS
    # ========================================================

    global_hard = create_mask_dictionary(
        height,
        width,
    )

    global_soft = create_soft_mask_dictionary(
        height,
        width,
    )

    person_metadata = []

    # ========================================================
    # PROCESS PEOPLE
    # ========================================================

    for index, person in enumerate(
        people,
        start=1,
    ):

        logger.info(
            "[%d/%d] Processing person %d...",
            index,
            len(people),
            person["id"],
        )

        (
            metadata,
            person_hard,
            person_soft,
        ) = process_person(
            image=image,
            person=person,
            human_parser=human_parser,
            face_parser=face_parser,
            face_threshold=face_threshold,
            face_padding=face_padding,
            global_shape=(
                height,
                width,
            ),
            feather_sigma=feather_sigma,
        )

        person_metadata.append(
            metadata
        )

        # ----------------------------------------------------
        # SAVE PER-PERSON MASKS
        # ----------------------------------------------------

        if person_masks_dir is not None:

            person_root = (
                person_masks_dir
                /
                f"person_{person['id']:03d}"
            )

            binary_person_dir = (
                person_root
                /
                "binary"
            )

            soft_person_dir = (
                person_root
                /
                "soft"
            )

            save_binary_masks(
                person_hard,
                binary_person_dir,
            )

            save_soft_masks(
                person_soft,
                soft_person_dir,
            )

        # ----------------------------------------------------
        # MERGE INTO GLOBAL
        # ----------------------------------------------------

        for name in MASK_NAMES:

            if name in {
                "background",
                "non_person",
            }:
                continue

            global_hard[name] |= (
                person_hard[name]
            )

            global_soft[name] = np.maximum(
                global_soft[name],
                person_soft[name],
            )

    # ========================================================
    # PERSON UNION
    # ========================================================

    all_people_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for person in people:

        all_people_mask |= (
            person["mask"]
        )

    global_hard["person"] = (
        all_people_mask
    )

    global_soft["person"] = (
        all_people_mask.astype(
            np.float32
        )
    )

    # ========================================================
    # FINAL GLOBAL DERIVED MASKS
    # ========================================================

    build_derived_masks(
        global_hard,
        global_soft,
        all_people_mask,
    )

    # ========================================================
    # RESTRICT ALL PERSON MASKS
    # ========================================================

    for name in MASK_NAMES:

        if name in {
            "background",
            "non_person",
        }:
            continue

        global_hard[name] &= (
            all_people_mask
        )

        global_soft[name] *= (
            all_people_mask.astype(
                np.float32
            )
        )

    # ========================================================
    # BACKGROUND
    # ========================================================

    global_hard["background"] = (
        all_people_mask == 0
    ).astype(np.uint8)

    global_hard["non_person"] = (
        all_people_mask == 0
    ).astype(np.uint8)

    global_soft["background"] = (
        global_hard["background"]
        .astype(np.float32)
    )

    global_soft["non_person"] = (
        global_hard["non_person"]
        .astype(np.float32)
    )

    # ========================================================
    # SAVE GLOBAL BINARY MASKS
    # ========================================================

    if binary_dir is not None:

        logger.info(
            "Saving binary masks: %s",
            binary_dir,
        )

        save_binary_masks(
            global_hard,
            binary_dir,
        )

    # ========================================================
    # SAVE GLOBAL SOFT MASKS
    # ========================================================

    if soft_dir is not None:

        logger.info(
            "Saving soft masks: %s",
            soft_dir,
        )

        save_soft_masks(
            global_soft,
            soft_dir,
        )

    # ========================================================
    # VISUAL PREVIEW
    # ========================================================

    if visual_path is not None:

        logger.info(
            "Creating visual preview..."
        )

        preview = create_visual_preview(
            image=image,
            masks=global_hard,
            opacity=mask_opacity,
        )

        ensure_dir(
            visual_path.parent
        )

        cv2.imwrite(
            str(visual_path),
            preview,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                jpeg_quality,
            ],
        )

        logger.info(
            "Preview saved: %s",
            visual_path,
        )

    # ========================================================
    # OPTIONAL PERSON PREVIEW
    # ========================================================

    if (
        save_person_previews
        and visual_dir is not None
    ):

        ensure_dir(
            visual_dir
        )

        for person in people:

            pid = person["id"]

            preview = image.copy()

            overlay = np.zeros_like(
                preview
            )

            overlay[:, :, 1] = (
                person["mask"]
                *
                255
            )

            preview = cv2.addWeighted(
                preview,
                0.65,
                overlay,
                0.35,
                0,
            )

            box = person["bbox"]

            cv2.rectangle(
                preview,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 255, 255),
                max(
                    2,
                    min(width, height) // 1000,
                ),
            )

            output_path = (
                visual_dir
                /
                f"person_{pid:03d}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                preview,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    jpeg_quality,
                ],
            )

    # ========================================================
    # JSON
    # ========================================================

    logger.info(
        "Building JSON..."
    )

    data = {
        "version": "4.0",

        "image": {
            "path": str(input_path),
            "width": width,
            "height": height,
            "channels": 3,
            "color_format": "BGR",
        },

        "device": device,

        "models": {
            "person": person_model,
            "human_parser": human_model,
            "face_parser": face_model,
        },

        "parameters": {
            "person_imgsz": person_imgsz,
            "person_confidence": person_conf,
            "person_iou": person_iou,

            "face_threshold": face_threshold,
            "face_padding": face_padding,

            "feather_sigma": feather_sigma,

            "use_half": bool(
                use_half
                and device == "cuda"
            ),
        },

        "mask_schema": {
            "binary": "0 = outside, 1 = inside",
            "soft": "0..1 confidence/alpha",
            "person_masks": (
                "Per-person masks are stored separately "
                "when --person-masks-dir is provided."
            ),
            "background": (
                "Pixels not belonging to detected persons."
            ),
            "non_person": (
                "Alias for the detected-person complement."
            ),
        },

        "people_count": len(
            person_metadata
        ),

        "people": person_metadata,

        "masks": make_json_masks(
            global_hard,
            global_soft,
            include_rle=not no_rle,
            binary_dir=binary_dir,
            soft_dir=soft_dir,
        ),
    }

    save_json_atomic(
        data,
        output_json,
        pretty,
    )

    logger.info(
        "JSON saved: %s",
        output_json,
    )

    logger.info(
        "Processing complete."
    )


# ============================================================
# ARGUMENTS
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "High-quality multi-person human "
            "and face mask extraction pipeline."
        )
    )

    # --------------------------------------------------------
    # INPUT / OUTPUT
    # --------------------------------------------------------

    parser.add_argument(
        "input",
        type=Path,
        help="Input image",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output JSON",
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    parser.add_argument(
        "--person-model",
        default="yolo26l-seg.pt",
        help=(
            "Ultralytics person segmentation model."
        ),
    )

    parser.add_argument(
        "--human-model",
        default="fashn-ai/fashn-human-parser",
        help="FASHN human parser model.",
    )

    parser.add_argument(
        "--face-model",
        default="jonathandinu/face-parsing",
        help="Face parser model.",
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--half",
        action="store_true",
        help=(
            "Use FP16 mixed precision on CUDA."
        ),
    )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    parser.add_argument(
        "--person-imgsz",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--person-iou",
        type=float,
        default=0.50,
    )

    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--face-padding",
        type=float,
        default=0.30,
    )

    # --------------------------------------------------------
    # SOFT MASK
    # --------------------------------------------------------

    parser.add_argument(
        "--feather-sigma",
        type=float,
        default=1.25,
        help=(
            "Soft mask boundary feathering."
        ),
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    parser.add_argument(
        "--visual",
        type=Path,
        default=None,
        help="Combined visual mask preview.",
    )

    parser.add_argument(
        "--visual-dir",
        type=Path,
        default=None,
        help="Individual person preview directory.",
    )

    parser.add_argument(
        "--binary-dir",
        type=Path,
        default=None,
        help="Global binary mask directory.",
    )

    parser.add_argument(
        "--soft-dir",
        type=Path,
        default=None,
        help="Global soft mask directory.",
    )

    parser.add_argument(
        "--person-masks-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing per-person "
            "binary and soft masks."
        ),
    )

    parser.add_argument(
        "--mask-opacity",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
    )

    parser.add_argument(
        "--no-rle",
        action="store_true",
        help="Do not store RLE masks in JSON.",
    )

    parser.add_argument(
        "--save-person-previews",
        action="store_true",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )

    return parser


# ============================================================
# VALIDATION
# ============================================================

def validate_args(
    args: argparse.Namespace,
) -> None:

    if not (
        0.0 < args.person_conf <= 1.0
    ):
        fail(
            "--person-conf must be between 0 and 1."
        )

    if not (
        0.0 < args.person_iou <= 1.0
    ):
        fail(
            "--person-iou must be between 0 and 1."
        )

    if not (
        0.0 < args.face_threshold <= 1.0
    ):
        fail(
            "--face-threshold must be between 0 and 1."
        )

    if args.face_padding < 0:
        fail(
            "--face-padding cannot be negative."
        )

    if args.feather_sigma < 0:
        fail(
            "--feather-sigma cannot be negative."
        )

    if not (
        0.0 <= args.mask_opacity <= 1.0
    ):
        fail(
            "--mask-opacity must be between 0 and 1."
        )

    if not (
        1 <= args.jpeg_quality <= 100
    ):
        fail(
            "--jpeg-quality must be between 1 and 100."
        )

    if args.person_imgsz < 320:
        fail(
            "--person-imgsz should be at least 320."
        )


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    configure_logging(
        args.verbose
    )

    validate_args(
        args
    )

    device = choose_device(
        args.device
    )

    if device == "cuda":

        logger.info(
            "CUDA device: %s",
            torch.cuda.get_device_name(0),
        )

        torch.set_float32_matmul_precision(
            "high"
        )

        if args.half:
            logger.info(
                "CUDA FP16 mixed precision enabled."
            )

    else:

        logger.info(
            "Running on CPU."
        )

        if args.half:
            logger.warning(
                "--half ignored because device is CPU."
            )

    try:

        run_pipeline(
            input_path=args.input,
            output_json=args.output,

            person_model=args.person_model,
            human_model=args.human_model,
            face_model=args.face_model,

            device=device,

            person_imgsz=args.person_imgsz,
            person_conf=args.person_conf,
            person_iou=args.person_iou,

            face_threshold=args.face_threshold,
            face_padding=args.face_padding,

            visual_path=args.visual,
            visual_dir=args.visual_dir,

            binary_dir=args.binary_dir,
            soft_dir=args.soft_dir,
            person_masks_dir=args.person_masks_dir,

            mask_opacity=args.mask_opacity,
            jpeg_quality=args.jpeg_quality,

            pretty=args.pretty,
            no_rle=args.no_rle,

            save_person_previews=args.save_person_previews,

            use_half=args.half,
            feather_sigma=args.feather_sigma,
        )

    except KeyboardInterrupt:

        logger.error(
            "Interrupted."
        )

        raise SystemExit(130)

    except Exception as exc:

        logger.exception(
            "Fatal error: %s",
            exc,
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()