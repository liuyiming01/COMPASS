# pdf_processor/table_utils/table_image_processor.py
# Description: This script defines a pipeline for processing table images.
import logging
import os
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import cv2
from PIL import Image
from imutils.object_detection import non_max_suppression


class ImageOrientationCorrector:
    """Handles image orientation correction using EAST text detection model."""
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self._get_default_model_path()
        self.net = cv2.dnn.readNet(self.model_path)

    def _get_default_model_path(self) -> str:
        """Get absolute path to default EAST model."""
        module_dir = Path(__file__).parent.resolve()
        return str(module_dir / "models" / "frozen_east_text_detection.pb")

    def correct_orientation(self, image_path: Union[str, Path]) -> np.ndarray:
        """Correct image orientation based on text detection."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        angle = self._detect_text_orientation(image)
        return self._rotate_image(image, angle) if angle != 0 else image

    def _detect_text_orientation(self, image: np.ndarray) -> int:
        """Detect text orientation using EAST model."""
        # Preprocess image and run through the network
        processed_img = self._preprocess_image(image)
        blob = cv2.dnn.blobFromImage(
            processed_img, 1.0, (processed_img.shape[1], processed_img.shape[0]),
            (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        scores, geometry = self.net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

        # Process detections and determine orientation
        boxes = self._process_detections(scores, geometry)
        return self._calculate_rotation_angle(boxes, processed_img.shape)
    
    def _preprocess_image(self, image: np.ndarray, target_size: int = 320) -> np.ndarray:
        """Pad and crop image to target size for model input."""
        height, width = image.shape[:2]
        start_x, start_y = 0, 0
        # Handle images smaller than target size
        if width < target_size or height < target_size:
            pad_x = max(0, target_size - width)
            pad_y = max(0, target_size - height)
            image = cv2.copyMakeBorder(image, pad_y // 2, pad_y - pad_y // 2,
                                      pad_x // 2, pad_x - pad_x // 2,
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))
            # Recalculate coordinates
            height, width = image.shape[:2]
            start_x = (width - target_size) // 2
            start_y = (height - target_size) // 2

        return image[start_y:start_y + target_size, start_x:start_x + target_size]

    def _process_detections(self, scores, geometry, min_confidence: float = 0.5) -> List:
        """Process network outputs and apply non-max suppression."""
        (num_rows, num_cols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x_data0 = geometry[0, 0, y]
            x_data1 = geometry[0, 1, y]
            x_data2 = geometry[0, 2, y]
            x_data3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            for x in range(num_cols):
                if scores_data[x] < min_confidence:
                    continue

                (offset_x, offset_y) = (x * 4.0, y * 4.0)
                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]

                end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
                end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)

                rects.append((start_x, start_y, end_x, end_y))
                confidences.append(scores_data[x])

        return non_max_suppression(np.array(rects), probs=confidences)

    def _calculate_rotation_angle(self, boxes: List, image_shape: tuple) -> int:
        """Determine rotation angle based on detected text boxes."""
        if len(boxes) == 0:
            return 0

        # Calculate average height/width ratio of text boxes
        ratios = []
        for (start_x, start_y, end_x, end_y) in boxes:
            height = end_y - start_y
            width = end_x - start_x
            ratios.append(height / width)

        avg_ratio = np.mean(ratios)
        logging.info(f"Average height/width ratio: {round(avg_ratio, 3)}")

        # Determine rotation angle
        VERTICAL_THRESHOLD = 1.5
        if avg_ratio > VERTICAL_THRESHOLD:
            y_positions = [start_y for _, start_y, _, _ in boxes]
            # avg_y_pos = np.mean(y_positions) / image_shape[0] # image_shape[0] or height
            avg_y_pos = np.mean(y_positions) / height
            return 90 if avg_y_pos < 0.5 else 270
        return 0

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by specified angle."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # For 90/270 degree rotations, swap dimensions
        if abs(angle) in [90, 270]:
            target_width, target_height = height, width
        else:  # 180 degrees
            target_width, target_height = width, height

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        if abs(angle) in [90, 270]:
            rotation_matrix[0, 2] += (target_width - width) / 2
            rotation_matrix[1, 2] += (target_height - height) / 2

        return cv2.warpAffine(image, rotation_matrix, (target_width, target_height))

class TableExtractor:
    """Handles table extraction using RapidTable or other OCR engines."""

    def __init__(self, ocr_engine: str = "rapidtable", device: str = "cpu"):
        self.ocr_engine = ocr_engine
        self.device = device
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize OCR engine based on configuration."""
        if self.ocr_engine == "rapidtable":
            from rapid_table import RapidTable, VisTable, RapidTableInput
            from rapidocr_onnxruntime import RapidOCR
            # from rapidocr_paddle import RapidOCR

            if self.device == "cuda":
                input_args = RapidTableInput(use_cuda=True, device=self.device)
            else:
                input_args = RapidTableInput()
            self.table_engine = RapidTable(input_args)

            self.viser = VisTable()
            self.ocr_engine = RapidOCR()

    def extract(self, image_path: Union[str, Path]) -> str:
        """Extract table content from image."""
        img_path = Path(image_path)
        ocr_result, _ = self.ocr_engine(str(img_path))
        if ocr_result is None:
            logging.error(f"Failed to extract OCR result from image: {img_path}")
            return ""
        table_results = self.table_engine(str(img_path), ocr_result)
        return table_results.pred_html


def process_table_image_pipeline(
    table_image_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    ocr_engine: str = "rapidtable",
    device: str = "cpu"
) -> List[str]:
    """Full pipeline for processing table images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orientation_corrector = ImageOrientationCorrector()
    table_extractor = TableExtractor(ocr_engine, device)
    table_bodys = []

    for image_path in table_image_paths:
        # Correct orientation
        rotated_image = orientation_corrector.correct_orientation(image_path)

        # Save rotated image
        rotated_path = os.path.join(output_dir, f"rotated_{Path(image_path).name}")
        cv2.imwrite(rotated_path, rotated_image)

        # Extract table content
        table_html = table_extractor.extract(rotated_path)
        table_bodys.append(f"\n\n{table_html}\n\n")

    return table_bodys