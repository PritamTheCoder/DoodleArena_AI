"""
DoodlePreprocessor — Image preprocessing pipeline for DoodleNet.

Converts raw canvas images (base64/PIL) into normalized tensors
matching the exact preprocessing used during MobileNetV3 training.

Pipeline: Grayscale → Binarize → BBox Crop → Center/Pad → Resize → Invert → Normalize
"""
import base64
import io
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch


DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True", "TRUE")


class DoodlePreprocessor:
    """
    Processes raw doodle images into model-ready tensors.

    This pipeline is UNIFIED with the training script (train_mobilenetv3.py)
    to ensure inference accuracy matches training performance.

    Attributes:
        target_size: Output spatial resolution (default: 96x96).
        debug_dir: Directory for saving debug visualizations.
    """

    def __init__(
        self,
        target_size: int = 96,
        image_size: Optional[int] = None,
        debug_dir: str = "debug",
    ) -> None:
        self.target_size = image_size if image_size is not None else target_size
        self.debug_dir = debug_dir
        if DEBUG and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir, exist_ok=True)

    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """
        Decode a base64-encoded image string into a PIL RGB image.

        Handles both raw base64 and data-URI formatted strings (e.g.,
        ``data:image/png;base64,...``). RGBA images are composited onto
        a white background to preserve stroke visibility.
        """
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            return bg

        return image.convert("RGB")

    def extract_bbox(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Crop the image to the tight bounding box around ink strokes.

        Uses a threshold of 200 to separate black ink from the white
        background. Returns the original image unchanged if no ink is
        detected.
        """
        mask = img_gray < 200
        coords = np.column_stack(np.where(mask))

        if coords.shape[0] == 0:
            return img_gray

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return img_gray[y_min : y_max + 1, x_min : x_max + 1]

    def center_and_pad(self, img: np.ndarray) -> np.ndarray:
        """
        Pad the image to a square with white (255) background,
        centering the content. Matches training pipeline exactly.
        """
        h, w = img.shape
        size = max(h, w)
        canvas = np.full((size, size), 255, dtype=np.uint8)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        canvas[y_off : y_off + h, x_off : x_off + w] = img
        return canvas

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Full preprocessing pipeline — converts a PIL image to a
        model-ready ``(1, H, W)`` float tensor in ``[-1, 1]`` range.

        Steps:
            1. Convert to grayscale
            2. Binarize (threshold=200)
            3. Crop to bounding box
            4. Center and pad to square
            5. Resize to ``target_size``
            6. Normalize to [0, 1]
            7. Invert (white strokes on black)
            8. Normalize to [-1, 1]
        """
        # 1. Grayscale
        img_gray = np.array(image.convert("L"), dtype=np.uint8)

        # 2. Binarize
        _, img_gray = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

        # 3. Bounding box crop
        img_gray = self.extract_bbox(img_gray)

        # 4. Center + pad
        img_gray = self.center_and_pad(img_gray)

        # 5. Resize
        resized = cv2.resize(
            img_gray,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )

        # 6. Float [0, 1]
        img_f = resized.astype(np.float32) / 255.0

        # 7. Invert
        img_f = 1.0 - img_f

        # 8. Normalize [-1, 1]
        img_f = (img_f - 0.5) / 0.5

        # Add channel dimension → (1, H, W)
        tensor = torch.from_numpy(img_f[np.newaxis, :, :]).float()

        if DEBUG:
            self._save_debug(tensor)

        return tensor

    def preprocess_base64(self, base64_string: str) -> torch.Tensor:
        """Convenience method: decode base64 → preprocess → tensor."""
        image = self.decode_base64_image(base64_string)
        return self.preprocess_image(image)

    def validate_image(self, image: Image.Image) -> bool:
        """Check that an image has reasonable dimensions for processing."""
        w, h = image.size
        return 10 < w < 2000 and 10 < h < 2000

    def _save_debug(self, tensor: torch.Tensor) -> None:
        """Save a debug visualization of the preprocessed tensor."""
        debug_name = f"debug_preprocessed_{int(np.random.random() * 1e9)}.png"
        save_path = os.path.join(self.debug_dir, debug_name)
        save_img = ((tensor.squeeze(0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        cv2.imwrite(save_path, save_img)