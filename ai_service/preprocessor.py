# preprocessor.py
import base64
import io
import numpy as np
from PIL import Image
import torch
import cv2
import os

DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True", "TRUE")


class DoodlePreprocessor:
    """
    UNIFIED with training pipeline.
    Matches train_mobilenetv3.py exactly.
    """
    
    def __init__(self, target_size=96, image_size=None, debug_dir="debug"):
        self.target_size = image_size if image_size is not None else target_size
        self.debug_dir = debug_dir
        if DEBUG and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir, exist_ok=True)

    def decode_base64_image(self, base64_string):
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            image = bg
        else:
            image = image.convert("RGB")
        return image

    def extract_bbox(self, img_gray):
        """Extract bounding box (matches training exactly)"""
        mask = img_gray < 200  # Black ink (0-200) on white (255)
        coords = np.column_stack(np.where(mask))
        
        if coords.shape[0] == 0:
            return img_gray
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return img_gray[y_min:y_max + 1, x_min:x_max + 1]

    def center_and_pad(self, img):
        """Center and pad with WHITE (255) - MATCHES TRAINING"""
        h, w = img.shape
        size = max(h, w)
        canvas = np.full((size, size), 255, dtype=np.uint8)  # WHITE padding
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        canvas[y_off:y_off + h, x_off:x_off + w] = img
        return canvas

    def preprocess_image(self, image):
        """
        UNIFIED pipeline - matches training exactly.
        No extra morphology, no double thresholding.
        """
        # 1. Convert to grayscale
        img_gray = np.array(image.convert("L"), dtype=np.uint8)
        
        # 2. Binarize (clean up artifacts)
        _, img_gray = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
        
        # 3. Extract bbox
        img_gray = self.extract_bbox(img_gray)
        
        # 4. Center and pad to square (WHITE padding)
        img_gray = self.center_and_pad(img_gray)
        
        # 5. Resize to target size
        resized = cv2.resize(img_gray, (self.target_size, self.target_size),
                            interpolation=cv2.INTER_AREA)
        
        # 6. Convert to float [0,1]
        img_f = resized.astype(np.float32) / 255.0
        
        # 7. INVERT: white strokes on black background
        img_f = 1.0 - img_f
        
        # 8. Normalize to [-1, 1]
        img_f = (img_f - 0.5) / 0.5
        
        # Add channel dimension
        tensor = torch.from_numpy(img_f[np.newaxis, :, :]).float()
        
        if DEBUG:
            debug_name = f"debug_preprocessed_{int(np.random.random()*1e9)}.png"
            save_path = os.path.join(self.debug_dir, debug_name)
            save_img = ((tensor.squeeze(0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            cv2.imwrite(save_path, save_img)
        
        return tensor

    def preprocess_base64(self, base64_string):
        image = self.decode_base64_image(base64_string)
        return self.preprocess_image(image)

    def validate_image(self, image):
        w, h = image.size
        return 10 < w < 2000 and 10 < h < 2000