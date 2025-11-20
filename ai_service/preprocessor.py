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
    Preprocessor matching the training pipeline but with a 28x28 quantization
    step to simulate QuickDraw bitmaps.
    
    Revised Pipeline:
      1. Convert to grayscale (Black strokes, White BG)
      2. Binarize
      3. Extract bounding box
      4. INVERT -> (White strokes, Black BG)
      5. Center & pad with BLACK (0) to square
      6. Downscale to 28x28 (INTER_AREA)
      7. Morphology (Dilate properly thickens white strokes now)
      8. Upscale to target_size
      9. Normalize to [-1, 1]
    """

    def __init__(self, target_size=96, image_size=None, debug_dir="debug",
                 simulate_28=True, morphology=True):
        self.target_size = image_size if image_size is not None else target_size
        self.debug_dir = debug_dir
        self.simulate_28 = simulate_28
        self.morphology = morphology
        if DEBUG and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # Base64 image decoder
    # ----------------------------------------------------------------------
    def decode_base64_image(self, base64_string):
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert RGBA -> RGB (white background)
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            image = bg
        else:
            image = image.convert("RGB")

        return image

    # ----------------------------------------------------------------------
    # BBox extraction
    # ----------------------------------------------------------------------
    def extract_bbox(self, img_gray):
        # Input: Black strokes (0) on White (255)
        # Drawn pixels are darker (black) than background (white)
        mask = img_gray < 200
        coords = np.column_stack(np.where(mask))

        if coords.shape[0] == 0:
            return img_gray

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return img_gray[y_min:y_max + 1, x_min:x_max + 1]

    # ----------------------------------------------------------------------
    # Center + pad
    # ----------------------------------------------------------------------
    def center_and_pad(self, img):
        # We assume img is already INVERTED (White strokes on Black BG)
        h, w = img.shape
        size = max(h, w)

        # Pad with 0 (Black) instead of 255
        canvas = np.zeros((size, size), dtype=np.uint8)

        y_off = (size - h) // 2
        x_off = (size - w) // 2

        canvas[y_off:y_off + h, x_off:x_off + w] = img
        return canvas

    # ----------------------------------------------------------------------
    # Full preprocessing
    # ----------------------------------------------------------------------
    def preprocess_image(self, image):
        # 1. Convert to Grayscale
        img_gray = np.array(image.convert("L"), dtype=np.uint8)

        # 2. Threshold to clean up canvas artifacts
        _, bin_img = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
        
        # 3. Crop (Black on White)
        cropped = self.extract_bbox(bin_img)

        # 4. Pad to square (Black on White)
        squared = self.center_and_pad(cropped)

        if self.simulate_28:
            # --- Thicken High-Res ---
            # Erode (since background is white, erode expands the black ink)
            # This makes the lines thick enough to survive the drop to 28x28
            kernel_high_res = np.ones((3, 3), np.uint8)
            squared = cv2.erode(squared, kernel_high_res, iterations=2)

            # 5. Downscale to 28x28 (Black on White)
            down28 = cv2.resize(squared, (28, 28), interpolation=cv2.INTER_AREA)

            # --- Force Binary ---
            # Resize creates gray pixels. We kill them. 
            # Anything not pure white becomes pure black ink.
            _, down28 = cv2.threshold(down28, 200, 255, cv2.THRESH_BINARY)

            # 6. INVERT (White Ink on Black BG)
            down28 = cv2.bitwise_not(down28)

            # 7. Optional Low-Res Thickening
            if self.morphology:
                # One last small pump to ensure connectivity
                kernel = np.ones((1, 1), np.uint8) 
                down28 = cv2.dilate(down28, kernel, iterations=1)

            # 8. Upscale to target size (Nearest Neighbor to preserve blocky look)
            resized = cv2.resize(down28, (self.target_size, self.target_size),
                                 interpolation=cv2.INTER_NEAREST)
        else:
            squared = cv2.bitwise_not(squared)
            resized = cv2.resize(squared, (self.target_size, self.target_size),
                                 interpolation=cv2.INTER_AREA)

        # 9. Normalize
        img_f = resized.astype(np.float32) / 255.0
        
        # 10. Scale to [-1, 1]
        img_f = (img_f - 0.5) / 0.5

        tensor = torch.from_numpy(img_f).unsqueeze(0).float()

        # Debugging
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