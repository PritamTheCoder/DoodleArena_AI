"""
MobileNetV3-Small architecture adapted for doodle recognition.

The model takes single-channel (grayscale) 96×96 images and
outputs logits over 30 doodle classes from Google Quick, Draw!
"""
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class MobileNetV3Doodle(nn.Module):
    """
    MobileNetV3-Small adapted for single-channel doodle input.

    Architecture modifications from the standard MobileNetV3:
        1. First conv: 3 → 1 input channels (grayscale).
        2. Final classifier: 1000 → ``num_classes`` outputs.

    Args:
        num_classes: Number of output classes (default: 30).
    """

    def __init__(self, num_classes: int = 30) -> None:
        super().__init__()

        # Base model without pretrained weights
        self.model = mobilenet_v3_small(weights=None)

        # Replace first conv layer: 3-channel RGB → 1-channel grayscale
        old_first = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=old_first.out_channels,
            kernel_size=old_first.kernel_size,
            stride=old_first.stride,
            padding=old_first.padding,
            bias=old_first.bias is not None,
        )

        # Replace final classifier head
        if hasattr(self.model, "classifier") and isinstance(
            self.model.classifier, nn.Sequential
        ):
            last_idx = len(self.model.classifier) - 1
            last_layer = self.model.classifier[last_idx]

            if isinstance(last_layer, nn.Linear):
                in_features = last_layer.in_features
                self.model.classifier[last_idx] = nn.Linear(in_features, num_classes)
            else:
                self.model.classifier[last_idx] = nn.Linear(1024, num_classes)
        else:
            # Fallback for older torchvision versions
            self.model.classifier = nn.Sequential(
                nn.Linear(576, 1024),
                nn.Hardswish(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, 1, 96, 96) → (B, num_classes)."""
        return self.model(x)


def load_model(
    model_path: str,
    num_classes: int = 30,
    device: Optional[Union[str, torch.device]] = None,
) -> MobileNetV3Doodle:
    """
    Load a trained MobileNetV3Doodle from a checkpoint file.

    Supports both raw state_dict and checkpoint-dict formats
    (with ``model_state_dict`` key).

    Args:
        model_path: Path to the ``.pth`` checkpoint file.
        num_classes: Number of output classes.
        device: Target device (auto-detected if ``None``).

    Returns:
        The loaded model in eval mode on the requested device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileNetV3Doodle(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def get_model_info() -> Dict[str, Any]:
    """
    Return metadata about the MobileNetV3Doodle architecture.

    Useful for health-check and dashboard display.
    """
    model = MobileNetV3Doodle()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "architecture": "MobileNetV3-Small-Doodle",
        "input_shape": [1, 96, 96],
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
