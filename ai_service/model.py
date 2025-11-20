import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


class MobileNetV3Doodle(nn.Module):
    """
    MobileNetV3-Small adapted exactly like in the training script.
    Input: (1, 96, 96)
    Output: logits over classes.
    """
    def __init__(self, num_classes=30):
        super().__init__()

        # Load MobileNetV3-Small without pretrained weights
        self.model = mobilenet_v3_small(weights=None)

        # Replace first conv (3→1 channel)
        old_first = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            1,
            old_first.out_channels,
            kernel_size=old_first.kernel_size,
            stride=old_first.stride,
            padding=old_first.padding,
            bias=old_first.bias is not None,
        )

        # Replace final classifier to match training classifier
        if hasattr(self.model, "classifier") and isinstance(self.model.classifier, nn.Sequential):
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
                nn.Linear(1024, num_classes)
            )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------
# Utility Loader
# ---------------------------------------------------

def load_model(model_path, num_classes=30, device=None):
    """
    Load a MobileNetV3Doodle with training-compatible architecture.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MobileNetV3Doodle(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def get_model_info():
    model = MobileNetV3Doodle()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "architecture": "MobileNetV3-Small-Doodle",
        "input_shape": (1, 96, 96),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
