"""
Utility functions for DoodleNet inference.

Handles class-list management, confidence calculation,
and top-k prediction extraction.
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# Module-level class list (initialized by main.py on startup)
_CLASS_LIST: Optional[List[str]] = None


def set_class_list(class_list: List[str]) -> None:
    """Set the global class list (called once during model loading)."""
    global _CLASS_LIST
    _CLASS_LIST = class_list


def get_class_list() -> List[str]:
    """
    Return the active class list.

    Raises:
        RuntimeError: If the class list has not been initialized yet.
    """
    if _CLASS_LIST is None:
        raise RuntimeError("Class list has not been initialized!")
    return _CLASS_LIST


def get_class_index(class_name: str) -> Optional[int]:
    """
    Look up the index of a class by name (case-insensitive).

    Returns:
        The integer index, or ``None`` if the class is not found.
    """
    target = class_name.lower()
    for i, name in enumerate(get_class_list()):
        if name.lower() == target:
            return i
    return None


def get_class_name(index: int) -> Optional[str]:
    """
    Return the class name at the given index.

    Returns:
        The class name string, or ``None`` if the index is out of range.
    """
    classes = get_class_list()
    if 0 <= index < len(classes):
        return classes[index]
    return None


def calculate_confidence(model_output: torch.Tensor, target_class: str) -> float:
    """
    Compute the softmax confidence for a specific target class.

    Args:
        model_output: Raw logits tensor of shape ``(1, num_classes)``.
        target_class: Name of the target class (case-insensitive).

    Returns:
        Confidence score in ``[0, 1]``, or ``0.0`` if the class is unknown.
    """
    probs = F.softmax(model_output, dim=1)
    idx = get_class_index(target_class)
    if idx is None:
        return 0.0
    return float(probs[0, idx].item())


def get_top_predictions(
    model_output: torch.Tensor, top_k: int = 5
) -> List[Dict[str, object]]:
    """
    Extract the top-k predictions from model output.

    Args:
        model_output: Raw logits tensor of shape ``(1, num_classes)``.
        top_k: Number of top predictions to return.

    Returns:
        A list of dicts with ``"class"`` (str) and ``"confidence"`` (float) keys,
        sorted by descending confidence.
    """
    classes = get_class_list()
    probs = F.softmax(model_output, dim=1)
    top_probs, top_idx = torch.topk(probs, k=min(top_k, len(classes)), dim=1)

    return [
        {"class": classes[int(i.item())], "confidence": float(p.item())}
        for p, i in zip(top_probs[0], top_idx[0])
    ]
