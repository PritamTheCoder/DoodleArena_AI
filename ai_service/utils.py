# utils.py
import torch
import torch.nn.functional as F

# This will be set by main.py after model loads checkpoint
CLASS_LIST = None


def set_class_list(class_list):
    """Called by main.py after loading checkpoint."""
    global CLASS_LIST
    CLASS_LIST = class_list


def get_class_list():
    """Return the class list exactly as given by checkpoint."""
    if CLASS_LIST is None:
        raise RuntimeError("Class list is not initialized!")
    return CLASS_LIST


def get_class_index(class_name):
    """Return index of class (case-insensitive)."""
    cls = class_name.lower()
    for i, c in enumerate(get_class_list()):
        if c.lower() == cls:
            return i
    return None


def get_class_name(index):
    classes = get_class_list()
    if 0 <= index < len(classes):
        return classes[index]
    return None


def calculate_confidence(model_output, target_class):
    """Softmax confidence for the requested class."""
    probs = F.softmax(model_output, dim=1)

    idx = get_class_index(target_class)
    if idx is None:
        return 0.0

    return float(probs[0, idx].item())


def get_top_predictions(model_output, top_k=5):
    """Return top-k predictions."""
    classes = get_class_list()
    probs = F.softmax(model_output, dim=1)

    top_probs, top_idx = torch.topk(probs, k=top_k, dim=1)

    result = []
    for p, i in zip(top_probs[0], top_idx[0]):
        result.append({
            "class": classes[int(i.item())],
            "confidence": float(p.item())
        })
    return result
