"""
Model discovery.

Scans MODEL_DIR for ONNX models. Each model is `<name>.onnx`; its class
list is the file with the same base name and a `.txt` (or `.json`)
extension, e.g.:

    traptracker_uk_mammals_yolo26x_v1.0.0.onnx
    traptracker_uk_mammals_yolo26x_v1.0.0.txt

The base name (without extension) is the model's id/key, which is what
each project stores in `model_folder`.
"""
import json
import os

from .config import Config


def _classes_path_for(onnx_path):
    base, _ = os.path.splitext(onnx_path)
    for ext in ('.txt', '.json', '.names'):
        cand = base + ext
        if os.path.exists(cand):
            return cand
    # Fall back to a generic classes.txt in the same dir, if present.
    generic = os.path.join(os.path.dirname(onnx_path), 'classes.txt')
    return generic if os.path.exists(generic) else None


def list_models():
    """Return a sorted list of dicts: {id, onnx_path, classes_path, num_classes}."""
    out = []
    model_dir = Config.MODEL_DIR
    if not os.path.isdir(model_dir):
        return out
    for fn in sorted(os.listdir(model_dir)):
        if not fn.lower().endswith('.onnx'):
            continue
        onnx_path = os.path.join(model_dir, fn)
        model_id = os.path.splitext(fn)[0]
        classes_path = _classes_path_for(onnx_path)
        out.append({
            'id': model_id,
            'onnx_path': onnx_path,
            'classes_path': classes_path,
            'num_classes': len(_read_classes(classes_path)) if classes_path else 0,
        })
    return out


def get_model(model_id):
    """Resolve a stored model id to its paths, with sensible fallbacks."""
    models = list_models()
    if not models:
        return None
    if model_id:
        for m in models:
            if m['id'] == model_id:
                return m
        # Tolerate legacy/loose values: match by substring.
        low = model_id.lower()
        for m in models:
            if low in m['id'].lower():
                return m
    # No match: fall back to the first available model.
    return models[0]


def _read_classes(path):
    if not path or not os.path.exists(path):
        return []
    try:
        if path.lower().endswith('.json'):
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
            return []
        with open(path, 'r', encoding='utf-8') as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    except Exception:
        return []


def classes_for_model(model_id):
    m = get_model(model_id)
    if not m:
        return []
    return _read_classes(m['classes_path'])
