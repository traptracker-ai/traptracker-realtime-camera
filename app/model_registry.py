"""
Model discovery.

Scans MODEL_DIR for ONNX models. A single-stage model is `<name>.onnx` in
MODEL_DIR/single-stage-detectors/ (or MODEL_DIR itself); its class list is
the file with the same base name and a `.txt` (or `.json`) extension, e.g.:

    single-stage-detectors/cai_uk_mammals_yolo26x_v1.0.0.onnx
    single-stage-detectors/cai_uk_mammals_yolo26x_v1.0.0.txt

A two-stage model (e.g. MegaDetector + a DeepFaune species classifier) is a
pair in MODEL_DIR/two-stage-detectors/ (or MODEL_DIR itself):

    deepfaune_ssa_v1.2_stage1_detector.onnx
    deepfaune_ssa_v1.2_stage2_classifier.onnx
    deepfaune_ssa_v1.2_stage2_classifier.txt     (optional; the ONNX's own
                                                  class list wins)

The base name (without extension, or without the _stage1/_stage2 suffix
for a pair) is the model's id/key, which is what each project stores in
`model_folder`.
"""
import functools
import json
import os

from .config import Config

SINGLE_STAGE_DIR = 'single-stage-detectors'
TWO_STAGE_DIR = 'two-stage-detectors'
STAGE1_SUFFIX = '_stage1_detector.onnx'
STAGE2_SUFFIX = '_stage2_classifier.onnx'


def _classes_path_for(onnx_path):
    # Only the model's own class list: a shared classes.txt could belong to a
    # different model and would put the wrong species names on every box.
    base, _ = os.path.splitext(onnx_path)
    for ext in ('.txt', '.json', '.names'):
        cand = base + ext
        if os.path.exists(cand):
            return cand
    return None


def _list_two_stage(model_dir):
    out = []
    for d in (os.path.join(model_dir, TWO_STAGE_DIR), model_dir):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(STAGE1_SUFFIX):
                continue
            name = fn[:-len(STAGE1_SUFFIX)]
            classifier = os.path.join(d, name + STAGE2_SUFFIX)
            if not os.path.exists(classifier):
                continue
            classes_path = os.path.splitext(classifier)[0] + '.txt'
            classes_path = classes_path if os.path.exists(classes_path) else None
            out.append({
                'id': name,
                'pipeline': 'two_stage',
                'onnx_path': os.path.join(d, fn),
                'classifier_path': classifier,
                'classes_path': classes_path,
                'num_classes': len(_read_classes(classes_path)) if classes_path else 0,
            })
    return out


def list_models():
    """Return a sorted list of dicts: {id, pipeline, onnx_path, classes_path, num_classes},
    plus classifier_path for two-stage models."""
    out = []
    model_dir = Config.MODEL_DIR
    if not os.path.isdir(model_dir):
        return out
    seen = set()
    # Same layout as the Offline Processor; top-level models are still accepted.
    for d in (os.path.join(model_dir, SINGLE_STAGE_DIR), model_dir):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            low = fn.lower()
            if not low.endswith('.onnx') or low.endswith((STAGE1_SUFFIX, STAGE2_SUFFIX)):
                continue
            model_id = os.path.splitext(fn)[0]
            if model_id in seen:
                continue
            seen.add(model_id)
            onnx_path = os.path.join(d, fn)
            classes_path = _classes_path_for(onnx_path)
            out.append({
                'id': model_id,
                'pipeline': 'single_stage',
                'onnx_path': onnx_path,
                'classes_path': classes_path,
                'num_classes': len(_read_classes(classes_path)) if classes_path else 0,
            })
    for m in _list_two_stage(model_dir):
        if m['id'] not in seen:
            seen.add(m['id'])
            out.append(m)
    return sorted(out, key=lambda m: m['id'])


def get_model(model_id):
    """Resolve a stored model id to its paths, or None if that model isn't in MODEL_DIR.

    Exact matches only: substituting a similarly named or the first model
    would quietly classify a project's images with a model it didn't choose.
    """
    if not model_id:
        return None
    for m in list_models():
        if m['id'] == model_id:
            return m
    return None


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


@functools.lru_cache(maxsize=16)
def _detector_names(onnx_path, _mtime):
    """Class names embedded in a stage 1 detector (Ultralytics `names` metadata)."""
    try:
        import onnxruntime as ort
        from .detector import _meta_names
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        return tuple(_meta_names(sess.get_modelmeta().custom_metadata_map))
    except Exception:
        return ()


def classes_for_model(model_id):
    """Labels a model can output (used for the alert watch-list).

    For a two-stage model: the classifier's species, plus the detector
    classes that are not reclassified (e.g. MegaDetector's person/vehicle).
    """
    m = get_model(model_id)
    if not m:
        return []
    classes = _read_classes(m['classes_path'])
    if m.get('pipeline') == 'two_stage':
        try:
            names = _detector_names(m['onnx_path'], os.path.getmtime(m['onnx_path']))
        except OSError:
            names = ()
        if any(n.lower() == 'animal' for n in names):
            classes += [n for n in names if n.lower() != 'animal']
    return classes
