"""
ONNX object detector for TrapTracker RT.

Plain (unencrypted) ONNX model loading. Ported from the desktop/Flask
reference detector so detection behaviour matches the existing product.

Two pipelines:
  * single stage: one YOLO detector whose classes are the species (OnnxDetector)
  * two stage: a detector such as MegaDetector finds animal/person/vehicle
    boxes, then a DeepFaune-style classifier labels each animal crop to
    species (TwoStageDetector). Ported from the Trap Tracker Offline Processor
    so both apps label the same image the same way.
"""
from pathlib import Path
import ast
import json
import threading

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
DEFAULT_INPUT_SIZE = 640


def load_classes(path: str):
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    return [x.strip() for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def available_providers():
    return ort.get_available_providers()


# Populated when a detector is created, so the app/UI can report what
# actually happened with GPU vs CPU selection.
LAST_PROVIDER_REPORT = {
    'requested': [],
    'active': None,
    'gpu_attempted': False,
    'gpu_ok': False,
    'message': 'No model loaded yet.',
}


def _build_session(model_path, ordered_providers, log=None):
    """
    Create an InferenceSession, trying the requested providers in order.

    Unlike passing all providers at once, this attempts a GPU provider on
    its own first and verifies the session genuinely initialises on it; if
    that fails (no driver, wrong CUDA, no GPU, ARM build, etc.) it falls
    back to CPU and records a clear message rather than crashing.
    """
    def _log(m):
        if log:
            log(m)

    installed = ort.get_available_providers()
    gpu_providers = [p for p in ordered_providers
                     if p in ('CUDAExecutionProvider', 'TensorrtExecutionProvider')]
    report = {
        'requested': list(ordered_providers),
        'active': None,
        'gpu_attempted': False,
        'gpu_ok': False,
        'message': '',
    }

    # Try each requested GPU provider first, individually, and verify.
    for gp in gpu_providers:
        if gp not in installed:
            report['gpu_attempted'] = True
            _log(f'{gp} not available in this build (onnxruntime-gpu not '
                 f'installed or unsupported on this platform). Using CPU.')
            continue
        try:
            report['gpu_attempted'] = True
            sess = ort.InferenceSession(model_path, providers=[gp, 'CPUExecutionProvider'])
            active = sess.get_providers()
            if gp in active:
                report['gpu_ok'] = True
                report['active'] = gp
                report['message'] = f'GPU acceleration active ({gp}).'
                _log(report['message'])
                _store_report(report)
                return sess
            # Provider listed but ORT silently fell back to CPU.
            _log(f'{gp} could not be initialised (likely missing/incompatible '
                 f'driver or CUDA version). Falling back to CPU.')
        except Exception as ex:
            _log(f'{gp} failed to initialise ({ex}). Falling back to CPU.')

    # CPU fallback.
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    report['active'] = 'CPUExecutionProvider'
    if not gpu_providers:
        report['message'] = 'Running on CPU (no GPU provider requested).'
    elif not report['gpu_ok']:
        report['message'] = ('Running on CPU. A GPU provider was requested but '
                             'could not be used — check NVIDIA driver / CUDA, or '
                             'use the default CPU build.')
    _log(report['message'])
    _store_report(report)
    return sess


def _store_report(report):
    LAST_PROVIDER_REPORT.update(report)


def _metadata(session):
    try:
        return dict(session.get_modelmeta().custom_metadata_map or {})
    except Exception:
        return {}


def _parse_literal(value):
    """Parse a metadata value written as JSON or as a Python literal (Ultralytics)."""
    if not value:
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(value)
        except Exception:
            continue
    return None


def _meta_names(meta):
    """Class names Ultralytics embeds in an export (`names` metadata), or []."""
    names = _parse_literal(meta.get('names'))
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return []


def _nms(boxes, scores, iou_thresh=0.45):
    """Pure-numpy NMS. boxes are x1,y1,x2,y2 in model space."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou <= iou_thresh]
    return keep


def annotate(image_bgr, detections):
    out = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, [d['x1'], d['y1'], d['x2'], d['y2']])
        label = f"{d['species']} {d['confidence']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (20, 110, 60), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (20, 110, 60), -1)
        cv2.putText(out, label, (x1 + 4, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


class OnnxDetector:
    """
    Wraps an ONNX Runtime session for a YOLO-style model (YOLOv8/v10/11/26
    exports). Also used as stage 1 of the two-stage pipeline.

    Class names come from the class list file, falling back to the `names`
    Ultralytics embeds in the model. Input size: a static ONNX input shape
    wins, then `input_size`, then the model's `imgsz` metadata, then 640;
    pass input_size=None to use the size the model was trained at.
    """

    def __init__(self, model_path, classes_path, providers=None,
                 input_size=640, conf=0.25, iou=0.45, log=None):
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.iou = float(iou)

        ordered = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = _build_session(self.model_path, ordered, log=log)
        self.providers = self.session.get_providers()
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        meta = _metadata(self.session)

        # A class list for a different model would silently put the wrong
        # species names on every box, so check it against the names the model
        # was trained with (when it carries them) and refuse a count mismatch.
        self.classes = load_classes(classes_path)
        names = _meta_names(meta)
        if self.classes and names:
            if len(self.classes) != len(names):
                raise ValueError(
                    f'{Path(classes_path).name} lists {len(self.classes)} classes but '
                    f'{Path(self.model_path).name} was trained with {len(names)}. It looks like the '
                    f'class list of a different model; use the .txt that came with this model.')
            if self.classes != names and log:
                log(f'Note: {Path(classes_path).name} renames some of the classes embedded in '
                    f'{Path(self.model_path).name}; using the .txt names.')
        self.classes = self.classes or names

        size = int(input_size or 0)
        if size <= 0:
            imgsz = _parse_literal(meta.get('imgsz'))
            if isinstance(imgsz, (list, tuple)) and imgsz:
                size = int(max(imgsz))
            elif isinstance(imgsz, int):
                size = imgsz
        if size <= 0:
            size = DEFAULT_INPUT_SIZE
        shape = inp.shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                size = int(h)
        self.input_size = int(size)

        # ONNX Runtime sessions are not guaranteed thread-safe for concurrent
        # run() calls in all builds; serialise to be safe across project runners.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def preprocess(self, image_bgr):
        h, w = image_bgr.shape[:2]
        size = self.input_size
        scale = min(size / w, size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dw, dh = (size - nw) // 2, (size - nh) // 2
        canvas[dh:dh + nh, dw:dw + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        return np.ascontiguousarray(tensor), scale, dw, dh, w, h

    def _parse_outputs(self, outputs):
        out = np.asarray(outputs[0], dtype=np.float32)
        if out.ndim == 3:
            out = out[0]
        return out

    def _decode(self, raw):
        """Return (x1, y1, x2, y2, score, cls_id) tuples in letterboxed space.

        Handles the three common YOLO ONNX export layouts:
          1. end-to-end / NMS export: (N, 6) -> x1,y1,x2,y2,score,cls
          2. raw transposed:          (4+nc, num) -> transpose, argmax, NMS
          3. raw:                     (num, 4+nc) -> argmax, NMS
        """
        if raw.size == 0 or raw.ndim != 2:
            return []
        if raw.shape[1] == 6 and raw.shape[0] != 6:
            return [(float(x1), float(y1), float(x2), float(y2), float(s), int(round(c)))
                    for x1, y1, x2, y2, s, c in raw[:, :6] if s >= self.conf]

        # Raw grid: the predictions axis is the larger one.
        preds = raw.T if raw.shape[0] < raw.shape[1] else raw
        if preds.shape[1] < 5:
            return []
        class_scores = preds[:, 4:]
        cls_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), cls_ids]
        mask = scores >= self.conf
        if not np.any(mask):
            return []
        cx, cy, bw, bh = preds[mask, :4].T
        scores, cls_ids = scores[mask], cls_ids[mask]
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        keep = _nms(xyxy, scores, self.iou)
        return [(*map(float, xyxy[i]), float(scores[i]), int(cls_ids[i])) for i in keep]

    # ------------------------------------------------------------------
    def detect_image_array(self, image_bgr, filename='image'):
        tensor, scale, dw, dh, orig_w, orig_h = self.preprocess(image_bgr)

        with self._lock:
            outputs = self.session.run(None, {self.input_name: tensor})

        rows = []
        for x1, y1, x2, y2, score, cls_i in self._decode(self._parse_outputs(outputs)):
            x1 = (x1 - dw) / scale
            y1 = (y1 - dh) / scale
            x2 = (x2 - dw) / scale
            y2 = (y2 - dh) / scale
            x1 = max(0, min(orig_w, x1)); x2 = max(0, min(orig_w, x2))
            y1 = max(0, min(orig_h, y1)); y2 = max(0, min(orig_h, y2))
            label = self.classes[cls_i] if 0 <= cls_i < len(self.classes) else str(cls_i)
            rows.append({
                'filename': filename,
                'class_id': cls_i,
                'species': label,
                'confidence': round(float(score), 6),
                'x1': round(float(x1), 2),
                'y1': round(float(y1), 2),
                'x2': round(float(x2), 2),
                'y2': round(float(y2), 2),
            })
        return rows

    def detect_image_file(self, path, filename=None):
        img = cv2.imread(str(path))
        if img is None:
            return []
        return self.detect_image_array(img, filename or Path(path).name)

    def annotate(self, image_bgr, detections):
        return annotate(image_bgr, detections)

    def annotate_file(self, src_path, dst_path, detections):
        img = cv2.imread(str(src_path))
        if img is None:
            return False
        out = self.annotate(img, detections)
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(dst_path), out))


class CropClassifierONNX:
    """Stage 2 species classifier (DeepFaune-style) that labels detector crops.

    Input size, normalisation, crop padding and class names are read from the
    ONNX metadata when present, so a mismatched .txt file cannot silently
    shift the labels. Problems are collected in `warnings` and logged.
    """

    def __init__(self, model_path, classes_path='', providers=None, batch_size=16, log=None):
        self.model_path = str(model_path)
        self.batch_size = max(1, int(batch_size))
        self.warnings = []
        ordered = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = _build_session(self.model_path, ordered, log=log)
        self.providers = self.session.get_providers()
        meta = _metadata(self.session)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        out = self.session.get_outputs()[0]
        n_out = out.shape[-1] if isinstance(out.shape[-1], int) else None
        name = Path(self.model_path).name

        size = 0
        if isinstance(inp.shape, (list, tuple)) and len(inp.shape) == 4 and isinstance(inp.shape[2], int):
            size = inp.shape[2]
        if size <= 0:
            try:
                size = int(meta.get('input_size', 0))
            except ValueError:
                size = 0
        self.input_size = size if size > 0 else 182

        # Fraction of the box's longer side added as context on every side before
        # the square crop -- must match how the classifier's training crops were made.
        try:
            self.crop_padding = min(max(float(meta.get('crop_padding', 0) or 0), 0.0), 1.0)
        except ValueError:
            self.crop_padding = 0.0
            self.warnings.append(f'Ignoring malformed crop_padding metadata in {name}.')

        mean = _parse_literal(meta.get('mean')) or [0.485, 0.456, 0.406]
        std = _parse_literal(meta.get('std')) or [0.229, 0.224, 0.225]
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

        meta_classes = _parse_literal(meta.get('classes'))
        file_classes = load_classes(classes_path)
        if isinstance(meta_classes, list) and meta_classes:
            self.classes = [str(c) for c in meta_classes]
            if file_classes and file_classes != self.classes:
                self.warnings.append(
                    f'Class file {Path(classes_path).name} does not match the class list embedded in '
                    f'{name}; using the embedded list.')
        else:
            self.classes = file_classes
        if n_out and self.classes and len(self.classes) != n_out:
            raise ValueError(
                f'{name} outputs {n_out} classes but {len(self.classes)} class names were found. '
                'The class list belongs to a different model; use the .txt that came with this classifier.')

        # Optional class_groups metadata: {label: detector class it belongs under}.
        # TwoStageDetector uses it to keep each box's label within its detector class.
        self.class_groups = None
        groups = _parse_literal(meta.get('class_groups'))
        if groups is not None:
            if isinstance(groups, dict) and self.classes and (not n_out or len(self.classes) == n_out):
                self.class_groups = {str(k): str(v).lower() for k, v in groups.items()}
                ungrouped = [c for c in self.classes if c not in self.class_groups]
                if ungrouped:
                    self.warnings.append(
                        f'class_groups in {name} has no group for {", ".join(ungrouped)}; '
                        'those labels will never be predicted.')
            else:
                self.warnings.append(
                    f'Ignoring class_groups metadata in {name}: it is malformed or the class '
                    'list does not match the model outputs.')

        self._lock = threading.Lock()

    def _crop(self, image_bgr, box):
        """Square crop around the box (DeepFaune convention), clipped to the image.

        The box is first padded by crop_padding x its longer side on every side,
        then expanded to a square about its centre.
        """
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        pad = self.crop_padding * max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) / 2 + pad
        x1, y1 = max(0, int(np.floor(cx - half))), max(0, int(np.floor(cy - half)))
        x2, y2 = min(w, int(np.ceil(cx + half))), min(h, int(np.ceil(cy + half)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return image_bgr[y1:y2, x1:x2]

    def _preprocess(self, crop_bgr):
        # Antialiased bilinear resize (PIL), the same resampling torchvision's
        # Resize applies at training time; cv2's cubic doesn't antialias, which
        # adds aliasing when large crops are shrunk to the classifier's size.
        size = self.input_size
        rgb = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        rgb = np.asarray(rgb.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        return np.transpose(rgb, (2, 0, 1))

    def classify(self, image_bgr, boxes, masks=None):
        """Return one (class_id, label, prob) per box, or None where no crop was possible.

        masks, if given, holds one boolean array per box marking the labels that
        box may take. The prob returned is still from the softmax over all
        labels, so a crop that doesn't look like any allowed label keeps a low
        confidence instead of being inflated by the restriction.
        """
        results = [None] * len(boxes)
        tensors, idx = [], []
        for i, box in enumerate(boxes):
            crop = self._crop(image_bgr, box)
            if crop is None:
                continue
            tensors.append(self._preprocess(crop))
            idx.append(i)
        for start in range(0, len(tensors), self.batch_size):
            batch = np.ascontiguousarray(np.stack(tensors[start:start + self.batch_size]).astype(np.float32))
            with self._lock:
                logits = self.session.run(None, {self.input_name: batch})[0]
            logits = np.asarray(logits, dtype=np.float32)
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            for row, i in zip(probs, idx[start:start + self.batch_size]):
                if masks is not None and masks[i] is not None:
                    c = int(np.argmax(np.where(masks[i], row, -1.0)))
                else:
                    c = int(np.argmax(row))
                label = self.classes[c] if 0 <= c < len(self.classes) else str(c)
                results[i] = (c, label, float(row[c]))
        return results


class TwoStageDetector:
    """Stage 1 finds boxes, stage 2 classifies each crop to species.

    If the classifier carries class_groups metadata ({label: detector class}),
    each box is classified only among the labels grouped under its detector
    class, so an 'animal' box can never come back as e.g. 'Car'. Boxes whose
    detector class has no labels keep their detector label.

    Without class_groups: if the stage 1 detector has an 'animal' class
    (MegaDetector-style), only animal boxes are sent to the classifier and
    person/vehicle/etc. keep their detector label; if it has no 'animal' class
    (e.g. a species detector), every box is reclassified over all labels.

    Rows carry 'detector_class' and 'detector_confidence' alongside the final
    'species' and 'confidence'.
    """

    def __init__(self, detector_path, classifier_path, classifier_classes_path='',
                 providers=None, input_size=None, conf=0.25, iou=0.45, log=None):
        self.detector = OnnxDetector(detector_path, '', providers=providers,
                                     input_size=input_size, conf=conf, iou=iou, log=log)
        self.classifier = CropClassifierONNX(classifier_path, classifier_classes_path,
                                             providers=providers, log=log)
        det_classes = [c.lower() for c in self.detector.classes]
        self.input_size = self.detector.input_size
        self.providers = self.detector.providers
        self.warnings = list(self.classifier.warnings)

        # Per detector class id: boolean mask of the labels its boxes may take.
        self.group_masks = None
        groups = self.classifier.class_groups
        if groups:
            label_groups = [groups.get(c) for c in self.classifier.classes]
            masks = {i: np.array([g == name for g in label_groups])
                     for i, name in enumerate(det_classes)}
            self.group_masks = {i: m for i, m in masks.items() if m.any()}
            if not self.group_masks:
                self.warnings.append(
                    'Classifier class_groups match none of the detector classes '
                    f'({", ".join(self.detector.classes)}); detector labels are kept unchanged.')
            self.classify_ids = set(self.group_masks)
        else:
            self.classify_ids = {i for i, c in enumerate(det_classes) if c == 'animal'} or None

        # Every label this pipeline can output: the classifier's labels, plus
        # the detector labels of boxes that are never reclassified (e.g. person).
        kept = [] if self.classify_ids is None else [
            c for i, c in enumerate(self.detector.classes) if i not in self.classify_ids]
        self.classes = list(self.classifier.classes) + kept

        if log:
            for w in self.warnings:
                log(f'Warning: {w}')

    def detect_image_array(self, image_bgr, filename='image'):
        rows = self.detector.detect_image_array(image_bgr, filename=filename)
        for d in rows:
            d['detector_class'] = d['species']
            d['detector_confidence'] = d['confidence']
        targets = [d for d in rows if self.classify_ids is None or d['class_id'] in self.classify_ids]
        if targets:
            boxes = [(d['x1'], d['y1'], d['x2'], d['y2']) for d in targets]
            masks = ([self.group_masks[d['class_id']] for d in targets]
                     if self.group_masks is not None else None)
            for d, res in zip(targets, self.classifier.classify(image_bgr, boxes, masks)):
                if res is None:
                    continue
                c, label, prob = res
                d['class_id'] = c
                d['species'] = label
                d['confidence'] = round(prob, 6)
        return rows

    def detect_image_file(self, path, filename=None):
        img = cv2.imread(str(path))
        if img is None:
            return []
        return self.detect_image_array(img, filename or Path(path).name)

    def annotate(self, image_bgr, detections):
        return annotate(image_bgr, detections)

    def annotate_file(self, src_path, dst_path, detections):
        return self.detector.annotate_file(src_path, dst_path, detections)


# Cache of loaded detectors keyed by model file path(s), so each project's
# chosen model is loaded once and reused (multiple models can coexist).
_detectors = {}
_detectors_lock = threading.Lock()


def get_detector(model_path, classes_path, providers=None, input_size=640, conf=0.25, log=None,
                 classifier_path=None, classifier_classes_path=None):
    """Load (or reuse) a detector. With classifier_path it is a two-stage pipeline,
    whose stage 1 runs at the detector's own training size."""
    key = (str(model_path), str(classifier_path or ''))
    with _detectors_lock:
        det = _detectors.get(key)
        if det is None:
            if classifier_path:
                det = TwoStageDetector(
                    model_path, classifier_path, classifier_classes_path or '',
                    providers=providers, input_size=None, conf=conf, log=log,
                )
            else:
                det = OnnxDetector(
                    model_path, classes_path,
                    providers=providers, input_size=input_size, conf=conf, log=log,
                )
            _detectors[key] = det
        return det


def reset_detector(model_path=None):
    with _detectors_lock:
        if model_path is None:
            _detectors.clear()
        else:
            for key in [k for k in _detectors if str(model_path) in k]:
                _detectors.pop(key, None)
