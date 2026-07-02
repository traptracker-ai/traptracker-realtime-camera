"""
ONNX object detector for TrapTracker RT.

Plain (unencrypted) ONNX model loading. Ported from the desktop/Flask
reference detector so detection behaviour matches the existing product.
"""
from pathlib import Path
import threading

import cv2
import numpy as np
import onnxruntime as ort

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def load_classes(path: str):
    p = Path(path)
    if not p.exists():
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


class OnnxDetector:
    """
    Wraps an ONNX Runtime session for a YOLO-style end-to-end model whose
    output rows are [x1, y1, x2, y2, score, class_id] in letterboxed
    input-space coordinates.
    """

    def __init__(self, model_path, classes_path, providers=None,
                 input_size=640, conf=0.25, log=None):
        self.model_path = str(model_path)
        self.classes = load_classes(classes_path)
        self.input_size = int(input_size)
        self.conf = float(conf)

        ordered = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = _build_session(self.model_path, ordered, log=log)
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name

        # ONNX Runtime sessions are not guaranteed thread-safe for concurrent
        # run() calls in all builds; serialise to be safe across project runners.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def preprocess(self, image_bgr):
        h, w = image_bgr.shape[:2]
        size = self.input_size
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dw, dh = (size - nw) // 2, (size - nh) // 2
        canvas[dh:dh + nh, dw:dw + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        return tensor, scale, dw, dh, w, h

    def _parse_outputs(self, outputs):
        out = outputs[0]
        if isinstance(out, list):
            out = np.array(out)
        out = np.asarray(out)
        if out.ndim == 3:
            out = out[0]
        return out

    # ------------------------------------------------------------------
    def detect_image_array(self, image_bgr, filename='image'):
        tensor, scale, dw, dh, orig_w, orig_h = self.preprocess(image_bgr)

        with self._lock:
            outputs = self.session.run(None, {self.input_name: tensor})

        detections = self._parse_outputs(outputs)
        rows = []
        for det in detections:
            det = np.asarray(det).astype(float).tolist()
            if len(det) < 6:
                continue
            x1, y1, x2, y2, score, cls_id = det[:6]
            if score < self.conf:
                continue
            x1 = (x1 - dw) / scale
            y1 = (y1 - dh) / scale
            x2 = (x2 - dw) / scale
            y2 = (y2 - dh) / scale
            x1 = max(0, min(orig_w, x1)); x2 = max(0, min(orig_w, x2))
            y1 = max(0, min(orig_h, y1)); y2 = max(0, min(orig_h, y2))
            cls_i = int(round(cls_id))
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
        out = image_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, [d['x1'], d['y1'], d['x2'], d['y2']])
            label = f"{d['species']} {d['confidence']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (20, 110, 60), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (20, 110, 60), -1)
            cv2.putText(out, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return out

    def annotate_file(self, src_path, dst_path, detections):
        img = cv2.imread(str(src_path))
        if img is None:
            return False
        out = self.annotate(img, detections)
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(dst_path), out))


# Module-level singleton (one model shared by all project runners).
# Cache of loaded detectors keyed by model file path, so each project's
# chosen model is loaded once and reused (multiple models can coexist).
_detectors = {}
_detectors_lock = threading.Lock()


def get_detector(model_path, classes_path, providers=None, input_size=640, conf=0.25, log=None):
    key = str(model_path)
    with _detectors_lock:
        det = _detectors.get(key)
        if det is None:
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
            _detectors.pop(str(model_path), None)
