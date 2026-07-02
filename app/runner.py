"""
Project runner.

Ported from ProjectRunner.cs. Each running project gets a background thread
that, on independent timers:
  1. polls the project mailbox and downloads camera images into Inbound/
  2. scans Inbound/, runs ONNX detection on new images, writes originals +
     boxed outputs into Processed/, persists images + detections, and fires
     alerts for matching species.

A RunnerManager tracks which projects are running and exposes recent log
lines for the UI.
"""
import collections
import os
import shutil
import threading
import time
from datetime import datetime

import cv2

from .config import Config, IMAGE_EXTS
from .models import Session, Project, Camera, ImageItem, Detection
from . import project_folders as pf
from . import gmail_ingest
from . import alerting
from . import model_registry
from .detector import get_detector


class _ProjectWorker:
    def __init__(self, project_id, manager):
        self.project_id = project_id
        self.manager = manager
        self._stop = threading.Event()
        self._thread = None
        self.logs = collections.deque(maxlen=300)
        self.status = 'starting'
        self.started_at = datetime.utcnow()

    def log(self, message):
        line = f'[{datetime.utcnow():%H:%M:%S}] {message}'
        self.logs.append(line)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _run(self):
        try:
            self._load_model_and_loop()
        except Exception as ex:
            self.log(f'Runner stopped with error: {ex}')
            self.status = 'error'
        else:
            self.status = 'stopped'

    def _load_model_and_loop(self):
        s = Session()
        try:
            project = s.query(Project).get(self.project_id)
            if project is None:
                raise RuntimeError(f'Project {self.project_id} not found.')
            pf.ensure_project_folders(project)
            model_id = project.model_folder
        finally:
            s.close()

        model = model_registry.get_model(model_id)
        if model is None:
            raise RuntimeError(
                f'No ONNX models found in {Config.MODEL_DIR}. '
                f'Add a .onnx file (with a matching .txt class list).')
        if not model['classes_path']:
            raise RuntimeError(
                f"Model '{model['id']}' has no class list. Add "
                f"{model['id']}.txt next to the .onnx file.")

        self.log(f"Loading model: {model['id']} ...")
        detector = get_detector(
            model['onnx_path'], model['classes_path'],
            providers=Config.ORT_PROVIDERS,
            input_size=Config.INPUT_SIZE, conf=Config.CONF_THRESHOLD,
            log=self.log,
        )
        self.log(f"Model loaded ({model['id']}). Provider: {', '.join(detector.providers)}. "
                 f'Labels: {len(detector.classes)}. Starting runner...')
        self.status = 'running'

        next_email = 0.0
        next_inbound = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                if now >= next_email:
                    self._poll_email_once()
                    next_email = time.monotonic() + self._email_backoff_seconds()
                if now >= next_inbound:
                    self._process_inbound_once(detector)
                    next_inbound = time.monotonic() + Config.POLL_INBOUND_SECONDS
            except Exception as ex:
                self.log(f'Runner loop error: {ex}')
            self._stop.wait(Config.LOOP_DELAY_MS / 1000.0)

    # ------------------------------------------------------------------
    def _poll_email_once(self):
        s = Session()
        try:
            project = s.query(Project).get(self.project_id)
            if project is None:
                return
            _ = project.email_settings  # load
            cameras = s.query(Camera).filter_by(project_id=self.project_id, is_enabled=True).all()
            downloaded = gmail_ingest.pull_new_images(project, cameras, log=self.log, delete_emails=True)
            if downloaded > 0:
                self.log(f'Email poll: downloaded {downloaded} new image(s).')
            self._email_fail_count = 0
        except Exception as ex:
            self._email_fail_count = getattr(self, '_email_fail_count', 0) + 1
            self.log(f'Email poll error (attempt {self._email_fail_count}): {ex}')
            if self._email_fail_count == 3:
                self.log('Email polling has failed repeatedly. Backing off — '
                         'fix the email settings, then stop and start monitoring.')
        finally:
            s.close()

    def _email_backoff_seconds(self):
        """Lengthen the poll interval after consecutive failures (cap 5 min)."""
        fails = getattr(self, '_email_fail_count', 0)
        if fails <= 0:
            return Config.POLL_EMAIL_SECONDS
        return min(Config.POLL_EMAIL_SECONDS * (2 ** min(fails, 4)), 300)

    def _process_inbound_once(self, detector):
        s = Session()
        try:
            project = s.query(Project).get(self.project_id)
            if project is None:
                return
            inbound = pf.inbound_folder(project)
            originals_dir = pf.originals_folder(project)
            classified_dir = pf.classified_folder(project)
            for d in (inbound, originals_dir, classified_dir):
                os.makedirs(d, exist_ok=True)
        finally:
            s.close()

        # Enumerate ready inbound images.
        files = []
        for root, _dirs, names in os.walk(inbound):
            for n in names:
                if os.path.splitext(n)[1].lower() in IMAGE_EXTS:
                    p = os.path.join(root, n)
                    if _file_ready(p):
                        files.append(p)
        if not files:
            return

        self.log(f'Inbound poll: found {len(files)} image(s).')

        for file in files:
            if self._stop.is_set():
                return
            try:
                self._process_one(file, inbound, originals_dir, classified_dir, detector)
            except Exception as ex:
                self.log(f"Processing failed for '{os.path.basename(file)}': {ex}")

    def _process_one(self, file, inbound, originals_dir, classified_dir, detector):
        s = Session()
        try:
            # Dedupe by inbound path.
            if s.query(ImageItem).filter_by(full_path=file).first():
                return

            # Camera subfolder relative to inbound.
            camera_subdir = ''
            rel = os.path.relpath(os.path.dirname(file), inbound)
            if rel not in ('.', '..', ''):
                camera_subdir = rel

            self.log(f'Detecting: {os.path.basename(file)}' +
                     (f' (camera: {camera_subdir})' if camera_subdir else ''))

            img = cv2.imread(file)
            if img is None:
                self.log(f'Skipped (unreadable image): {os.path.basename(file)}')
                return
            dets = detector.detect_image_array(img, os.path.basename(file))

            dets_classified = [d for d in dets if d['confidence'] >= Config.CONF_THRESHOLD]
            if dets:
                best = max(dets, key=lambda d: d['confidence'])
                self.log(f"Best: {best['species']} {best['confidence']:.2f}")
            else:
                self.log('Best: (no detections)')

            project = s.query(Project).get(self.project_id)

            # Persist image + detections.
            item = ImageItem(
                project_id=self.project_id,
                camera_name=camera_subdir,
                file_name=os.path.basename(file),
                full_path=file,
                received_at_utc=datetime.utcnow(),
            )
            s.add(item)
            s.flush()
            for d in dets:
                s.add(Detection(
                    image_item_id=item.id, label=d['species'], confidence=d['confidence'],
                    x1=d['x1'], y1=d['y1'], x2=d['x2'], y2=d['y2'],
                ))
            item.processed_at_utc = datetime.utcnow()
            s.commit()
            image_id = item.id

            # Output paths (preserve camera subfolder).
            stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')[:-3]
            base = os.path.splitext(os.path.basename(file))[0]
            ext = os.path.splitext(file)[1]
            orig_out_dir = originals_dir if not camera_subdir else os.path.join(originals_dir, camera_subdir)
            class_out_dir = classified_dir if not camera_subdir else os.path.join(classified_dir, camera_subdir)
            os.makedirs(orig_out_dir, exist_ok=True)
            os.makedirs(class_out_dir, exist_ok=True)
            original_out = _unique_path(os.path.join(orig_out_dir, f'{stamp}_{base}{ext}'))
            boxed_out = os.path.join(class_out_dir, f'{stamp}_{base}_boxed.jpg')

            # Move original out of inbound.
            try:
                shutil.move(file, original_out)
            except Exception as ex:
                self.log(f'Move failed: {ex}')
                return

            boxed_or_none = None
            if dets_classified:
                try:
                    annotated = detector.annotate(cv2.imread(original_out), dets_classified)
                    cv2.imwrite(boxed_out, annotated)
                    boxed_or_none = boxed_out
                    self.log(f'Classified image created: {os.path.basename(boxed_out)}')
                except Exception as ex:
                    self.log(f'Annotator failed: {ex}')
            else:
                self.log(f'No detections >= {Config.CONF_THRESHOLD:.2f}; skipping classified output.')

            # Update stored path + alert.
            item = s.query(ImageItem).get(image_id)
            if item:
                item.full_path = original_out
                item.boxed_path = boxed_or_none or ''
                s.commit()

            self.log(f'Alert candidates (>= {Config.CONF_THRESHOLD:.2f}): {len(dets_classified)}')
            alerting.evaluate_and_send(
                project_id=self.project_id, image_item_id=image_id,
                dets=dets_classified, original_path=original_out,
                boxed_path=boxed_or_none, log=self.log,
            )
            self.log(f'Processed -> {os.path.basename(original_out)} '
                     f'(dets: {len(dets)}, classified: {len(dets_classified)})')
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()


def _file_ready(path):
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _unique_path(path):
    if not os.path.exists(path):
        return path
    d = os.path.dirname(path)
    name, ext = os.path.splitext(os.path.basename(path))
    i = 1
    while True:
        cand = os.path.join(d, f'{name}_{i}{ext}')
        if not os.path.exists(cand):
            return cand
        i += 1


class RunnerManager:
    def __init__(self):
        self._workers = {}
        self._lock = threading.Lock()

    def start(self, project_id):
        with self._lock:
            w = self._workers.get(project_id)
            if w and w.is_alive():
                return False
            w = _ProjectWorker(project_id, self)
            self._workers[project_id] = w
            w.start()
            return True

    def stop(self, project_id):
        with self._lock:
            w = self._workers.get(project_id)
            if not w:
                return False
            w.stop()
            return True

    def is_running(self, project_id):
        w = self._workers.get(project_id)
        return bool(w and w.is_alive())

    def status(self, project_id):
        w = self._workers.get(project_id)
        if not w:
            return 'stopped'
        return w.status if w.is_alive() or w.status == 'error' else 'stopped'

    def logs(self, project_id):
        w = self._workers.get(project_id)
        return list(w.logs) if w else []

    def running_ids(self):
        return [pid for pid, w in self._workers.items() if w.is_alive()]


manager = RunnerManager()
