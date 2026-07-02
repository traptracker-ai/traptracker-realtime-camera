"""
Central application configuration for TrapTracker RT (Docker).

Constants and environment-driven defaults only. Per-project email
settings are stored in the database and edited in the UI; the values
here are only used as form defaults / global model settings.
"""
import os


def _bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ('1', 'true', 'yes', 'on')


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Config:
    # ---- Paths (mounted volumes) ----
    DATA_DIR = os.environ.get('DATA_DIR', '/app/data')
    PROJECTS_DIR = os.environ.get('PROJECTS_DIR', '/app/projects')
    MODEL_DIR = os.environ.get('MODEL_DIR', '/app/models')
    DB_PATH = os.environ.get('DB_PATH', os.path.join(DATA_DIR, 'traptracker_rt.db'))

    INBOUND_FOLDER_NAME = 'Inbound'
    PROCESSED_FOLDER_NAME = 'Processed'
    ORIGINALS_FOLDER_NAME = 'Original Images'
    CLASSIFIED_FOLDER_NAME = 'Classified Images'

    # ---- Detection / model ----
    # Models are auto-discovered from MODEL_DIR: each <name>.onnx is paired
    # with a same-named <name>.txt class list, and projects pick one.
    # MODEL_PATH/CLASSES_PATH below are only legacy fallbacks.
    MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(MODEL_DIR, 'best.onnx'))
    CLASSES_PATH = os.environ.get('CLASSES_PATH', os.path.join(MODEL_DIR, 'classes.txt'))
    INPUT_SIZE = _int('INPUT_SIZE', 640)
    CONF_THRESHOLD = _float('CONF_THRESHOLD', 0.10)
    ORT_PROVIDERS = [
        p.strip() for p in
        os.environ.get('ORT_PROVIDERS', 'CUDAExecutionProvider,CPUExecutionProvider').split(',')
        if p.strip()
    ]

    # ---- Email defaults (form pre-fill only; real values per project in UI) ----
    DEFAULT_PROVIDER = 'Gmail'
    DEFAULT_IMAP_HOST = os.environ.get('DEFAULT_IMAP_HOST', 'imap.gmail.com')
    DEFAULT_IMAP_PORT = _int('DEFAULT_IMAP_PORT', 993)
    DEFAULT_IMAP_SSL = _bool('DEFAULT_IMAP_SSL', True)
    DEFAULT_SMTP_HOST = os.environ.get('DEFAULT_SMTP_HOST', 'smtp.gmail.com')
    DEFAULT_SMTP_PORT = _int('DEFAULT_SMTP_PORT', 587)
    DEFAULT_SMTP_STARTTLS = _bool('DEFAULT_SMTP_STARTTLS', True)

    # ---- Alert defaults ----
    DEFAULT_MIN_CONFIDENCE = _float('DEFAULT_MIN_CONFIDENCE', 0.50)
    DEFAULT_COOLDOWN_SECONDS = _int('DEFAULT_COOLDOWN_SECONDS', 300)

    # ---- Runner timings ----
    POLL_EMAIL_SECONDS = _int('POLL_EMAIL_SECONDS', 30)
    POLL_INBOUND_SECONDS = _int('POLL_INBOUND_SECONDS', 10)
    LOOP_DELAY_MS = _int('LOOP_DELAY_MS', 500)

    # ---- Web ----
    # Binds to localhost by default so the credential-bearing UI is not
    # exposed on the network. Override HOST=0.0.0.0 only on a trusted LAN.
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = _int('PORT', 8501)
    # If SECRET_KEY is unset, a persistent random key is generated at runtime
    # (see app.py:_resolve_secret_key) rather than shipping a known constant.
    SECRET_KEY = os.environ.get('SECRET_KEY')


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
