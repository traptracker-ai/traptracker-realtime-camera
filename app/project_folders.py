"""Filesystem layout helpers for projects (mirrors ProjectFolders.cs)."""
import os
import re

from .config import Config


def _safe_name(name, fallback='Project'):
    name = (name or '').strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = name.strip().rstrip('.')
    return name or fallback


def project_root(project):
    """Return (and not create) the per-project root folder."""
    if project.root_folder:
        return project.root_folder
    return os.path.join(Config.PROJECTS_DIR, f'{project.id}_{_safe_name(project.name)}')


def inbound_folder(project):
    return os.path.join(project_root(project), Config.INBOUND_FOLDER_NAME)


def processed_root(project):
    return os.path.join(project_root(project), Config.PROCESSED_FOLDER_NAME)


def originals_folder(project):
    return os.path.join(processed_root(project), Config.ORIGINALS_FOLDER_NAME)


def classified_folder(project):
    return os.path.join(processed_root(project), Config.CLASSIFIED_FOLDER_NAME)


def ensure_project_folders(project):
    for path in (inbound_folder(project), originals_folder(project), classified_folder(project)):
        os.makedirs(path, exist_ok=True)


def sanitize_folder_name(name):
    return _safe_name(name, 'Unknown')
