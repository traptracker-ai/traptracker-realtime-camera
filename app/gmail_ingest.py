"""
Gmail / IMAP ingest service.

Ported from the desktop GmailIngestService.cs. Connects to the project's
mailbox, finds unseen messages whose subject contains a camera token in
the form  (= CameraName =), matches that token to a configured + enabled
camera (by Name then RoutingKey), extracts image attachments into
Inbound/<CameraKey>/, marks the message seen and moves it to Trash.

Only messages whose token matches a configured camera are downloaded.
"""
import email
import imaplib
import os
import re
from datetime import datetime

from .config import IMAGE_EXTS
from . import project_folders as pf

# Capture text between "=" and "=" inside the subject, e.g. (=PaulsCamera01=)
_CAMERA_TOKEN_RE = re.compile(r'=\s*(?P<name>[^=]+?)\s*=')


def _extract_camera_name(subject):
    if not subject:
        return None
    m = _CAMERA_TOKEN_RE.search(subject)
    if not m:
        return None
    name = (m.group('name') or '').strip()
    return name or None


def _find_matching_camera_key(cameras, token):
    token_l = token.strip().lower()
    for c in cameras:
        if c.name and c.name.strip().lower() == token_l:
            return (c.routing_key or c.name).strip(), c
    for c in cameras:
        if c.routing_key and c.routing_key.strip().lower() == token_l:
            return c.routing_key.strip(), c
    return None, None


def _decode_subject(msg):
    raw = msg.get('Subject', '')
    try:
        parts = email.header.decode_header(raw)
        out = []
        for text, enc in parts:
            if isinstance(text, bytes):
                out.append(text.decode(enc or 'utf-8', errors='replace'))
            else:
                out.append(text)
        return ''.join(out)
    except Exception:
        return raw or ''


def _iter_image_attachments(msg):
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        filename = part.get_filename()
        ctype = (part.get_content_type() or '').lower()
        is_image = ctype.startswith('image/')
        ext = os.path.splitext(filename)[1].lower() if filename else ''
        if not is_image and ext not in IMAGE_EXTS:
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        yield filename or 'image', ext, payload


def pull_new_images(project, cameras, log=None, delete_emails=True):
    """
    Returns the number of image attachments downloaded.
    `cameras` is the list of enabled Camera rows for the project.
    `project.email_settings` must be loaded.
    """
    def _log(m):
        if log:
            log(m)

    settings = project.email_settings
    if settings is None:
        raise RuntimeError('No email settings found for this project.')
    if not settings.email_address or not settings.app_password:
        raise RuntimeError('Email address / app password not configured.')

    inbound_root = pf.inbound_folder(project)
    os.makedirs(inbound_root, exist_ok=True)

    host = settings.imap_host or 'imap.gmail.com'
    port = settings.imap_port or 993

    if settings.use_ssl:
        client = imaplib.IMAP4_SSL(host, port)
    else:
        client = imaplib.IMAP4(host, port)
        try:
            client.starttls()
        except Exception:
            pass

    downloaded = 0
    try:
        try:
            client.login(settings.email_address, settings.app_password)
        except imaplib.IMAP4.error as ex:
            raise RuntimeError(
                'IMAP login failed — check the email address and app password. '
                'For Gmail, you must use a 16-character App Password (not your '
                f'account password) and have IMAP enabled. ({ex})')
        client.select('INBOX')

        typ, data = client.search(None, 'UNSEEN')
        if typ != 'OK':
            return 0
        uids = data[0].split()
        if not uids:
            return 0

        for uid in uids:
            typ, msg_data = client.fetch(uid, '(RFC822)')
            if typ != 'OK' or not msg_data or not msg_data[0]:
                continue
            msg = email.message_from_bytes(msg_data[0][1])
            subject = _decode_subject(msg)

            token = _extract_camera_name(subject)
            if not token:
                _log(f'Ignored email (no camera token): "{subject[:120]}"')
                _seen_and_remove(client, uid, delete_emails)
                continue

            camera_key, camera = _find_matching_camera_key(cameras, token)
            if camera_key is None:
                _log(f"Ignored email (unknown camera '{token}'): \"{subject[:120]}\"")
                _seen_and_remove(client, uid, delete_emails)
                continue

            attachments = list(_iter_image_attachments(msg))
            if not attachments:
                _seen_and_remove(client, uid, delete_emails)
                continue

            camera_folder = pf.sanitize_folder_name(camera_key)
            dest_dir = os.path.join(inbound_root, camera_folder)
            os.makedirs(dest_dir, exist_ok=True)

            for original_name, ext, payload in attachments:
                if not ext or ext not in IMAGE_EXTS:
                    ext = '.jpg'
                base = os.path.splitext(os.path.basename(original_name))[0] or 'image'
                base = pf.sanitize_folder_name(base)
                stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')[:-3]
                out_path = os.path.join(dest_dir, f'{stamp}_{base}{ext}')
                with open(out_path, 'wb') as fh:
                    fh.write(payload)
                downloaded += 1
                _log(f'Downloaded: {camera_folder}/{os.path.basename(out_path)}')

            _seen_and_remove(client, uid, delete_emails)

        if delete_emails:
            try:
                client.expunge()
            except Exception:
                pass
    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            client.logout()
        except Exception:
            pass

    return downloaded


def _seen_and_remove(client, uid, delete_emails):
    try:
        client.store(uid, '+FLAGS', '\\Seen')
    except Exception:
        pass
    if not delete_emails:
        return
    # Prefer Gmail trash; fall back to Deleted flag.
    moved = False
    try:
        client.copy(uid, '[Gmail]/Trash')
        moved = True
    except Exception:
        moved = False
    try:
        client.store(uid, '+FLAGS', '\\Deleted')
    except Exception:
        pass
