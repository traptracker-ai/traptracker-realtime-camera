"""
Alerting service.

Ported from AlertingService.cs. For each enabled alert rule whose label
matches a detection in the image at or above the rule's min-confidence,
sends one email (with the boxed + original image attached) to all project
recipients. Per (rule, image) dedupe prevents double-sends; each new image
that matches still generates its own alert.
"""
import os
import mimetypes
import smtplib
from datetime import datetime
from email.message import EmailMessage

from .models import (
    Session, Project, AlertRule, AlertRecipient, AlertSent, ProjectEmailSettings,
)


def _normalize_recipients(rows):
    out = []
    seen = set()
    for r in rows:
        if not r:
            continue
        for tok in r.replace(';', ' ').replace(',', ' ').split():
            tok = tok.strip()
            if len(tok) > 3 and '@' in tok and tok.lower() not in seen:
                seen.add(tok.lower())
                out.append(tok)
    return out


def evaluate_and_send(project_id, image_item_id, dets, original_path, boxed_path, log=None):
    """
    dets: list of dicts with 'species' and 'confidence' (already filtered to
          the classification threshold by the runner).
    """
    def _log(m):
        if log:
            log(m)

    s = Session()
    try:
        project = s.query(Project).get(project_id)
        if project is None:
            _log(f'Alerting: project {project_id} not found.')
            return

        rules = s.query(AlertRule).filter_by(project_id=project_id, is_enabled=True).all()
        if not rules:
            return

        recipient_rows = [r.email for r in
                          s.query(AlertRecipient).filter_by(project_id=project_id).all()]
        to_addresses = _normalize_recipients(recipient_rows)
        if not to_addresses:
            _log('Alerting: no recipients configured.')
            return

        email_settings = s.query(ProjectEmailSettings).filter_by(project_id=project_id).first()
        if email_settings is None or not email_settings.smtp_ready:
            _log('Alerting: project email settings missing (cannot send).')
            return

        # Best confidence per label for this image.
        best_by_label = {}
        for d in dets:
            label = (d.get('species') or '').strip()
            if not label:
                continue
            conf = float(d.get('confidence') or 0.0)
            key = label.lower()
            if key not in best_by_label or conf > best_by_label[key][1]:
                best_by_label[key] = (label, conf)

        if not best_by_label:
            return

        now = datetime.utcnow()
        attach_paths = []
        for p in (boxed_path, original_path):
            if p and os.path.exists(p) and p not in attach_paths:
                attach_paths.append(p)

        for rule in rules:
            rule_label = (rule.label or '').strip()
            if not rule_label:
                continue
            match = best_by_label.get(rule_label.lower())
            if match is None:
                continue
            _, best_conf = match
            if best_conf < rule.min_confidence:
                continue

            already = s.query(AlertSent).filter_by(
                alert_rule_id=rule.id, image_item_id=image_item_id).first()
            if already:
                _log(f"Alert suppressed (already sent for this image): rule='{rule_label}' imageId={image_item_id}")
                continue

            try:
                _send_email(
                    settings=email_settings,
                    to_addresses=to_addresses,
                    subject=f'TrapTrackerRT Alert: {rule_label} ({best_conf:.2f}) — {project.name}',
                    body=(
                        f'Project: {project.name}\n'
                        f'Rule: {rule_label}\n'
                        f'Best confidence: {best_conf:.2f}\n'
                        f'ImageId: {image_item_id}\n'
                        f'Time (UTC): {now:%Y-%m-%d %H:%M:%S}\n'
                    ),
                    attach_paths=attach_paths,
                )
                s.add(AlertSent(alert_rule_id=rule.id, image_item_id=image_item_id, sent_at_utc=now))
                s.commit()
                _log(f"Alert sent: {rule_label} from {email_settings.email_address} "
                     f"-> {', '.join(to_addresses)} (imageId={image_item_id})")
            except Exception as ex:
                s.rollback()
                _log(f"Alert send failed for rule '{rule_label}': {ex}")
    except Exception as ex:
        _log(f'AlertingService fatal error: {ex}')
    finally:
        s.close()


def _send_email(settings, to_addresses, subject, body, attach_paths):
    msg = EmailMessage()
    msg['From'] = settings.email_address.strip()
    msg['To'] = ', '.join(to_addresses)
    msg['Subject'] = subject
    msg.set_content(body)

    for path in attach_paths or []:
        try:
            with open(path, 'rb') as fh:
                data = fh.read()
            ctype, _ = mimetypes.guess_type(path)
            if ctype and '/' in ctype:
                maintype, subtype = ctype.split('/', 1)
            else:
                maintype, subtype = 'image', 'jpeg'
            msg.add_attachment(data, maintype=maintype, subtype=subtype,
                               filename=os.path.basename(path))
        except Exception:
            pass

    host = settings.smtp_host or 'smtp.gmail.com'
    port = settings.smtp_port or 587

    refused = _smtp_deliver(host, port, settings.smtp_mode, settings, msg)
    if refused:
        raise RuntimeError(f'SMTP accepted connection but refused recipients: {refused}')


def _smtp_deliver(host, port, mode, settings, msg):
    """Send and return the dict of refused recipients (empty if all accepted).

    mode is 'starttls', 'ssl' or 'none' (plain SMTP, for a relay on the local
    network). Login is skipped when no password is set, as an internal relay
    usually accepts mail from the LAN without authentication.
    """
    smtp_cls = smtplib.SMTP_SSL if mode == 'ssl' else smtplib.SMTP
    with smtp_cls(host, port, timeout=20) as smtp:
        if mode == 'starttls':
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
        if settings.app_password:
            smtp.login(settings.email_address.strip(), settings.app_password)
        return smtp.send_message(msg)


def send_test_email(settings, to_address, log=None):
    """Used by the UI 'send test' button."""
    msg = EmailMessage()
    msg['From'] = settings.email_address.strip()
    msg['To'] = to_address
    msg['Subject'] = 'TrapTrackerRT — test email'
    msg.set_content('This is a test email from TrapTracker RT. Email settings are working.')
    host = settings.smtp_host or 'smtp.gmail.com'
    port = settings.smtp_port or 587
    refused = _smtp_deliver(host, port, settings.smtp_mode, settings, msg)
    if refused:
        raise RuntimeError(f'SMTP refused recipients: {refused}')
