"""
TrapTracker RT — Flask web application.

Real-time camera-trap monitoring: create projects, assign cameras, set
the species watch-list (alert rules) + recipients, configure the project
mailbox, then start a background runner that ingests camera emails,
classifies images and emails alerts.

Single trusted deployment (no login). Branding matches TrapTracker DT.
"""
import os
import secrets
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, abort, jsonify,
)

from .config import Config
from .models import (
    Session, init_engine, Project, Camera, ProjectEmailSettings,
    AlertRule, AlertRecipient, ImageItem, Detection,
)
from .detector import load_classes, available_providers
from . import project_folders as pf
from . import alerting
from . import model_registry
from .runner import manager


def _resolve_secret_key():
    """Use SECRET_KEY if provided, else generate and persist a random one."""
    if Config.SECRET_KEY:
        return Config.SECRET_KEY
    key_path = os.path.join(Config.DATA_DIR, '.secret_key')
    try:
        if os.path.exists(key_path):
            with open(key_path, 'r', encoding='utf-8') as fh:
                existing = fh.read().strip()
            if existing:
                return existing
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        generated = secrets.token_hex(32)
        with open(key_path, 'w', encoding='utf-8') as fh:
            fh.write(generated)
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass
        return generated
    except OSError:
        # Last resort: ephemeral key (sessions reset on restart).
        return secrets.token_hex(32)


app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = _resolve_secret_key()
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

init_engine()


# ----------------------------------------------------------------------
# Template context (sidebar status)
# ----------------------------------------------------------------------
@app.context_processor
def inject_globals():
    from .detector import LAST_PROVIDER_REPORT
    providers = available_providers()
    cuda_installed = any('CUDA' in p or 'Tensorrt' in p for p in providers)
    active = LAST_PROVIDER_REPORT.get('active')
    if active == 'CUDAExecutionProvider':
        compute_label = 'GPU'
    elif active == 'CPUExecutionProvider':
        compute_label = 'CPU'
    else:
        # No model loaded yet — show what the build is capable of.
        compute_label = 'GPU-capable' if cuda_installed else 'CPU'
    models = model_registry.list_models()
    return dict(
        providers=providers,
        gpu_ready=cuda_installed,
        compute_label=compute_label,
        model_present=len(models) > 0,
        model_count=len(models),
        running_count=len(manager.running_ids()),
    )


def _species_list(model_id=None):
    """Class names for a given project's model (falls back to first model)."""
    return model_registry.classes_for_model(model_id)


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------
@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')


@app.route('/')
def dashboard():
    s = Session()
    try:
        projects = s.query(Project).order_by(Project.created_at_utc.desc()).all()
        total_projects = len(projects)
        total_cameras = s.query(Camera).count()
        total_images = s.query(ImageItem).count()
        total_detections = s.query(Detection).count()
        running = manager.running_ids()
        rows = []
        for p in projects:
            rows.append({
                'project': p,
                'cameras': s.query(Camera).filter_by(project_id=p.id).count(),
                'rules': s.query(AlertRule).filter_by(project_id=p.id).count(),
                'images': s.query(ImageItem).filter_by(project_id=p.id).count(),
                'running': p.id in running,
            })
        return render_template(
            'dashboard.html', rows=rows,
            total_projects=total_projects, total_cameras=total_cameras,
            total_images=total_images, total_detections=total_detections,
            running_count=len(running),
        )
    finally:
        s.close()


# ----------------------------------------------------------------------
# Projects
# ----------------------------------------------------------------------
@app.route('/projects/new', methods=['GET', 'POST'])
def project_new():
    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()
        model_folder = (request.form.get('model_folder') or '').strip()
        if not name:
            flash('Project name is required.')
            return redirect(url_for('project_new'))
        s = Session()
        try:
            if s.query(Project).filter_by(name=name).first():
                flash('A project with that name already exists.')
                return redirect(url_for('project_new'))
            p = Project(name=name, model_folder=model_folder, created_at_utc=datetime.utcnow())
            s.add(p)
            s.flush()
            p.root_folder = os.path.join(Config.PROJECTS_DIR, f'{p.id}_{pf.sanitize_folder_name(name)}')
            # Default email settings row.
            s.add(ProjectEmailSettings(
                project_id=p.id, provider=Config.DEFAULT_PROVIDER,
                imap_host=Config.DEFAULT_IMAP_HOST, imap_port=Config.DEFAULT_IMAP_PORT,
                use_ssl=Config.DEFAULT_IMAP_SSL, smtp_host=Config.DEFAULT_SMTP_HOST,
                smtp_port=Config.DEFAULT_SMTP_PORT, smtp_starttls=Config.DEFAULT_SMTP_STARTTLS,
            ))
            s.commit()
            pf.ensure_project_folders(p)
            flash('Project created.')
            return redirect(url_for('project_detail', project_id=p.id))
        finally:
            s.close()
    return render_template('project_form.html', project=None,
                           models=model_registry.list_models())


@app.route('/projects/<int:project_id>')
def project_detail(project_id):
    s = Session()
    try:
        p = s.query(Project).get(project_id)
        if not p:
            abort(404)
        cameras = s.query(Camera).filter_by(project_id=project_id).all()
        rules = s.query(AlertRule).filter_by(project_id=project_id).all()
        recipients = s.query(AlertRecipient).filter_by(project_id=project_id).all()
        settings = p.email_settings
        image_count = s.query(ImageItem).filter_by(project_id=project_id).count()
        return render_template(
            'project_detail.html', project=p, cameras=cameras, rules=rules,
            recipients=recipients, settings=settings,
            species=_species_list(p.model_folder),
            models=model_registry.list_models(),
            current_model=model_registry.get_model(p.model_folder),
            image_count=image_count, running=manager.is_running(project_id),
            status=manager.status(project_id),
        )
    finally:
        s.close()


@app.route('/projects/<int:project_id>/edit', methods=['GET', 'POST'])
def project_edit(project_id):
    s = Session()
    try:
        p = s.query(Project).get(project_id)
        if not p:
            abort(404)
        if request.method == 'POST':
            name = (request.form.get('name') or '').strip()
            if name:
                p.name = name
            p.model_folder = (request.form.get('model_folder') or p.model_folder).strip()
            s.commit()
            flash('Project updated.')
            return redirect(url_for('project_detail', project_id=project_id))
        return render_template('project_form.html', project=p,
                               models=model_registry.list_models())
    finally:
        s.close()


@app.route('/projects/<int:project_id>/delete', methods=['POST'])
def project_delete(project_id):
    manager.stop(project_id)
    s = Session()
    try:
        p = s.query(Project).get(project_id)
        if p:
            s.delete(p)
            s.commit()
            flash('Project deleted.')
    finally:
        s.close()
    return redirect(url_for('dashboard'))


# ----------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------
@app.route('/projects/<int:project_id>/cameras/add', methods=['POST'])
def camera_add(project_id):
    name = (request.form.get('name') or '').strip()
    routing_key = (request.form.get('routing_key') or '').strip() or name
    s = Session()
    try:
        if not name:
            flash('Camera name is required.')
        elif s.query(Camera).filter_by(project_id=project_id, name=name).first():
            flash('A camera with that name already exists in this project.')
        else:
            s.add(Camera(
                project_id=project_id, name=name, routing_key=routing_key,
                camera_code=(request.form.get('camera_code') or '').strip() or None,
                phone_number=(request.form.get('phone_number') or '').strip() or None,
                connection_type=(request.form.get('connection_type') or 'Unknown').strip(),
                location_description=(request.form.get('location_description') or '').strip() or None,
                notes=(request.form.get('notes') or '').strip() or None,
                is_enabled=True,
            ))
            s.commit()
            flash('Camera added.')
    finally:
        s.close()
    return redirect(url_for('project_detail', project_id=project_id))


@app.route('/cameras/<int:camera_id>/toggle', methods=['POST'])
def camera_toggle(camera_id):
    s = Session()
    try:
        c = s.query(Camera).get(camera_id)
        if c:
            c.is_enabled = not c.is_enabled
            s.commit()
            pid = c.project_id
        else:
            pid = None
    finally:
        s.close()
    return redirect(url_for('project_detail', project_id=pid) if pid else url_for('dashboard'))


@app.route('/cameras/<int:camera_id>/delete', methods=['POST'])
def camera_delete(camera_id):
    s = Session()
    try:
        c = s.query(Camera).get(camera_id)
        pid = c.project_id if c else None
        if c:
            s.delete(c)
            s.commit()
            flash('Camera removed.')
    finally:
        s.close()
    return redirect(url_for('project_detail', project_id=pid) if pid else url_for('dashboard'))


# ----------------------------------------------------------------------
# Email settings
# ----------------------------------------------------------------------
@app.route('/projects/<int:project_id>/email', methods=['POST'])
def email_settings_save(project_id):
    s = Session()
    try:
        settings = s.query(ProjectEmailSettings).filter_by(project_id=project_id).first()
        if not settings:
            settings = ProjectEmailSettings(project_id=project_id)
            s.add(settings)
        settings.provider = (request.form.get('provider') or 'Gmail').strip()
        settings.imap_host = (request.form.get('imap_host') or 'imap.gmail.com').strip()
        settings.imap_port = int(request.form.get('imap_port') or 993)
        settings.use_ssl = request.form.get('use_ssl') == 'on'
        settings.smtp_host = (request.form.get('smtp_host') or 'smtp.gmail.com').strip()
        settings.smtp_port = int(request.form.get('smtp_port') or 587)
        settings.smtp_starttls = request.form.get('smtp_starttls') == 'on'
        settings.email_address = (request.form.get('email_address') or '').strip()
        pw = request.form.get('app_password') or ''
        if pw.strip():  # only overwrite when provided
            settings.app_password = pw
        settings.search_query = (request.form.get('search_query') or '').strip()
        s.commit()
        flash('Email settings saved.')
    finally:
        s.close()
    return redirect(url_for('project_detail', project_id=project_id) + '#email')


@app.route('/projects/<int:project_id>/email/test', methods=['POST'])
def email_settings_test(project_id):
    s = Session()
    try:
        settings = s.query(ProjectEmailSettings).filter_by(project_id=project_id).first()
        if not settings or not settings.email_address or not settings.app_password:
            flash('Configure and save email settings first.')
            return redirect(url_for('project_detail', project_id=project_id) + '#email')
        to = (request.form.get('test_to') or settings.email_address).strip()
        try:
            alerting.send_test_email(settings, to)
            flash(f'Test email sent to {to}.')
        except Exception as ex:
            flash(f'Test failed: {ex}')
    finally:
        s.close()
    return redirect(url_for('project_detail', project_id=project_id) + '#email')


# ----------------------------------------------------------------------
# Alert rules (species watch-list) + recipients
# ----------------------------------------------------------------------
@app.route('/projects/<int:project_id>/rules/add', methods=['POST'])
def rule_add(project_id):
    label = (request.form.get('label') or '').strip()
    s = Session()
    try:
        if not label:
            flash('Select a species.')
        elif s.query(AlertRule).filter_by(project_id=project_id, label=label).first():
            flash('That species is already on the watch-list.')
        else:
            min_conf = float(request.form.get('min_confidence') or Config.DEFAULT_MIN_CONFIDENCE)
            cooldown = int(request.form.get('cooldown_seconds') or Config.DEFAULT_COOLDOWN_SECONDS)
            s.add(AlertRule(project_id=project_id, label=label,
                            min_confidence=max(0.0, min(1.0, min_conf)),
                            cooldown_seconds=max(0, cooldown), is_enabled=True))
            s.commit()
            flash(f'Added {label} to the watch-list.')
    finally:
        s.close()
    return redirect(url_for('project_detail', project_id=project_id) + '#alerts')


@app.route('/rules/<int:rule_id>/toggle', methods=['POST'])
def rule_toggle(rule_id):
    s = Session()
    try:
        r = s.query(AlertRule).get(rule_id)
        pid = r.project_id if r else None
        if r:
            r.is_enabled = not r.is_enabled
            s.commit()
    finally:
        s.close()
    return redirect(url_for('project_detail', project_id=pid) + '#alerts' if pid else url_for('dashboard'))


@app.route('/rules/<int:rule_id>/delete', methods=['POST'])
def rule_delete(rule_id):
    s = Session()
    try:
        r = s.query(AlertRule).get(rule_id)
        pid = r.project_id if r else None
        if r:
            s.delete(r)
            s.commit()
    finally:
        s.close()
    return redirect(url_for('project_detail', project_id=pid) + '#alerts' if pid else url_for('dashboard'))


@app.route('/projects/<int:project_id>/recipients/add', methods=['POST'])
def recipient_add(project_id):
    email_addr = (request.form.get('email') or '').strip()
    s = Session()
    try:
        if '@' not in email_addr:
            flash('Enter a valid email address.')
        elif s.query(AlertRecipient).filter_by(project_id=project_id, email=email_addr).first():
            flash('That recipient is already added.')
        else:
            s.add(AlertRecipient(project_id=project_id, email=email_addr))
            s.commit()
            flash('Recipient added.')
    finally:
        s.close()
    return redirect(url_for('project_detail', project_id=project_id) + '#alerts')


@app.route('/recipients/<int:recipient_id>/delete', methods=['POST'])
def recipient_delete(recipient_id):
    s = Session()
    try:
        r = s.query(AlertRecipient).get(recipient_id)
        pid = r.project_id if r else None
        if r:
            s.delete(r)
            s.commit()
    finally:
        s.close()
    return redirect(url_for('project_detail', project_id=pid) + '#alerts' if pid else url_for('dashboard'))


# ----------------------------------------------------------------------
# Runner control
# ----------------------------------------------------------------------
@app.route('/projects/<int:project_id>/start', methods=['POST'])
def project_start(project_id):
    s = Session()
    try:
        p = s.query(Project).get(project_id)
        if not p:
            abort(404)
        settings = p.email_settings
        if not settings or not settings.email_address or not settings.app_password:
            flash('Configure email settings before starting monitoring.')
            return redirect(url_for('project_detail', project_id=project_id))
        if not os.path.exists(Config.MODEL_PATH) and not model_registry.list_models():
            flash(f'No ONNX models found in {Config.MODEL_DIR}. Mount your model file(s) first.')
            return redirect(url_for('project_detail', project_id=project_id))
        if model_registry.get_model(p.model_folder) is None:
            flash('No usable model found for this project. Check the models folder.')
            return redirect(url_for('project_detail', project_id=project_id))
    finally:
        s.close()
    if manager.start(project_id):
        _set_desired_running(project_id, True)
        flash('Monitoring started.')
    else:
        flash('Monitoring is already running.')
    return _redirect_back(project_id)


@app.route('/projects/<int:project_id>/stop', methods=['POST'])
def project_stop(project_id):
    manager.stop(project_id)
    _set_desired_running(project_id, False)
    flash('Monitoring stopping...')
    return _redirect_back(project_id)


def _set_desired_running(project_id, value):
    s = Session()
    try:
        p = s.query(Project).get(project_id)
        if p:
            p.desired_running = bool(value)
            s.commit()
    finally:
        s.close()


def _redirect_back(project_id):
    """Return to the dashboard if the action came from there, else the detail page."""
    if (request.form.get('return_to') or '') == 'dashboard':
        return redirect(url_for('dashboard'))
    return redirect(url_for('project_detail', project_id=project_id))


@app.route('/projects/<int:project_id>/logs')
def project_logs(project_id):
    return jsonify({
        'status': manager.status(project_id),
        'running': manager.is_running(project_id),
        'logs': manager.logs(project_id),
    })


# ----------------------------------------------------------------------
# Gallery + image serving
# ----------------------------------------------------------------------
@app.route('/projects/<int:project_id>/gallery')
def gallery(project_id):
    s = Session()
    try:
        p = s.query(Project).get(project_id)
        if not p:
            abort(404)
        images = (s.query(ImageItem)
                  .filter_by(project_id=project_id)
                  .order_by(ImageItem.received_at_utc.desc())
                  .limit(200).all())
        rows = []
        for img in images:
            dets = s.query(Detection).filter_by(image_item_id=img.id).all()
            best = max(dets, key=lambda d: d.confidence) if dets else None
            rows.append({
                'image': img,
                'best': best,
                'count': len(dets),
                'has_boxed': bool(img.boxed_path and os.path.exists(img.boxed_path)),
            })
        return render_template('gallery.html', project=p, rows=rows)
    finally:
        s.close()


@app.route('/image/<int:image_id>')
@app.route('/image/<int:image_id>/<kind>')
def serve_image(image_id, kind='boxed'):
    s = Session()
    try:
        img = s.query(ImageItem).get(image_id)
        if not img:
            abort(404)
        path = img.boxed_path if (kind == 'boxed' and img.boxed_path) else img.full_path
        if not path or not os.path.exists(path):
            path = img.full_path
        if not path or not os.path.exists(path):
            abort(404)
        return send_file(path)
    finally:
        s.close()


@app.route('/healthz')
def healthz():
    models = model_registry.list_models()
    from .detector import LAST_PROVIDER_REPORT
    return jsonify({
        'ok': True,
        'models': [{'id': m['id'], 'classes': m['num_classes']} for m in models],
        'compute': {
            'available_providers': available_providers(),
            'active': LAST_PROVIDER_REPORT.get('active'),
            'gpu_ok': LAST_PROVIDER_REPORT.get('gpu_ok'),
            'message': LAST_PROVIDER_REPORT.get('message'),
        },
    })


def _resume_running_projects():
    """Restart monitoring for any project that was running before a restart."""
    s = Session()
    try:
        ids = [p.id for p in s.query(Project).filter_by(desired_running=True).all()]
    except Exception:
        ids = []
    finally:
        s.close()
    for pid in ids:
        try:
            if model_registry.get_model(_project_model_id(pid)) is not None:
                manager.start(pid)
                print(f'Resumed monitoring for project {pid}')
        except Exception as ex:
            print(f'Could not resume project {pid}: {ex}')


def _project_model_id(project_id):
    s = Session()
    try:
        p = s.query(Project).get(project_id)
        return p.model_folder if p else None
    finally:
        s.close()


def main():
    from waitress import serve
    _resume_running_projects()
    models = model_registry.list_models()
    print('=' * 60)
    print(f'TrapTracker RT listening on http://{Config.HOST}:{Config.PORT}')
    print(f'Models found: {len(models)}')
    if Config.HOST not in ('127.0.0.1', 'localhost'):
        print('WARNING: HOST is not localhost. The UI exposes email')
        print('         credentials and should only be reachable on a')
        print('         trusted network. Set HOST=127.0.0.1 unless you')
        print('         have secured access another way.')
    print('=' * 60)
    serve(app, host=Config.HOST, port=Config.PORT, threads=8)


if __name__ == '__main__':
    main()
