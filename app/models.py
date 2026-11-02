"""
Database models for TrapTracker RT (Docker).

Schema mirrors the desktop .NET EF Core entities so the domain logic
(projects, cameras, per-project email settings, alert rules/recipients,
images, detections, alert dedupe) is preserved.
"""
import os
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, UniqueConstraint, Index,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, scoped_session

from .config import Config

Base = declarative_base()


class Project(Base):
    __tablename__ = 'projects'
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    root_folder = Column(String(1024), default='')
    model_folder = Column(String(200), default='uk-mammals')
    is_active = Column(Boolean, default=True)
    desired_running = Column(Boolean, default=False)
    created_at_utc = Column(DateTime, default=datetime.utcnow)

    cameras = relationship('Camera', back_populates='project', cascade='all, delete-orphan')
    email_settings = relationship('ProjectEmailSettings', back_populates='project',
                                  uselist=False, cascade='all, delete-orphan')
    alert_rules = relationship('AlertRule', back_populates='project', cascade='all, delete-orphan')
    recipients = relationship('AlertRecipient', back_populates='project', cascade='all, delete-orphan')
    images = relationship('ImageItem', back_populates='project', cascade='all, delete-orphan')


class Camera(Base):
    __tablename__ = 'cameras'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    name = Column(String(200), nullable=False)
    routing_key = Column(String(200), default='')
    camera_code = Column(String(200))
    phone_number = Column(String(50))
    connection_type = Column(String(50), default='Unknown')
    location_description = Column(String(500))
    notes = Column(String(1000))
    is_enabled = Column(Boolean, default=True)

    project = relationship('Project', back_populates='cameras')
    __table_args__ = (UniqueConstraint('project_id', 'name', name='uq_camera_project_name'),)


class ProjectEmailSettings(Base):
    __tablename__ = 'project_email_settings'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False, unique=True)
    provider = Column(String(50), default='Gmail')
    imap_host = Column(String(255), default='imap.gmail.com')
    imap_port = Column(Integer, default=993)
    use_ssl = Column(Boolean, default=True)
    smtp_host = Column(String(255), default='smtp.gmail.com')
    smtp_port = Column(Integer, default=587)
    smtp_starttls = Column(Boolean, default=True)
    email_address = Column(String(254), default='')
    app_password = Column(String(255), default='')
    search_query = Column(String(500), default='has:attachment newer_than:7d')
    created_at_utc = Column(DateTime, default=datetime.utcnow)
    # Off = images only arrive in Inbound/ by other means (FTP/SFTP, synced folder).
    imap_enabled = Column(Boolean, default=True)
    # 'starttls' | 'ssl' | 'none'. Empty on older databases: derived from smtp_starttls.
    smtp_security = Column(String(10), default='')

    project = relationship('Project', back_populates='email_settings')

    @property
    def imap_ready(self):
        """Mailbox ingest is switched on and has what it needs to log in."""
        return bool(self.imap_enabled is not False and self.imap_host
                    and self.email_address and self.app_password)

    @property
    def smtp_mode(self):
        if self.smtp_security in ('starttls', 'ssl', 'none'):
            return self.smtp_security
        return 'starttls' if self.smtp_starttls is not False else 'ssl'

    @property
    def smtp_ready(self):
        """Enough to send alerts: a server and a From address (login is optional)."""
        return bool(self.smtp_host and self.email_address)


class AlertRule(Base):
    __tablename__ = 'alert_rules'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    label = Column(String(200), nullable=False)
    min_confidence = Column(Float, default=0.50)
    cooldown_seconds = Column(Integer, default=300)
    is_enabled = Column(Boolean, default=True)

    project = relationship('Project', back_populates='alert_rules')
    __table_args__ = (UniqueConstraint('project_id', 'label', name='uq_rule_project_label'),)


class AlertRecipient(Base):
    __tablename__ = 'alert_recipients'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    email = Column(String(254), nullable=False)

    project = relationship('Project', back_populates='recipients')
    __table_args__ = (UniqueConstraint('project_id', 'email', name='uq_recipient_project_email'),)


class ImageItem(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    camera_id = Column(Integer, ForeignKey('cameras.id'), nullable=True)
    camera_name = Column(String(200), default='')
    file_name = Column(String(260), default='')
    full_path = Column(String(2048), default='')
    boxed_path = Column(String(2048), default='')
    received_at_utc = Column(DateTime, default=datetime.utcnow)
    processed_at_utc = Column(DateTime, nullable=True)

    project = relationship('Project', back_populates='images')
    detections = relationship('Detection', back_populates='image', cascade='all, delete-orphan')
    __table_args__ = (Index('ix_images_full_path', 'full_path'),)


class Detection(Base):
    __tablename__ = 'detections'
    id = Column(Integer, primary_key=True)
    image_item_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    label = Column(String(200), default='')
    confidence = Column(Float, default=0.0)
    x1 = Column(Float, default=0.0)
    y1 = Column(Float, default=0.0)
    x2 = Column(Float, default=0.0)
    y2 = Column(Float, default=0.0)
    # Two-stage models only: stage 1's own label/score (e.g. 'animal' 0.91)
    # before the classifier assigned the species. NULL for single-stage.
    detector_label = Column(String(200), nullable=True)
    detector_confidence = Column(Float, nullable=True)
    created_at_utc = Column(DateTime, default=datetime.utcnow)

    image = relationship('ImageItem', back_populates='detections')
    __table_args__ = (Index('ix_detections_image', 'image_item_id'),)


class AlertSent(Base):
    __tablename__ = 'alerts_sent'
    id = Column(Integer, primary_key=True)
    alert_rule_id = Column(Integer, ForeignKey('alert_rules.id'), nullable=False)
    image_item_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    sent_at_utc = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (Index('ix_alertsent_rule_time', 'alert_rule_id', 'sent_at_utc'),)


_engine = None
_SessionFactory = None


def init_engine():
    global _engine, _SessionFactory
    if _engine is not None:
        return _engine
    os.makedirs(os.path.dirname(Config.DB_PATH), exist_ok=True)
    _engine = create_engine(
        f'sqlite:///{Config.DB_PATH}',
        connect_args={'check_same_thread': False, 'timeout': 30},
        future=True,
    )
    # SQLite hardening (mirrors desktop WAL settings).
    from sqlalchemy import event

    @event.listens_for(_engine, 'connect')
    def _set_pragmas(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute('PRAGMA journal_mode=WAL;')
        cur.execute('PRAGMA synchronous=NORMAL;')
        cur.execute('PRAGMA busy_timeout=5000;')
        cur.close()

    Base.metadata.create_all(_engine)
    _run_lightweight_migrations(_engine)
    _SessionFactory = scoped_session(sessionmaker(bind=_engine, future=True, expire_on_commit=False))
    return _engine


def _run_lightweight_migrations(engine):
    """Additive, idempotent column adds for databases created by older versions."""
    from sqlalchemy import text
    wanted = {
        'projects': [('desired_running', 'BOOLEAN DEFAULT 0')],
        'detections': [('detector_label', 'VARCHAR(200)'), ('detector_confidence', 'FLOAT')],
        'project_email_settings': [('imap_enabled', 'BOOLEAN DEFAULT 1'),
                                   ("smtp_security", "VARCHAR(10) DEFAULT ''")],
    }
    with engine.begin() as conn:
        for table, cols in wanted.items():
            existing = {row[1] for row in conn.execute(text(f'PRAGMA table_info({table})'))}
            for col_name, col_def in cols:
                if col_name not in existing:
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {col_name} {col_def}'))


def Session():
    if _SessionFactory is None:
        init_engine()
    return _SessionFactory()
