import json
import uuid
import hashlib
from datetime import datetime, timedelta

from sqlalchemy import select, delete, text
from sqlalchemy.orm import selectinload

from .config import SUPERADMIN_EMAIL, SUPERADMIN_PASSWORD, SYSTEM_LOCK_DEFAULT, ANON_SALT
from .db import engine, SessionLocal
from .models import (
    User,
    Role,
    Permission,
    UserRole,
    RolePermission,
    MoodEntry,
    AIFlag,
    AuditLog,
    ActivityLog,
    SystemSetting,
)
from .security import (
    hash_password,
    verify_password,
)


PERMISSIONS = {
    "flags:view": "View flagged cases (anonymized)",
    "flags:review": "Review AI-flagged content",
    "users:warn": "Warn users",
    "users:suspend": "Suspend users",
    "cases:escalate": "Escalate severe cases",
    "dashboard:view": "View system dashboard",
    "users:manage": "Manage users",
    "settings:write": "Edit system settings",
    "logs:view": "View system logs",
    "roles:assign": "Assign roles and permissions",
    "keys:manage": "Manage API keys",
    "backup:manage": "Backup/restore database",
    "system:lock": "Lock system during emergency",
    "db:full": "Full database access",
}

ROLE_PERMS = {
    "user": [],
    "moderator": [
        "flags:view",
        "flags:review",
        "users:warn",
        "users:suspend",
        "cases:escalate",
    ],
    "admin": [
        "dashboard:view",
        "users:manage",
        "settings:write",
        "logs:view",
        "flags:view",
    ],
    "superadmin": list(PERMISSIONS.keys()),
}


USER_LOAD_OPTIONS = (
    selectinload(User.roles).selectinload(Role.permissions),
)


def _normalize_username(username):
    return str(username or "").strip().lower()


def hash_subject(value):
    normalized = _normalize_username(value)
    if not normalized:
        return ""
    base = f"{ANON_SALT}:{normalized}" if ANON_SALT else normalized
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def init_db():
    from .db import Base

    Base.metadata.create_all(engine)
    ensure_user_columns()
    seed_permissions_and_roles()
    seed_system_settings()
    bootstrap_superadmin()


def ensure_user_columns():
    from sqlalchemy import text

    def _table_columns(conn, table_name):
        try:
            rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
        except Exception:
            return set()
        return {row[1] for row in rows}

    def _ensure_column(conn, table_name, column_name, ddl):
        cols = _table_columns(conn, table_name)
        if cols and column_name not in cols:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}"))
            cols.add(column_name)
        return cols

    with engine.connect() as conn:
        # Users table evolved from a legacy schema that used anon/email_hash fields.
        users_cols = _table_columns(conn, "users")
        if users_cols:
            if "twofa_secret_enc" not in users_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN twofa_secret_enc TEXT"))
            if "username" not in users_cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN username VARCHAR(80)"))
                users_cols.add("username")
            if "username" in users_cols:
                if "anon_id" in users_cols:
                    conn.execute(
                        text(
                            "UPDATE users "
                            "SET username = COALESCE(NULLIF(username, ''), anon_id) "
                            "WHERE anon_id IS NOT NULL"
                        )
                    )
                elif "email_hash" in users_cols:
                    conn.execute(
                        text(
                            "UPDATE users "
                            "SET username = COALESCE(NULLIF(username, ''), email_hash) "
                            "WHERE email_hash IS NOT NULL"
                        )
                    )
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_username ON users(username)"))

        # Mood/event tables previously stored anon_id; current code writes subject_id.
        mood_cols = _ensure_column(conn, "mood_entries", "subject_id", "VARCHAR(64)")
        if mood_cols and "anon_id" in mood_cols:
            conn.execute(
                text(
                    "UPDATE mood_entries "
                    "SET subject_id = COALESCE(NULLIF(subject_id, ''), anon_id) "
                    "WHERE anon_id IS NOT NULL"
                )
            )
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_mood_entries_subject_id ON mood_entries(subject_id)"))

        activity_cols = _ensure_column(conn, "activity_logs", "subject_id", "VARCHAR(64)")
        if activity_cols and "anon_id" in activity_cols:
            conn.execute(
                text(
                    "UPDATE activity_logs "
                    "SET subject_id = COALESCE(NULLIF(subject_id, ''), anon_id) "
                    "WHERE anon_id IS NOT NULL"
                )
            )
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_activity_logs_subject_id ON activity_logs(subject_id)"))

        conn.commit()


def seed_permissions_and_roles():
    with SessionLocal() as db:
        existing_perms = {p.name for p in db.execute(select(Permission)).scalars().all()}
        for name, desc in PERMISSIONS.items():
            if name not in existing_perms:
                db.add(Permission(name=name, description=desc))
        db.commit()

        roles = {r.name: r for r in db.execute(select(Role)).scalars().all()}
        for role_name in ROLE_PERMS.keys():
            if role_name not in roles:
                db.add(Role(name=role_name, description=f"{role_name} role"))
        db.commit()

        # Attach permissions to roles
        roles = {r.name: r for r in db.execute(select(Role)).scalars().all()}
        perms = {p.name: p for p in db.execute(select(Permission)).scalars().all()}

        for role_name, perm_list in ROLE_PERMS.items():
            role = roles.get(role_name)
            if not role:
                continue
            current = {p.name for p in role.permissions}
            for perm_name in perm_list:
                perm = perms.get(perm_name)
                if perm and perm.name not in current:
                    role.permissions.append(perm)
        db.commit()


def seed_system_settings():
    with SessionLocal() as db:
        existing = {s.key for s in db.execute(select(SystemSetting)).scalars().all()}
        defaults = {
            "ai_threshold": "0.7",
            "llm_mode": "llm",
            "system_lock": "true" if SYSTEM_LOCK_DEFAULT else "false",
            "enforce_2fa": "false",
            "model_accuracy": "n/a",
        }
        for key, val in defaults.items():
            if key not in existing:
                db.add(SystemSetting(key=key, value=val))
        db.commit()


def bootstrap_superadmin():
    username = _normalize_username(SUPERADMIN_EMAIL)
    if not username or not SUPERADMIN_PASSWORD:
        return
    with SessionLocal() as db:
        if get_user_by_username(db, username):
            return
        user = create_user(
            db,
            username=username,
            password=SUPERADMIN_PASSWORD,
            roles=["superadmin"],
        )
        if user:
            db.commit()


def create_user(db, username, password, roles=None):
    username = _normalize_username(username)
    if not username or not password:
        return None
    if get_user_by_username(db, username):
        return None
    user = User(
        username=username,
        password_hash=hash_password(password),
        status="active",
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    if roles:
        assign_roles(db, user, roles)
    return user


def get_user_by_username(db, username):
    username = _normalize_username(username)
    if not username:
        return None
    stmt = select(User).options(*USER_LOAD_OPTIONS).where(User.username == username)
    return db.execute(stmt).scalars().first()


def get_user_by_id(db, user_id):
    stmt = select(User).options(*USER_LOAD_OPTIONS).where(User.id == user_id)
    return db.execute(stmt).scalars().first()


def get_user_by_anon_id(db, anon_id):
    anon_id = str(anon_id or "").strip().lower()
    if not anon_id:
        return None
    stmt = select(User).options(*USER_LOAD_OPTIONS).where(User.username == anon_id)
    return db.execute(stmt).scalars().first()


def assign_roles(db, user, roles):
    role_rows = db.execute(select(Role).where(Role.name.in_(roles))).scalars().all()
    for role in role_rows:
        if role not in user.roles:
            user.roles.append(role)
    db.commit()
    db.refresh(user)
    return user


def update_role_permissions(role_name, perm_names):
    with SessionLocal() as db:
        role = db.execute(select(Role).where(Role.name == role_name)).scalars().first()
        if not role:
            return False
        perms = db.execute(select(Permission).where(Permission.name.in_(perm_names))).scalars().all()
        role.permissions = perms
        db.commit()
        return True


def serialize_user(user):
    data = {
        "id": user.id,
        "username": user.username,
        "status": user.status,
        "created_at": user.created_at.isoformat() if user.created_at else "",
        "last_login": user.last_login.isoformat() if user.last_login else "",
        "roles": [r.name for r in user.roles],
        "twofa_enabled": bool(user.twofa_secret_enc),
    }
    return data


def authenticate_user(username, password):
    with SessionLocal() as db:
        user = get_user_by_username(db, username)
        if not user or not user.password_hash:
            return None
        if user.status in ("blocked", "deleted"):
            return None
        if user.suspended_until and user.suspended_until > datetime.utcnow():
            return None
        if not verify_password(password, user.password_hash):
            return None
        user.last_login = datetime.utcnow()
        db.commit()
        db.refresh(user)
        return user


def upsert_user_from_auth(identifier):
    """Create/update a minimal, pseudonymous user record without PII."""
    hashed = hash_subject(identifier)
    if not hashed:
        return None
    with SessionLocal() as db:
        user = get_user_by_username(db, hashed)
        if user:
            user.last_login = datetime.utcnow()
            db.commit()
            db.refresh(user)
            return user

        user = User(
            username=hashed,
            password_hash=hash_password(uuid.uuid4().hex),
            status="active",
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        assign_roles(db, user, ["user"])
        return user


def get_setting(key, default=None):
    with SessionLocal() as db:
        row = db.execute(select(SystemSetting).where(SystemSetting.key == key)).scalars().first()
        return row.value if row else default


def set_setting(key, value, updated_by=None):
    with SessionLocal() as db:
        row = db.execute(select(SystemSetting).where(SystemSetting.key == key)).scalars().first()
        if row:
            row.value = str(value)
            row.updated_at = datetime.utcnow()
            row.updated_by = updated_by
        else:
            db.add(SystemSetting(key=key, value=str(value), updated_by=updated_by))
        db.commit()


def is_system_locked():
    return get_setting("system_lock", "false") == "true"


def record_mood_entry(subject_id, mood, confidence, severity, source="unknown"):
    # Store mood events using anonymized IDs to protect identity.
    if not subject_id:
        return

    now = datetime.utcnow()
    payload = {
        "mood": mood or "neutral",
        "confidence": float(confidence or 0.0),
        "severity": severity or "low",
        "source": source or "unknown",
        "created_at": now,
    }

    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(mood_entries)")).fetchall()}
        insert_cols = []
        insert_params = dict(payload)

        if "subject_id" in cols:
            insert_cols.append("subject_id")
            insert_params["subject_id"] = subject_id
        if "anon_id" in cols:
            insert_cols.append("anon_id")
            insert_params["anon_id"] = subject_id

        for col in ("mood", "confidence", "severity", "source", "created_at"):
            if col in cols:
                insert_cols.append(col)

        if not insert_cols:
            return

        placeholders = ", ".join([f":{col}" for col in insert_cols])
        conn.execute(
            text(f"INSERT INTO mood_entries ({', '.join(insert_cols)}) VALUES ({placeholders})"),
            {key: insert_params[key] for key in insert_cols},
        )


def record_activity(subject_id, event_type, mood="", confidence=0.0, detail=""):
    now = datetime.utcnow()
    payload = {
        "event_type": event_type,
        "mood": mood or "",
        "confidence": float(confidence or 0.0),
        "detail": detail or "",
        "created_at": now,
    }

    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(activity_logs)")).fetchall()}
        insert_cols = []
        insert_params = dict(payload)

        if "subject_id" in cols:
            insert_cols.append("subject_id")
            insert_params["subject_id"] = subject_id
        if "anon_id" in cols:
            insert_cols.append("anon_id")
            insert_params["anon_id"] = subject_id

        for col in ("event_type", "mood", "confidence", "detail", "created_at"):
            if col in cols:
                insert_cols.append(col)

        if not insert_cols:
            return

        placeholders = ", ".join([f":{col}" for col in insert_cols])
        conn.execute(
            text(f"INSERT INTO activity_logs ({', '.join(insert_cols)}) VALUES ({placeholders})"),
            {key: insert_params[key] for key in insert_cols},
        )


def flag_content(subject_id, flag_type="high_risk", severity="medium", snippet=""):
    with SessionLocal() as db:
        db.add(
            AIFlag(
                subject_id=subject_id or "",
                flag_type=flag_type,
                severity=severity,
                status="open",
                review_note="",
                created_at=datetime.utcnow(),
            )
        )
        db.commit()


def log_audit(actor_id, actor_role, action, target="", detail="", ip=None):
    with SessionLocal() as db:
        db.add(
            AuditLog(
                actor_id=actor_id,
                actor_role=actor_role,
                action=action,
                target=target,
                detail=detail,
                created_at=datetime.utcnow(),
            )
        )
        db.commit()


def get_mood_distribution():
    with SessionLocal() as db:
        rows = db.execute(select(MoodEntry)).scalars().all()
    dist = {}
    for row in rows:
        dist[row.mood] = dist.get(row.mood, 0) + 1
    return dist


def get_user_counts():
    with SessionLocal() as db:
        total = db.execute(select(User)).scalars().all()
    active = [u for u in total if u.status == "active"]
    return {"total": len(total), "active": len(active)}


def list_users(limit=500):
    with SessionLocal() as db:
        users = db.execute(select(User).options(*USER_LOAD_OPTIONS)).scalars().all()
    users.sort(key=lambda u: u.created_at or datetime.utcnow(), reverse=True)
    if limit:
        users = users[: int(limit)]
    return users


def list_audit_logs(limit=500):
    with SessionLocal() as db:
        rows = db.execute(select(AuditLog)).scalars().all()
    rows.sort(key=lambda r: r.created_at or datetime.utcnow(), reverse=True)
    if limit:
        rows = rows[: int(limit)]
    return rows


def list_activity_logs(limit=500):
    with SessionLocal() as db:
        rows = db.execute(select(ActivityLog)).scalars().all()
    rows.sort(key=lambda r: r.created_at or datetime.utcnow(), reverse=True)
    if limit:
        rows = rows[: int(limit)]
    return rows


def get_flags(status="open"):
    with SessionLocal() as db:
        q = select(AIFlag)
        if status:
            q = q.where(AIFlag.status == status)
        return db.execute(q).scalars().all()


def update_flag_status(flag_id, status, reviewer_id=None, note=""):
    with SessionLocal() as db:
        flag = db.execute(select(AIFlag).where(AIFlag.id == flag_id)).scalars().first()
        if not flag:
            return None
        flag.status = status
        flag.reviewed_by = reviewer_id
        flag.review_note = note or ""
        db.commit()
        db.refresh(flag)
        return flag


def update_user_status(user_id=None, username=None, status="active", suspend_minutes=None):
    with SessionLocal() as db:
        user = get_user_by_username(db, username) if username else get_user_by_id(db, user_id)
        if not user:
            return None
        user.status = status
        if suspend_minutes:
            user.suspended_until = datetime.utcnow() + timedelta(minutes=int(suspend_minutes))
        db.commit()
        db.refresh(user)
        return user


def reset_user_password(user_id=None, username=None):
    with SessionLocal() as db:
        user = get_user_by_username(db, username) if username else get_user_by_id(db, user_id)
        if not user:
            return None
        temp_password = uuid.uuid4().hex[:10]
        user.password_hash = hash_password(temp_password)
        db.commit()
        return temp_password


def delete_inactive(days=90):
    cutoff = datetime.utcnow() - timedelta(days=days)
    with SessionLocal() as db:
        users = db.execute(select(User)).scalars().all()
        removed = 0
        for user in users:
            if user.last_login and user.last_login < cutoff:
                user.status = "deleted"
                removed += 1
        db.commit()
        return removed
