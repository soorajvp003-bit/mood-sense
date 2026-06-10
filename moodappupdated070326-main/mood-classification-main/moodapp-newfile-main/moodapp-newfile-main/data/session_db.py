import csv
import json
import os
import sqlite3
from datetime import datetime
from threading import Lock


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.getenv("MOODSENSE_DB_PATH", os.path.join(CURRENT_DIR, "ops.db"))
WEEKLY_TARGET = f"{DB_FILE} (table: weekly_overview)"
FACE_PROFILE_TARGET = "memory (non-persistent)"

LEGACY_TEXT_FILE = os.path.join(CURRENT_DIR, "text_sessions.csv")
LEGACY_VISUAL_FILE = os.path.join(CURRENT_DIR, "visual_sessions.csv")
LEGACY_OVERALL_FILE = os.path.join(CURRENT_DIR, "overall_sessions.csv")
LEGACY_WEEKLY_FILE = os.path.join(CURRENT_DIR, "weekly_overview.csv")
LEGACY_RESPONSES_FILE = os.path.join(CURRENT_DIR, "admin_responses.csv")
LEGACY_USERS_FILE = os.path.join(CURRENT_DIR, "users.csv")
LEGACY_ACTIVITY_FILE = os.path.join(CURRENT_DIR, "activity_log.csv")

TEXT_WEIGHT = 0.6
VISUAL_WEIGHT = 0.4

LEGACY_FILE_TO_TABLE = {
    LEGACY_TEXT_FILE.lower(): "text_sessions",
    LEGACY_VISUAL_FILE.lower(): "visual_sessions",
    LEGACY_OVERALL_FILE.lower(): "overall_sessions",
    LEGACY_WEEKLY_FILE.lower(): "weekly_overview",
    # Legacy tables intentionally excluded to avoid migrating sensitive data.
}

ADMIN_EVENTS_TABLE = "admin_events"
TRACKED_SUBJECTS_TABLE = "tracked_subjects"
ACTIVITY_EVENTS_TABLE = "activity_events"

_FACE_PROFILE_LOCK = Lock()
_FACE_PROFILE_STORE = {}

_STORAGE_READY = False


def _connect():
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_mood(mood):
    value = str(mood or "").strip().lower()
    aliases = {
        "joy": "happy",
        "love": "happy",
        "sadness": "sad",
        "anger": "angry",
        "annoyance": "angry",
        "disapproval": "contempt",
        "anxiety": "fear",
        "nervousness": "fear",
    }
    return aliases.get(value, value or "neutral")


normalize_mood = _normalize_mood


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _table_has_rows(conn, table_name):
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
    return bool(row and row["count"] > 0)


def _migrate_simple_csv(conn, csv_path, table_name, columns, transform_row):
    if _table_has_rows(conn, table_name) or not os.path.exists(csv_path):
        return

    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return

    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    payload = []
    for row in rows:
        values = transform_row(row)
        if values:
            payload.append(values)

    if payload:
        conn.executemany(sql, payload)


def initialize_storage():
    global _STORAGE_READY
    if _STORAGE_READY and os.path.exists(DB_FILE):
        return

    os.makedirs(CURRENT_DIR, exist_ok=True)

    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS text_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                mood TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visual_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                mood TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS overall_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                overall_mood TEXT NOT NULL,
                overall_confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                subject_id TEXT NOT NULL DEFAULT '',
                input_len INTEGER NOT NULL DEFAULT 0,
                response_len INTEGER NOT NULL DEFAULT 0,
                meta TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracked_subjects (
                subject_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT '',
                first_seen TEXT NOT NULL DEFAULT '',
                last_seen TEXT NOT NULL DEFAULT '',
                logins INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                subject_id TEXT NOT NULL DEFAULT '',
                event_type TEXT NOT NULL,
                mood TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                detail TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_admin_events_subject ON {ADMIN_EVENTS_TABLE}(subject_id)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_activity_events_subject ON {ACTIVITY_EVENTS_TABLE}(subject_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weekly_overview (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_start TEXT NOT NULL,
                week_end TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                checkins INTEGER NOT NULL DEFAULT 0,
                avg_confidence REAL NOT NULL DEFAULT 0.0,
                dominant_mood TEXT NOT NULL DEFAULT 'none',
                heavy_count INTEGER NOT NULL DEFAULT 0,
                supportive_count INTEGER NOT NULL DEFAULT 0,
                saved_at TEXT NOT NULL
            )
            """
        )
        _migrate_simple_csv(
            conn,
            LEGACY_TEXT_FILE,
            "text_sessions",
            ["mood", "confidence", "timestamp"],
            lambda row: (
                _normalize_mood(row.get("mood") or row.get("Emotion")),
                max(0.0, min(_safe_float(row.get("confidence", 0.0), 0.0), 1.0)),
                str(row.get("timestamp") or row.get("DateTime") or "").strip(),
            ),
        )
        _migrate_simple_csv(
            conn,
            LEGACY_VISUAL_FILE,
            "visual_sessions",
            ["mood", "confidence", "timestamp"],
            lambda row: (
                _normalize_mood(row.get("mood") or row.get("Emotion")),
                max(0.0, min(_safe_float(row.get("confidence", 0.0), 0.0), 1.0)),
                str(row.get("timestamp") or row.get("DateTime") or "").strip(),
            ),
        )
        _migrate_simple_csv(
            conn,
            LEGACY_OVERALL_FILE,
            "overall_sessions",
            ["overall_mood", "overall_confidence", "severity", "timestamp"],
            lambda row: (
                _normalize_mood(row.get("overall_mood")),
                max(0.0, min(_safe_float(row.get("overall_confidence", 0.0), 0.0), 1.0)),
                str(row.get("severity", "low")).strip().lower() or "low",
                str(row.get("timestamp", "")).strip(),
            ),
        )
        _migrate_simple_csv(
            conn,
            LEGACY_WEEKLY_FILE,
            "weekly_overview",
            [
                "week_start",
                "week_end",
                "label",
                "checkins",
                "avg_confidence",
                "dominant_mood",
                "heavy_count",
                "supportive_count",
                "saved_at",
            ],
            lambda row: (
                str(row.get("week_start", "")).strip(),
                str(row.get("week_end", "")).strip(),
                str(row.get("label", "")).strip(),
                max(0, _safe_int(row.get("checkins", 0), 0)),
                max(0.0, min(_safe_float(row.get("avg_confidence", 0.0), 0.0), 1.0)),
                _normalize_mood(row.get("dominant_mood", "none")),
                max(0, _safe_int(row.get("heavy_count", 0), 0)),
                max(0, _safe_int(row.get("supportive_count", 0), 0)),
                str(row.get("saved_at", "")).strip(),
            ),
        )

        conn.commit()

    _STORAGE_READY = True


def _read_table_rows(table_name, columns, order_by, limit=None):
    initialize_storage()
    query = f"SELECT {', '.join(columns)} FROM {table_name} ORDER BY {order_by} DESC"
    params = []
    if limit:
        limit = max(1, int(limit))
        query += " LIMIT ?"
        params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    payload = [dict(row) for row in reversed(rows)]
    return payload


def read_legacy_rows(file_path, limit=None):
    key = str(file_path or "").strip().lower()
    table_name = LEGACY_FILE_TO_TABLE.get(key)
    if table_name == "text_sessions":
        return _read_table_rows(table_name, ["session_id", "mood", "confidence", "timestamp"], "session_id", limit=limit)
    if table_name == "visual_sessions":
        return _read_table_rows(table_name, ["session_id", "mood", "confidence", "timestamp"], "session_id", limit=limit)
    if table_name == "overall_sessions":
        return _read_table_rows(
            table_name,
            ["session_id", "overall_mood", "overall_confidence", "severity", "timestamp"],
            "session_id",
            limit=limit,
        )
    if table_name == "weekly_overview":
        return _read_table_rows(
            table_name,
            [
                "week_start",
                "week_end",
                "label",
                "checkins",
                "avg_confidence",
                "dominant_mood",
                "heavy_count",
                "supportive_count",
                "saved_at",
            ],
            "id",
            limit=limit,
        )
    return []


def _safe_text_len(value):
    try:
        return len(str(value or ""))
    except Exception:
        return 0


def log_response(response_type, input_text, response_text, source, meta=None, subject_id=""):
    initialize_storage()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_len = max(0, _safe_int(_safe_text_len(input_text), 0))
    response_len = max(0, _safe_int(_safe_text_len(response_text), 0))
    subject_id = str(subject_id or "").strip().lower()
    with _connect() as conn:
        conn.execute(
            f"""
            INSERT INTO {ADMIN_EVENTS_TABLE}
                (timestamp, type, source, subject_id, input_len, response_len, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                str(response_type or ""),
                str(source or ""),
                subject_id,
                input_len,
                response_len,
                str(meta or ""),
            ),
        )
        conn.commit()


def read_responses(limit=200):
    rows = _read_table_rows(
        ADMIN_EVENTS_TABLE,
        ["timestamp", "type", "source", "subject_id", "input_len", "response_len", "meta"],
        "id",
        limit=limit,
    )
    return [
        {
            "timestamp": row.get("timestamp", ""),
            "type": row.get("type", ""),
            "source": row.get("source", ""),
            "subject_id": row.get("subject_id", ""),
            "input_len": row.get("input_len", 0),
            "response_len": row.get("response_len", 0),
            "meta": row.get("meta", ""),
        }
        for row in rows
    ]


def upsert_user(subject_id, created_at=""):
    initialize_storage()
    subject_id = str(subject_id or "").strip().lower()
    if not subject_id:
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        row = conn.execute(
            f"SELECT subject_id, logins, first_seen, created_at FROM {TRACKED_SUBJECTS_TABLE} WHERE subject_id = ?",
            (subject_id,),
        ).fetchone()
        if row:
            conn.execute(
                f"""
                UPDATE {TRACKED_SUBJECTS_TABLE}
                SET created_at = ?, last_seen = ?, logins = ?
                WHERE subject_id = ?
                """,
                (
                    str(created_at or row["created_at"] or ""),
                    now,
                    max(1, _safe_int(row["logins"], 0) + 1),
                    subject_id,
                ),
            )
        else:
            conn.execute(
                f"""
                INSERT INTO {TRACKED_SUBJECTS_TABLE}
                    (subject_id, created_at, first_seen, last_seen, logins)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    subject_id,
                    str(created_at or ""),
                    now,
                    now,
                    1,
                ),
            )
        conn.commit()
    return True


def read_users(limit=500):
    return _read_table_rows(
        TRACKED_SUBJECTS_TABLE,
        ["subject_id", "created_at", "first_seen", "last_seen", "logins"],
        "last_seen",
        limit=limit,
    )


def _coerce_float_list(values):
    if values is None:
        return []

    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            return []

    if not isinstance(values, (list, tuple)):
        return []

    output = []
    for value in values:
        try:
            output.append(round(float(value), 8))
        except (TypeError, ValueError):
            continue
    return output


def save_face_profile(
    subject_id,
    template_vector,
    template_hist,
    sample_count,
    label="",
    match_threshold=0.0,
    enabled=True,
):
    initialize_storage()
    subject_id = str(subject_id or "").strip().lower()
    vector = _coerce_float_list(template_vector)
    hist = _coerce_float_list(template_hist)
    if not subject_id or not vector or not hist:
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bounded_threshold = max(0.0, min(_safe_float(match_threshold, 0.0), 1.0))
    bounded_count = max(1, _safe_int(sample_count, 1))

    with _FACE_PROFILE_LOCK:
        existing = _FACE_PROFILE_STORE.get(subject_id)
        created_at = str(existing.get("created_at") or now).strip() if existing else now
        _FACE_PROFILE_STORE[subject_id] = {
            "subject_id": subject_id,
            "label": str(label or "").strip(),
            "template_vector": vector,
            "template_hist": hist,
            "sample_count": bounded_count,
            "match_threshold": bounded_threshold,
            "enabled": bool(enabled),
            "created_at": created_at,
            "updated_at": now,
        }

    return True


def get_face_profile(subject_id):
    initialize_storage()
    subject_id = str(subject_id or "").strip().lower()
    if not subject_id:
        return None

    with _FACE_PROFILE_LOCK:
        profile = _FACE_PROFILE_STORE.get(subject_id)
        if not profile:
            return None
        return {
            "subject_id": profile.get("subject_id", ""),
            "label": profile.get("label", ""),
            "template_vector": _coerce_float_list(profile.get("template_vector")),
            "template_hist": _coerce_float_list(profile.get("template_hist")),
            "sample_count": max(0, _safe_int(profile.get("sample_count"), 0)),
            "match_threshold": max(0.0, min(_safe_float(profile.get("match_threshold"), 0.0), 1.0)),
            "enabled": bool(profile.get("enabled", True)),
            "created_at": str(profile.get("created_at") or "").strip(),
            "updated_at": str(profile.get("updated_at") or "").strip(),
        }


def read_face_profiles(limit=500):
    initialize_storage()
    if limit:
        limit = max(1, int(limit))
    with _FACE_PROFILE_LOCK:
        profiles = list(_FACE_PROFILE_STORE.values())

    profiles.sort(
        key=lambda row: (str(row.get("updated_at") or ""), str(row.get("subject_id") or "")),
        reverse=True,
    )
    if limit:
        profiles = profiles[:limit]

    return [
        {
            "subject_id": str(row.get("subject_id") or "").strip().lower(),
            "label": str(row.get("label") or "").strip(),
            "sample_count": max(0, _safe_int(row.get("sample_count"), 0)),
            "match_threshold": round(max(0.0, min(_safe_float(row.get("match_threshold"), 0.0), 1.0)), 4),
            "enabled": bool(row.get("enabled", True)),
            "created_at": str(row.get("created_at") or "").strip(),
            "updated_at": str(row.get("updated_at") or "").strip(),
        }
        for row in profiles
    ]


def set_face_profile_enabled(subject_id, enabled):
    initialize_storage()
    subject_id = str(subject_id or "").strip().lower()
    if not subject_id:
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _FACE_PROFILE_LOCK:
        profile = _FACE_PROFILE_STORE.get(subject_id)
        if not profile:
            return False
        profile["enabled"] = bool(enabled)
        profile["updated_at"] = now
        _FACE_PROFILE_STORE[subject_id] = profile
    return True


def delete_face_profile(subject_id):
    initialize_storage()
    subject_id = str(subject_id or "").strip().lower()
    if not subject_id:
        return False

    with _FACE_PROFILE_LOCK:
        return bool(_FACE_PROFILE_STORE.pop(subject_id, None))


def _safe_detail(value, max_len=160):
    text = str(value or "")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def log_activity(subject_id, event_type, mood="", confidence="", detail=""):
    initialize_storage()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute(
            f"""
            INSERT INTO {ACTIVITY_EVENTS_TABLE} (timestamp, subject_id, event_type, mood, confidence, detail)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                str(subject_id or "").strip().lower(),
                str(event_type or ""),
                _normalize_mood(mood),
                max(0.0, min(_safe_float(confidence, 0.0), 1.0)),
                _safe_detail(detail),
            ),
        )
        conn.commit()


def read_activity(limit=500):
    return _read_table_rows(
        ACTIVITY_EVENTS_TABLE,
        ["timestamp", "subject_id", "event_type", "mood", "confidence", "detail"],
        "id",
        limit=limit,
    )


def _next_session_id(table_name):
    initialize_storage()
    with _connect() as conn:
        row = conn.execute(f"SELECT COALESCE(MAX(session_id), 0) + 1 AS next_id FROM {table_name}").fetchone()
    return int(row["next_id"]) if row else 1


def get_next_session_id(file_path):
    key = str(file_path or "").strip().lower()
    table_name = LEGACY_FILE_TO_TABLE.get(key)
    if table_name in {"text_sessions", "visual_sessions", "overall_sessions"}:
        return _next_session_id(table_name)
    return 1


def save_text_session(mood, confidence):
    initialize_storage()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized = _normalize_mood(mood)
    bounded_confidence = max(0.0, min(_safe_float(confidence, 0.0), 1.0))

    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO text_sessions (mood, confidence, timestamp) VALUES (?, ?, ?)",
            (normalized, bounded_confidence, timestamp),
        )
        conn.commit()
        return int(cursor.lastrowid)


def save_visual_session(mood, confidence):
    initialize_storage()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    normalized = _normalize_mood(mood)
    bounded_confidence = max(0.0, min(_safe_float(confidence, 0.0), 1.0))

    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO visual_sessions (mood, confidence, timestamp) VALUES (?, ?, ?)",
            (normalized, bounded_confidence, timestamp),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_latest_session(file_path):
    key = str(file_path or "").strip().lower()
    table_name = LEGACY_FILE_TO_TABLE.get(key)
    if table_name not in {"text_sessions", "visual_sessions", "overall_sessions"}:
        return None

    columns = {
        "text_sessions": ["session_id", "mood", "confidence", "timestamp"],
        "visual_sessions": ["session_id", "mood", "confidence", "timestamp"],
        "overall_sessions": ["session_id", "overall_mood", "overall_confidence", "severity", "timestamp"],
    }[table_name]

    with _connect() as conn:
        row = conn.execute(
            f"SELECT {', '.join(columns)} FROM {table_name} ORDER BY session_id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def _fuse_emotions(text_result, visual_result):
    text_conf = max(0.0, min(_safe_float(text_result.get("confidence", 0.0), 0.0), 1.0))
    visual_conf = max(0.0, min(_safe_float(visual_result.get("confidence", 0.0), 0.0), 1.0))
    text_score = text_conf * TEXT_WEIGHT
    visual_score = visual_conf * VISUAL_WEIGHT
    overall_mood = (
        _normalize_mood(text_result.get("mood"))
        if text_score >= visual_score
        else _normalize_mood(visual_result.get("mood"))
    )
    overall_confidence = round(text_score + visual_score, 2)
    return overall_mood, overall_confidence


def _determine_severity(mood, confidence):
    normalized = _normalize_mood(mood)
    bounded_confidence = max(0.0, min(_safe_float(confidence, 0.0), 1.0))
    if normalized in {"sad", "anxious"} and bounded_confidence >= 0.8:
        return "high"
    if normalized in {"sad", "anxious"} and bounded_confidence >= 0.6:
        return "medium"
    return "low"


def save_overall_session():
    initialize_storage()
    text_result = get_latest_session(LEGACY_TEXT_FILE)
    visual_result = get_latest_session(LEGACY_VISUAL_FILE)
    if not text_result or not visual_result:
        return None

    overall_mood, overall_confidence = _fuse_emotions(text_result, visual_result)
    severity = _determine_severity(overall_mood, overall_confidence)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO overall_sessions (overall_mood, overall_confidence, severity, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (overall_mood, overall_confidence, severity, timestamp),
        )
        conn.commit()
        session_id = int(cursor.lastrowid)

    return {
        "session_id": session_id,
        "overall_mood": overall_mood,
        "overall_confidence": overall_confidence,
        "severity": severity,
    }


def get_last_n_overall_sessions(n=3):
    rows = _read_table_rows(
        "overall_sessions",
        ["session_id", "overall_mood", "overall_confidence", "severity", "timestamp"],
        "session_id",
        limit=n,
    )
    return rows


def get_overall_sessions():
    initialize_storage()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT overall_mood, overall_confidence, severity, timestamp
            FROM overall_sessions
            ORDER BY session_id ASC
            """
        ).fetchall()

    sessions = []
    for row in rows:
        timestamp = str(row["timestamp"] or "").strip()
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        sessions.append(
            {
                "timestamp": dt,
                "mood": _normalize_mood(row["overall_mood"]),
                "confidence": max(0.0, min(_safe_float(row["overall_confidence"], 0.0), 1.0)),
                "severity": str(row["severity"] or "low").strip().lower() or "low",
            }
        )

    return sessions


def save_weekly_snapshot(weekly_points):
    initialize_storage()
    saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute("DELETE FROM weekly_overview")
        conn.executemany(
            """
            INSERT INTO weekly_overview (
                week_start, week_end, label, checkins, avg_confidence,
                dominant_mood, heavy_count, supportive_count, saved_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(point.get("week_start", "")),
                    str(point.get("week_end", "")),
                    str(point.get("label", "")),
                    max(0, _safe_int(point.get("count", 0), 0)),
                    max(0.0, min(_safe_float(point.get("avg_confidence", 0.0), 0.0), 1.0)),
                    _normalize_mood(point.get("dominant_mood", "none")),
                    max(0, _safe_int(point.get("heavy_count", 0), 0)),
                    max(0, _safe_int(point.get("supportive_count", 0), 0)),
                    saved_at,
                )
                for point in weekly_points
            ],
        )
        conn.commit()
