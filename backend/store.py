"""Persistence: patterns, version history (with branching), sessions, chat.

SQLite via the stdlib — no extra dependency, and the whole thing is one file you
can point at a Railway volume (`VLA_DB` env var) so history survives restarts.

Data model
----------
``patterns``  a named document (e.g. "Bodice sloper").
``versions``  an immutable snapshot of the pattern XML. ``parent_id`` makes this
              a **DAG, not a line** — saving from an older version branches, so
              you can explore variants without losing anything.
``sessions``  a working session opened against a version of a pattern.
``messages``  chat turns. ``parent_id`` threads them, so a session can **branch**
              from any earlier message: replaying a branch means walking parents
              from a leaf back to the root.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

DB_PATH = os.getenv("VLA_DB", str(Path(__file__).resolve().parent.parent / "data" / "vla.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patterns (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS versions (
    id            TEXT PRIMARY KEY,
    pattern_id    TEXT NOT NULL REFERENCES patterns(id),
    parent_id     TEXT REFERENCES versions(id),   -- NULL = root; branching DAG
    label         TEXT NOT NULL DEFAULT '',
    xml           TEXT NOT NULL,
    measurements  TEXT NOT NULL DEFAULT '',       -- filename of the measurement table
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_versions_pattern ON versions(pattern_id, created_at);
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    pattern_id  TEXT REFERENCES patterns(id),
    version_id  TEXT REFERENCES versions(id),
    title       TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT '',      -- per-session model choice
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    parent_id   TEXT REFERENCES messages(id),     -- threading -> branching
    role        TEXT NOT NULL,                    -- user | assistant | note
    text        TEXT NOT NULL,
    tool_calls  TEXT NOT NULL DEFAULT '[]',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id, created_at);
"""


def _now() -> float:
    return time.time()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations so an existing database keeps working."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
    if "model" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN model TEXT NOT NULL DEFAULT ''")
        conn.commit()


class Store:
    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or connect()

    # --- patterns ------------------------------------------------------------
    def create_pattern(self, name: str) -> str:
        pid = _id("pat")
        t = _now()
        self.conn.execute("INSERT INTO patterns(id,name,created_at,updated_at) VALUES(?,?,?,?)",
                          (pid, name, t, t))
        self.conn.commit()
        return pid

    def rename_pattern(self, pattern_id: str, name: str) -> None:
        self.conn.execute("UPDATE patterns SET name=?, updated_at=? WHERE id=?",
                          (name, _now(), pattern_id))
        self.conn.commit()

    def list_patterns(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT p.*, (SELECT COUNT(*) FROM versions v WHERE v.pattern_id=p.id) AS versions
               FROM patterns p ORDER BY p.updated_at DESC""").fetchall()
        return [dict(r) for r in rows]

    def get_pattern(self, pattern_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM patterns WHERE id=?", (pattern_id,)).fetchone()
        return dict(r) if r else None

    def delete_pattern(self, pattern_id: str) -> None:
        self.conn.execute(
            "DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE pattern_id=?)",
            (pattern_id,))
        self.conn.execute("DELETE FROM sessions WHERE pattern_id=?", (pattern_id,))
        self.conn.execute("DELETE FROM versions WHERE pattern_id=?", (pattern_id,))
        self.conn.execute("DELETE FROM patterns WHERE id=?", (pattern_id,))
        self.conn.commit()

    # --- versions ------------------------------------------------------------
    def save_version(self, pattern_id: str, xml: str, *, parent_id: str | None = None,
                     label: str = "", measurements: str = "") -> str:
        vid = _id("ver")
        self.conn.execute(
            """INSERT INTO versions(id,pattern_id,parent_id,label,xml,measurements,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (vid, pattern_id, parent_id, label, xml, measurements, _now()))
        self.conn.execute("UPDATE patterns SET updated_at=? WHERE id=?", (_now(), pattern_id))
        self.conn.commit()
        return vid

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM versions WHERE id=?", (version_id,)).fetchone()
        return dict(r) if r else None

    def list_versions(self, pattern_id: str) -> list[dict[str, Any]]:
        """Version metadata (no XML payload) oldest-first, for a history view."""
        rows = self.conn.execute(
            """SELECT id,pattern_id,parent_id,label,measurements,created_at,
                      LENGTH(xml) AS size
               FROM versions WHERE pattern_id=? ORDER BY created_at""",
            (pattern_id,)).fetchall()
        return [dict(r) for r in rows]

    def latest_version(self, pattern_id: str) -> dict[str, Any] | None:
        r = self.conn.execute(
            "SELECT * FROM versions WHERE pattern_id=? ORDER BY created_at DESC LIMIT 1",
            (pattern_id,)).fetchone()
        return dict(r) if r else None

    # --- sessions ------------------------------------------------------------
    def create_session(self, pattern_id: str | None, version_id: str | None,
                       title: str = "", model: str = "") -> str:
        sid = _id("ses")
        t = _now()
        self.conn.execute(
            "INSERT INTO sessions(id,pattern_id,version_id,title,model,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?)", (sid, pattern_id, version_id, title, model, t, t))
        self.conn.commit()
        return sid

    def set_session_model(self, session_id: str, model: str) -> None:
        self.conn.execute("UPDATE sessions SET model=?, updated_at=? WHERE id=?",
                          (model, _now(), session_id))
        self.conn.commit()

    def touch_session(self, session_id: str, version_id: str | None = None) -> None:
        if version_id:
            self.conn.execute("UPDATE sessions SET updated_at=?, version_id=? WHERE id=?",
                              (_now(), version_id, session_id))
        else:
            self.conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (_now(), session_id))
        self.conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(r) if r else None

    def list_sessions(self, pattern_id: str | None = None) -> list[dict[str, Any]]:
        if pattern_id:
            rows = self.conn.execute(
                """SELECT s.*, p.name AS pattern_name FROM sessions s
                   LEFT JOIN patterns p ON p.id=s.pattern_id
                   WHERE s.pattern_id=? ORDER BY s.updated_at DESC""", (pattern_id,)).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT s.*, p.name AS pattern_name FROM sessions s
                   LEFT JOIN patterns p ON p.id=s.pattern_id
                   ORDER BY s.updated_at DESC LIMIT 50""").fetchall()
        return [dict(r) for r in rows]

    # --- messages (threaded => branching) -----------------------------------
    def add_message(self, session_id: str, role: str, text: str, *,
                    parent_id: str | None = None, tool_calls: list | None = None) -> str:
        mid = _id("msg")
        self.conn.execute(
            """INSERT INTO messages(id,session_id,parent_id,role,text,tool_calls,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (mid, session_id, parent_id, role, text, json.dumps(tool_calls or []), _now()))
        self.conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (_now(), session_id))
        self.conn.commit()
        return mid

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY created_at", (session_id,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["tool_calls"] = json.loads(d["tool_calls"] or "[]")
            out.append(d)
        return out

    def message_chain(self, message_id: str) -> list[dict[str, Any]]:
        """The thread from the root down to ``message_id`` (a chat branch)."""
        chain: list[dict[str, Any]] = []
        cur: str | None = message_id
        seen: set[str] = set()
        while cur and cur not in seen:
            seen.add(cur)
            r = self.conn.execute("SELECT * FROM messages WHERE id=?", (cur,)).fetchone()
            if r is None:
                break
            d = dict(r)
            d["tool_calls"] = json.loads(d["tool_calls"] or "[]")
            chain.append(d)
            cur = d["parent_id"]
        return list(reversed(chain))

    def last_message_id(self, session_id: str) -> str | None:
        r = self.conn.execute(
            "SELECT id FROM messages WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
            (session_id,)).fetchone()
        return r["id"] if r else None
