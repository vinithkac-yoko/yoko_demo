"""Persistence: patterns, version history (with branching), sessions, runs.

SQLite via the stdlib — no extra dependency, and the whole thing is one file you
can point at a Railway volume (`VLA_DB` env var) so history survives restarts.

Data model
----------
``patterns``  a named document (e.g. "Bodice sloper").
``versions``  an immutable snapshot of the pattern XML. ``parent_id`` makes this
              a **DAG, not a line** — saving from an older version branches, so
              you can explore variants without losing anything.
``sessions``  a working session opened against a version of a pattern (the
              model picker's choice lives here).
``runs``      one row per typed instruction. It names the version the pattern
              was in when it started (``version_before_id``) and the version it
              produced on success (``version_after_id``) — branching happens
              here, not in a chat structure: "branch from this run" just means
              opening a new session on its ``version_after_id``.
``actions``   the individual tool calls one run made, in order — the action
              log the UI shows. Deliberately thin: no per-action state
              snapshot is stored. A before/after view is derived on demand by
              replaying a run's actions against its starting version (see
              ``replay_run`` in app.py) rather than duplicating pattern state
              in the database once per tool call.
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
    version_id  TEXT REFERENCES versions(id),     -- the session's current version
    title       TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT '',          -- per-session model choice
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id                 TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL REFERENCES sessions(id),
    instruction        TEXT NOT NULL,
    model              TEXT NOT NULL DEFAULT '',
    final_text         TEXT NOT NULL DEFAULT '',
    stopped_reason     TEXT NOT NULL DEFAULT '',
    version_before_id  TEXT REFERENCES versions(id),
    version_after_id   TEXT REFERENCES versions(id),   -- NULL until the run saves
    created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_runs_session ON runs(session_id, created_at);
CREATE TABLE IF NOT EXISTS actions (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(id),
    step_index   INTEGER NOT NULL,
    tool_name    TEXT NOT NULL,
    tool_input   TEXT NOT NULL DEFAULT '{}',
    reasoning    TEXT NOT NULL DEFAULT '',
    ok           INTEGER NOT NULL,
    message      TEXT NOT NULL DEFAULT '',
    blocked_by   TEXT NOT NULL DEFAULT '[]',
    touched_ids  TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS ix_actions_run ON actions(run_id, step_index);
CREATE TABLE IF NOT EXISTS presets (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    config      TEXT NOT NULL DEFAULT '{}',   -- the full RunConfig as JSON
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
"""

# Columns added after the first release. SQLite has no "ADD COLUMN IF NOT
# EXISTS", and the deployed database already holds real runs, so they're
# applied conditionally rather than by rewriting _SCHEMA.
_MIGRATIONS: list[tuple[str, str, str]] = [
    # (table, column, DDL)
    ("runs", "config", "ALTER TABLE runs ADD COLUMN config TEXT NOT NULL DEFAULT '{}'"),
    ("runs", "preset_name", "ALTER TABLE runs ADD COLUMN preset_name TEXT NOT NULL DEFAULT ''"),
    # -1 down, 0 unrated, 1 up — the tuning signal for collecting good samples.
    ("runs", "rating", "ALTER TABLE runs ADD COLUMN rating INTEGER NOT NULL DEFAULT 0"),
    ("runs", "note", "ALTER TABLE runs ADD COLUMN note TEXT NOT NULL DEFAULT ''"),
    ("sessions", "config", "ALTER TABLE sessions ADD COLUMN config TEXT NOT NULL DEFAULT '{}'"),
    (
        "sessions",
        "preset_name",
        "ALTER TABLE sessions ADD COLUMN preset_name TEXT NOT NULL DEFAULT ''",
    ),
]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, ddl in _MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(ddl)
    conn.commit()


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


class Store:
    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or connect()

    # --- patterns ------------------------------------------------------------
    def create_pattern(self, name: str) -> str:
        pid = _id("pat")
        t = _now()
        self.conn.execute(
            "INSERT INTO patterns(id,name,created_at,updated_at) VALUES(?,?,?,?)", (pid, name, t, t)
        )
        self.conn.commit()
        return pid

    def rename_pattern(self, pattern_id: str, name: str) -> None:
        self.conn.execute(
            "UPDATE patterns SET name=?, updated_at=? WHERE id=?", (name, _now(), pattern_id)
        )
        self.conn.commit()

    def list_patterns(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT p.*, (SELECT COUNT(*) FROM versions v WHERE v.pattern_id=p.id) AS versions
               FROM patterns p ORDER BY p.updated_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pattern(self, pattern_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM patterns WHERE id=?", (pattern_id,)).fetchone()
        return dict(r) if r else None

    def delete_pattern(self, pattern_id: str) -> None:
        self.conn.execute(
            """DELETE FROM actions WHERE run_id IN (
                   SELECT id FROM runs WHERE session_id IN (
                       SELECT id FROM sessions WHERE pattern_id=?))""",
            (pattern_id,),
        )
        self.conn.execute(
            "DELETE FROM runs WHERE session_id IN (SELECT id FROM sessions WHERE pattern_id=?)",
            (pattern_id,),
        )
        self.conn.execute("DELETE FROM sessions WHERE pattern_id=?", (pattern_id,))
        self.conn.execute("DELETE FROM versions WHERE pattern_id=?", (pattern_id,))
        self.conn.execute("DELETE FROM patterns WHERE id=?", (pattern_id,))
        self.conn.commit()

    # --- versions --------------------------------------------------------------
    def save_version(
        self,
        pattern_id: str,
        xml: str,
        *,
        parent_id: str | None = None,
        label: str = "",
        measurements: str = "",
    ) -> str:
        vid = _id("ver")
        self.conn.execute(
            """INSERT INTO versions(id,pattern_id,parent_id,label,xml,measurements,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (vid, pattern_id, parent_id, label, xml, measurements, _now()),
        )
        self.conn.execute("UPDATE patterns SET updated_at=? WHERE id=?", (_now(), pattern_id))
        self.conn.commit()
        return vid

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM versions WHERE id=?", (version_id,)).fetchone()
        return dict(r) if r else None

    def list_versions(self, pattern_id: str) -> list[dict[str, Any]]:
        """Version metadata (no XML payload) oldest-first, for a history view."""
        rows = self.conn.execute(
            """SELECT id,pattern_id,parent_id,label,measurements,created_at, LENGTH(xml) AS size
               FROM versions WHERE pattern_id=? ORDER BY created_at""",
            (pattern_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def latest_version(self, pattern_id: str) -> dict[str, Any] | None:
        r = self.conn.execute(
            "SELECT * FROM versions WHERE pattern_id=? ORDER BY created_at DESC LIMIT 1",
            (pattern_id,),
        ).fetchone()
        return dict(r) if r else None

    # --- sessions ----------------------------------------------------------
    def create_session(
        self, pattern_id: str | None, version_id: str | None, title: str = "", model: str = ""
    ) -> str:
        sid = _id("ses")
        t = _now()
        self.conn.execute(
            "INSERT INTO sessions(id,pattern_id,version_id,title,model,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
            (sid, pattern_id, version_id, title, model, t, t),
        )
        self.conn.commit()
        return sid

    def set_session_model(self, session_id: str, model: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET model=?, updated_at=? WHERE id=?", (model, _now(), session_id)
        )
        self.conn.commit()

    def touch_session(self, session_id: str, version_id: str | None = None) -> None:
        if version_id:
            self.conn.execute(
                "UPDATE sessions SET updated_at=?, version_id=? WHERE id=?",
                (_now(), version_id, session_id),
            )
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
                   WHERE s.pattern_id=? ORDER BY s.updated_at DESC""",
                (pattern_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT s.*, p.name AS pattern_name FROM sessions s
                   LEFT JOIN patterns p ON p.id=s.pattern_id
                   ORDER BY s.updated_at DESC LIMIT 50"""
            ).fetchall()
        return [dict(r) for r in rows]

    # --- runs + actions (the instruction/action log) --------------------------
    def create_run(
        self,
        session_id: str,
        instruction: str,
        model: str,
        version_before_id: str | None,
        *,
        config: dict | None = None,
        preset_name: str = "",
    ) -> str:
        """Open a run. ``config`` is the fully-resolved RunConfig used for it —
        stored verbatim so a result stays traceable to the exact prompt and
        parameters that produced it, even after the preset is edited."""
        rid = _id("run")
        self.conn.execute(
            """INSERT INTO runs(id,session_id,instruction,model,version_before_id,created_at,
                                config,preset_name)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                rid,
                session_id,
                instruction,
                model,
                version_before_id,
                _now(),
                json.dumps(config or {}),
                preset_name,
            ),
        )
        self.conn.commit()
        return rid

    def rate_run(self, run_id: str, rating: int, note: str = "") -> None:
        """-1 down, 0 unrated, 1 up."""
        self.conn.execute(
            "UPDATE runs SET rating=?, note=? WHERE id=?", (int(rating), note, run_id)
        )
        self.conn.commit()

    def finish_run(
        self, run_id: str, *, final_text: str, stopped_reason: str, version_after_id: str | None
    ) -> None:
        self.conn.execute(
            "UPDATE runs SET final_text=?, stopped_reason=?, version_after_id=? WHERE id=?",
            (final_text, stopped_reason, version_after_id, run_id),
        )
        self.conn.commit()

    def add_action(
        self,
        run_id: str,
        step_index: int,
        tool_name: str,
        tool_input: dict,
        *,
        reasoning: str = "",
        ok: bool,
        message: str = "",
        blocked_by: list[int] | None = None,
        touched_ids: list[int] | None = None,
    ) -> str:
        aid = _id("act")
        self.conn.execute(
            """INSERT INTO actions(id,run_id,step_index,tool_name,tool_input,reasoning,ok,message,
                                   blocked_by,touched_ids)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                aid,
                run_id,
                step_index,
                tool_name,
                json.dumps(tool_input),
                reasoning,
                int(ok),
                message,
                json.dumps(blocked_by or []),
                json.dumps(touched_ids or []),
            ),
        )
        self.conn.commit()
        return aid

    @staticmethod
    def _run_row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        d["config"] = json.loads(d.get("config") or "{}")
        return d

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._run_row(r) if r else None

    def list_runs(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM runs WHERE session_id=? ORDER BY created_at", (session_id,)
        ).fetchall()
        return [self._run_row(r) for r in rows]

    def list_rated_runs(self, rating: int | None = None) -> list[dict[str, Any]]:
        """Runs carrying a rating, newest first — the corpus for sample export.
        ``rating=None`` returns everything rated either way."""
        if rating is None:
            rows = self.conn.execute(
                "SELECT * FROM runs WHERE rating != 0 ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM runs WHERE rating=? ORDER BY created_at DESC", (int(rating),)
            ).fetchall()
        return [self._run_row(r) for r in rows]

    # --- presets (saved prompt/parameter configs) -----------------------------
    def create_preset(self, name: str, config: dict) -> str:
        pid = _id("pre")
        t = _now()
        self.conn.execute(
            "INSERT INTO presets(id,name,config,created_at,updated_at) VALUES(?,?,?,?,?)",
            (pid, name, json.dumps(config), t, t),
        )
        self.conn.commit()
        return pid

    def update_preset(self, preset_id: str, *, name: str | None = None, config: dict | None = None):
        if name is not None:
            self.conn.execute(
                "UPDATE presets SET name=?, updated_at=? WHERE id=?", (name, _now(), preset_id)
            )
        if config is not None:
            self.conn.execute(
                "UPDATE presets SET config=?, updated_at=? WHERE id=?",
                (json.dumps(config), _now(), preset_id),
            )
        self.conn.commit()

    def get_preset(self, preset_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM presets WHERE id=?", (preset_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["config"] = json.loads(d["config"])
        return d

    def list_presets(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM presets ORDER BY updated_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d["config"])
            out.append(d)
        return out

    def delete_preset(self, preset_id: str) -> None:
        self.conn.execute("DELETE FROM presets WHERE id=?", (preset_id,))
        self.conn.commit()

    # --- the session's working config (unsaved edits live here) ---------------
    def set_session_config(self, session_id: str, config: dict, preset_name: str = "") -> None:
        self.conn.execute(
            "UPDATE sessions SET config=?, preset_name=?, updated_at=? WHERE id=?",
            (json.dumps(config), preset_name, _now(), session_id),
        )
        self.conn.commit()

    def get_session_config(self, session_id: str) -> dict:
        r = self.conn.execute("SELECT config FROM sessions WHERE id=?", (session_id,)).fetchone()
        return json.loads(r["config"]) if r and r["config"] else {}

    def list_actions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM actions WHERE run_id=? ORDER BY step_index", (run_id,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["ok"] = bool(d["ok"])
            d["tool_input"] = json.loads(d["tool_input"])
            d["blocked_by"] = json.loads(d["blocked_by"])
            d["touched_ids"] = json.loads(d["touched_ids"])
            out.append(d)
        return out
