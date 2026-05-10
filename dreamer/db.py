"""SQLite logging — sessions, tokens, injections, phase transitions,
contamination events, self-states."""
import sqlite3
import json
import time
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL,
    model TEXT NOT NULL,
    perspective TEXT NOT NULL,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    ts REAL NOT NULL,
    temperature REAL NOT NULL,
    phase TEXT NOT NULL,
    token TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS injections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    ts REAL NOT NULL,
    phase TEXT NOT NULL,
    source TEXT NOT NULL,        -- day | world | latent
    trigger TEXT NOT NULL,        -- timed | stall | random | recovery
    fragment TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS phase_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    ts REAL NOT NULL,
    from_phase TEXT,
    to_phase TEXT NOT NULL,
    cycle_position REAL NOT NULL,
    window_tokens INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS contamination_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    ts REAL NOT NULL,
    phase TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'register',   -- 'register' | 'topical' | 'stickiness'
    pattern TEXT NOT NULL,
    snippet TEXT NOT NULL,
    action TEXT NOT NULL,             -- 'logged' | 'recovered'
    truncated_chars INTEGER,
    recovery_fragment TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS self_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    ts REAL NOT NULL,
    phase TEXT NOT NULL,
    summary TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_tokens_session ON tokens(session_id, step);
CREATE INDEX IF NOT EXISTS idx_injections_session ON injections(session_id);
CREATE INDEX IF NOT EXISTS idx_contamination_session ON contamination_events(session_id);
CREATE INDEX IF NOT EXISTS idx_self_states_session ON self_states(session_id);
"""


# Idempotent column additions for DBs created before these columns existed.
# `ALTER TABLE ... ADD COLUMN` raises OperationalError if the column already
# exists, which we swallow.
_MIGRATIONS = [
    "ALTER TABLE phase_transitions ADD COLUMN window_tokens INTEGER",
    "ALTER TABLE contamination_events ADD COLUMN kind TEXT NOT NULL DEFAULT 'register'",
]


class DB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.executescript(SCHEMA)
        for stmt in _MIGRATIONS:
            try:
                self.conn.execute(stmt)
            except sqlite3.OperationalError:
                pass

    def start_session(self, model: str, perspective: str, config: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (started_at, model, perspective, config_json) VALUES (?, ?, ?, ?)",
            (time.time(), model, perspective, json.dumps(config)),
        )
        return cur.lastrowid

    def end_session(self, session_id: int):
        self.conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (time.time(), session_id),
        )

    def log_token(self, session_id, step, temp, phase, token):
        self.conn.execute(
            "INSERT INTO tokens (session_id, step, ts, temperature, phase, token) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, step, time.time(), temp, phase, token),
        )

    def log_injection(self, session_id, step, phase, source, trigger, fragment):
        self.conn.execute(
            "INSERT INTO injections (session_id, step, ts, phase, source, trigger, fragment) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, step, time.time(), phase, source, trigger, fragment),
        )

    def log_phase(self, session_id, step, from_phase, to_phase, pos, window_tokens=None):
        self.conn.execute(
            "INSERT INTO phase_transitions (session_id, step, ts, from_phase, to_phase, cycle_position, window_tokens) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, step, time.time(), from_phase, to_phase, pos, window_tokens),
        )

    def log_contamination(
        self,
        session_id,
        step,
        phase,
        pattern,
        snippet,
        action,
        kind="register",
        truncated_chars=None,
        recovery_fragment=None,
    ):
        self.conn.execute(
            "INSERT INTO contamination_events (session_id, step, ts, phase, kind, pattern, snippet, action, truncated_chars, recovery_fragment) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                step,
                time.time(),
                phase,
                kind,
                pattern,
                snippet,
                action,
                truncated_chars,
                recovery_fragment,
            ),
        )

    def log_self_state(self, session_id, step, phase, summary):
        self.conn.execute(
            "INSERT INTO self_states (session_id, step, ts, phase, summary) VALUES (?, ?, ?, ?, ?)",
            (session_id, step, time.time(), phase, summary),
        )

    def fetch_session_transcript(self, session_id: int) -> str:
        cur = self.conn.execute(
            "SELECT token FROM tokens WHERE session_id = ? ORDER BY step, id",
            (session_id,),
        )
        return "".join(row[0] for row in cur)

    def close(self):
        self.conn.close()
