"""SQLite logging — sessions, tokens, injections, phase transitions."""
import sqlite3
import json
import time
from contextlib import contextmanager
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
    trigger TEXT NOT NULL,        -- timed | stall | random
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
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_tokens_session ON tokens(session_id, step);
CREATE INDEX IF NOT EXISTS idx_injections_session ON injections(session_id);
"""


class DB:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.executescript(SCHEMA)

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

    def log_phase(self, session_id, step, from_phase, to_phase, pos):
        self.conn.execute(
            "INSERT INTO phase_transitions (session_id, step, ts, from_phase, to_phase, cycle_position) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, step, time.time(), from_phase, to_phase, pos),
        )

    def close(self):
        self.conn.close()
