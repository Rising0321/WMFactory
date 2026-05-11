from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "wmarena.sqlite3"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS arena_votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                arena_round_id TEXT NOT NULL,
                client_id TEXT,
                client_ip TEXT,
                user_agent TEXT,
                vote_option TEXT NOT NULL,
                anonymous_mode INTEGER NOT NULL,
                left_model_id TEXT NOT NULL,
                right_model_id TEXT NOT NULL,
                left_session_id TEXT,
                right_session_id TEXT,
                left_visible_devices TEXT,
                right_visible_devices TEXT,
                extra_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def insert_vote(record: Dict[str, Any]) -> int:
    payload = dict(record)
    extra_json = json.dumps(payload.pop("extra", {}), ensure_ascii=False, sort_keys=True)
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO arena_votes (
                created_at,
                arena_round_id,
                client_id,
                client_ip,
                user_agent,
                vote_option,
                anonymous_mode,
                left_model_id,
                right_model_id,
                left_session_id,
                right_session_id,
                left_visible_devices,
                right_visible_devices,
                extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["created_at"],
                payload["arena_round_id"],
                payload.get("client_id"),
                payload.get("client_ip"),
                payload.get("user_agent"),
                payload["vote_option"],
                1 if payload.get("anonymous_mode") else 0,
                payload["left_model_id"],
                payload["right_model_id"],
                payload.get("left_session_id"),
                payload.get("right_session_id"),
                payload.get("left_visible_devices"),
                payload.get("right_visible_devices"),
                extra_json,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
