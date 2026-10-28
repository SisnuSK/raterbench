"""On-disk response cache (SQLite). Re-running never re-pays for an identical call."""

import sqlite3
import time
from pathlib import Path

from raterbench.llm.types import LLMResponse


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> LLMResponse | None:
        row = self._conn.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return LLMResponse.model_validate_json(row[0]) if row else None

    def put(self, key: str, response: LLMResponse) -> None:
        # Commit per write so a crash loses at most the in-flight calls.
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, response, created_at) VALUES (?, ?, ?)",
            (key, response.model_dump_json(), time.time()),
        )
        self._conn.commit()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
