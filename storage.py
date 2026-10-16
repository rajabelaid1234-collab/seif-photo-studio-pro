from __future__ import annotations

import os
import sqlite3
import time
import uuid
from pathlib import Path

DATA_DIR = Path(os.environ.get("SEIF_DATA_DIR", "data"))
RETENTION_DAYS = max(1, int(os.environ.get("SEIF_RETENTION_DAYS", "30")))
DAY = 86400


def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "files").mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATA_DIR / "library.db", timeout=15)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS items(
            id TEXT PRIMARY KEY, owner TEXT NOT NULL, filename TEXT NOT NULL,
            fmt TEXT NOT NULL, kind TEXT NOT NULL, created REAL NOT NULL,
            expires REAL NOT NULL, path TEXT NOT NULL)"""
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_owner ON items(owner)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_expires ON items(expires)")
    return connection


def save_item(owner: str, filename: str, data: bytes, fmt: str = "PNG",
              kind: str = "result") -> str:
    if not owner:
        raise ValueError("owner is required")
    item_id = uuid.uuid4().hex
    path = DATA_DIR / "files" / f"{item_id}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    now = time.time()
    try:
        with _conn() as connection:
            connection.execute(
                "INSERT INTO items VALUES(?,?,?,?,?,?,?,?)",
                (item_id, owner, filename, fmt, kind, now,
                 now + RETENTION_DAYS * DAY, str(path)),
            )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return item_id


def list_items(owner: str, kind: str | None = None):
    query = "SELECT id, filename, fmt, kind, created, expires FROM items WHERE owner=?"
    args = [owner]
    if kind:
        query += " AND kind=?"
        args.append(kind)
    query += " ORDER BY created DESC"
    with _conn() as connection:
        return connection.execute(query, args).fetchall()


def get_bytes(owner: str, item_id: str) -> bytes | None:
    with _conn() as connection:
        row = connection.execute(
            "SELECT path FROM items WHERE id=? AND owner=?", (item_id, owner)
        ).fetchone()
    if not row:
        return None
    path = Path(row[0])
    return path.read_bytes() if path.is_file() else None


def delete_item(owner: str, item_id: str) -> None:
    with _conn() as connection:
        row = connection.execute(
            "SELECT path FROM items WHERE id=? AND owner=?", (item_id, owner)
        ).fetchone()
        if row:
            Path(row[0]).unlink(missing_ok=True)
            connection.execute("DELETE FROM items WHERE id=? AND owner=?", (item_id, owner))


def delete_all(owner: str) -> int:
    rows = list_items(owner)
    for row in rows:
        delete_item(owner, row[0])
    return len(rows)


def cleanup_expired() -> int:
    """Delete only items whose retention period has elapsed."""
    now = time.time()
    removed = 0
    with _conn() as connection:
        rows = connection.execute(
            "SELECT id, path FROM items WHERE expires <= ?", (now,)
        ).fetchall()
        for item_id, filename in rows:
            try:
                Path(filename).unlink(missing_ok=True)
                connection.execute("DELETE FROM items WHERE id=?", (item_id,))
                removed += 1
            except OSError:
                # Keep metadata so the item can be retried during the next cleanup.
                pass
    return removed
