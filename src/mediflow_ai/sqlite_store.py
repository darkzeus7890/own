# sqlite_store.py
import aiosqlite
import json
from datetime import datetime
from typing import Any

# ================= CREATING SQL TABLE =================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_index INTEGER NOT NULL,
    role TEXT,
    text TEXT,
    timestamp TEXT,
    metadata TEXT
);
"""

DB_PRAGMAS = [
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
]

# ================= INITIALIZNG DATABASE =================
async def init_db(db_path: str):
    """Create DB file and messages table if not exists."""
    async with aiosqlite.connect(db_path) as db:
        for p in DB_PRAGMAS:
            await db.execute(p)
        await db.execute(CREATE_TABLE_SQL)
        await db.commit()

# ================= SAVING SESSION TO DB =================
async def save_session_to_db(db_path: str, completed_session: Any):
    """
    Save all events in 'completed_session' to messages table.
    completed_session must have: session_id and events (iterable),
    and each event should have event.content.role and event.content.parts[0].text
    (as in your example).
    """
    session_id = getattr(completed_session, "session_id", None) or getattr(completed_session, "id", None) or "unknown_session"
    events = getattr(completed_session, "events", []) or []

    async with aiosqlite.connect(db_path) as db:
        insert_sql = """
        INSERT INTO messages (session_id, event_index, role, text, timestamp, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        async with db.execute("BEGIN"):
            for idx, event in enumerate(events):
                try:
                    role = event.content.role if event.content and hasattr(event.content, "role") else None
                except Exception:
                    role = None
                try:
                    text = (
                        event.content.parts[0].text
                        if event.content and event.content.parts and len(event.content.parts) > 0
                        else None
                    )
                except Exception:
                    text = None

                # metadata: store any other useful attributes in json (safe fallback)
                metadata = {}
                # try to put a timestamp if the event has one
                if hasattr(event, "timestamp"):
                    metadata["event_timestamp"] = str(event.timestamp)
                # optionally you could add other attributes here

                timestamp = datetime.utcnow().isoformat() + "Z"

                await db.execute(insert_sql, (session_id, idx, role, text, timestamp, json.dumps(metadata)))
        await db.commit()


# ================= RETRIEVING SAVED EVENTS  =================
async def get_session_events(db_path: str, session_id: str):
    """Retrieve saved events for a session_id ordered by event_index."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT event_index, role, text, timestamp, metadata FROM messages WHERE session_id = ? ORDER BY event_index ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
    # Return as list of dicts
    results = []
    for row in rows:
        idx, role, text, timestamp, metadata_json = row
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            metadata = {}
        results.append({"event_index": idx, "role": role, "text": text, "timestamp": timestamp, "metadata": metadata})
    return results
