# interactive_chat_with_sqlite.py
import asyncio
import json
from datetime import datetime
from typing import Any, List, Optional

import aiosqlite

# -------------------------
# SQLite helper functions
# -------------------------
DB_PATH = "chat_history.db"
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

async def init_db(db_path: str = DB_PATH):
    async with aiosqlite.connect(db_path) as db:
        for p in DB_PRAGMAS:
            await db.execute(p)
        await db.execute(CREATE_TABLE_SQL)
        await db.commit()

async def _get_max_saved_index(db_path: str, session_id: str) -> Optional[int]:
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT MAX(event_index) FROM messages WHERE session_id = ?", (session_id,)
        )
        row = await cur.fetchone()
        await cur.close()
    if not row:
        return None
    return row[0] if row[0] is not None else -1

async def save_session_to_db(db_path: str, completed_session: Any):
    """
    Save new events from completed_session into DB, avoiding duplicates.
    Expects completed_session.session_id and completed_session.events (iterable).
    Each event is expected to have an index order (we use enumerate order).
    """
    session_id = getattr(completed_session, "session_id", None) or getattr(completed_session, "id", None) or "unknown_session"
    events = getattr(completed_session, "events", []) or []

    max_saved = await _get_max_saved_index(db_path, session_id)
    if max_saved is None:
        max_saved = -1

    insert_sql = """
        INSERT INTO messages (session_id, event_index, role, text, timestamp, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    """

    async with aiosqlite.connect(db_path) as db:
        # begin a transaction
        await db.execute("BEGIN")
        inserted = 0
        for idx, event in enumerate(events):
            # skip events already saved
            if idx <= max_saved:
                continue

            # robustly extract role/text
            try:
                role = event.content.role if event.content and hasattr(event.content, "role") else None
            except Exception:
                role = None
            try:
                text = (
                    event.content.parts[0].text
                    if event.content and getattr(event.content, "parts", None) and len(event.content.parts) > 0
                    else None
                )
            except Exception:
                text = None

            metadata = {}
            if hasattr(event, "timestamp"):
                metadata["event_timestamp"] = str(event.timestamp)
            timestamp = datetime.utcnow().isoformat() + "Z"

            await db.execute(insert_sql, (session_id, idx, role, text, timestamp, json.dumps(metadata)))
            inserted += 1

        await db.commit()
    return inserted

async def get_session_events(db_path: str, session_id: str):
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT event_index, role, text, timestamp, metadata FROM messages WHERE session_id = ? ORDER BY event_index ASC",
            (session_id,),
        )
        rows = await cur.fetchall()
        await cur.close()
    results = []
    for row in rows:
        idx, role, text, timestamp, metadata_json = row
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            metadata = {}
        results.append({"event_index": idx, "role": role, "text": text, "timestamp": timestamp, "metadata": metadata})
    return results

# -------------------------
# Interactive chat runner
# -------------------------

# NOTE:
# The following names must be defined in your environment (same as your prior code):
#   Runner, triage_doctor_finder_agent, session_service, memory_service, APP_NAME, USER_ID
# If they are in another module, import them accordingly.
# Example:
from agent import Runner, triage_doctor_finder_agent, session_service, memory_service, APP_NAME, USER_ID

async def interactive_chat_loop(
    runner,
    session_id: str,
    db_path: str = DB_PATH,
):
    """
    REPL-style chat:
    - reads user input without blocking the event loop
    - sends to runner.run_async(...)
    - prints final response
    - fetches session and saves only new events to sqlite
    """
    print("Interactive chat (type 'exit' or 'quit' to stop).")

    loop = asyncio.get_event_loop()
    while True:
        # get user input without blocking event loop
        user_text = await loop.run_in_executor(None, input, "\nYou: ")
        if user_text is None:
            continue
        user_text = user_text.strip()
        if user_text.lower() in ("exit", "quit"):
            print("Exiting chat.")
            break
        if user_text == "":
            continue

        # Build content object expected by your runner (reuse your Content/Part classes)
        # This assumes Content and Part are available in your environment.
        try:
            content = Content(parts=[Part(text=user_text)], role="user")
        except NameError:
            # Fallback simple object if Content/Part classes not available:
            class _P: 
                def __init__(self, text): self.text = text
            class _C:
                def __init__(self, parts, role): self.parts = parts; self.role = role
            content = _C(parts=[_P(user_text)], role="user")

        final_response_text = "(No final response)"
        # send message and consume stream
        try:
            async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=content):
                # print streaming chunks if you want; here we only capture final response
                if event.is_final_response() and getattr(event, "content", None) and getattr(event.content, "parts", None):
                    final_response_text = event.content.parts[0].text
        except Exception as ex:
            print(f"[Error while running runner.run_async] {ex}")
            continue

        print("\nAgent:", final_response_text)

        # fetch completed session and save only new events
        try:
            completed_session = await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        except Exception as ex:
            print(f"[Error fetching session] {ex}")
            continue

        try:
            inserted = await save_session_to_db(db_path, completed_session)
            if inserted:
                print(f"(Saved {inserted} new event(s) to {db_path})")
            else:
                print("(No new events to save)")
        except Exception as ex:
            print(f"[Error saving to DB] {ex}")

# -------------------------
# Entrypoint
# -------------------------
async def main():
    # --- initialize DB ---
    await init_db(DB_PATH)

    # --- create runner (adapt to your code) ---
    print("Initializing Runner...")
    # You must have Runner, triage_doctor_finder_agent, session_service, memory_service, APP_NAME, USER_ID
    # defined/imported in the same scope or imported here.
    try:
        runner = Runner(
            agent=triage_doctor_finder_agent,
            app_name=APP_NAME,
            session_service=session_service,
            memory_service=memory_service,
        )
    except NameError as e:
        raise RuntimeError(
            "Runner or required variables are not defined. Import/define Runner, triage_doctor_finder_agent, "
            "session_service, memory_service, APP_NAME, USER_ID before running this script."
        ) from e

    session_id = "chat001"
    # ensure session exists
    await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    # start interactive loop
    await interactive_chat_loop(runner, session_id, db_path=DB_PATH)

    # optional: print all rows saved for session when exiting
    saved = await get_session_events(DB_PATH, session_id)
    print("\nSaved session events (DB):")
    for r in saved:
        print(f"[{r['event_index']}] {r['role']}: {(r['text'] or '')[:200]} (saved at {r['timestamp']})")

if __name__ == "__main__":
    asyncio.run(main())
