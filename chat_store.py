import sqlite3
from datetime import datetime

DB_PATH = "chat_history.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        """
    )
    conn.commit()
    conn.close()


def create_conversation(name=None):
    conn = _connect()
    now = datetime.utcnow().isoformat()
    if not name:
        count = conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
        name = f"Conversation {count + 1}"
    cur = conn.execute(
        "INSERT INTO conversations (name, created_at, updated_at) VALUES (?, ?, ?)",
        (name, now, now),
    )
    conn.commit()
    conv_id = cur.lastrowid
    conn.close()
    return conv_id


def list_conversations():
    """Most recently active conversations first."""
    conn = _connect()
    rows = conn.execute(
        "SELECT id, name, updated_at FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def rename_conversation(conversation_id, new_name):
    conn = _connect()
    conn.execute(
        "UPDATE conversations SET name = ? WHERE id = ?",
        (new_name, conversation_id),
    )
    conn.commit()
    conn.close()


def get_conversation_name(conversation_id):
    conn = _connect()
    row = conn.execute(
        "SELECT name FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    conn.close()
    return row["name"] if row else None


def delete_conversation(conversation_id):
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()


def get_messages(conversation_id):
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def add_message(conversation_id, role, content):
    conn = _connect()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, now),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    conn.commit()
    conn.close()


def clear_conversation_messages(conversation_id):
    """Empties a conversation's messages but keeps the conversation itself
    (its name/id persist so it stays selectable in the switcher)."""
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.commit()
    conn.close()
