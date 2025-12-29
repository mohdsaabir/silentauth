import sqlite3
import numpy as np
from pathlib import Path

# The DB logic is as same as we learned in PHP or other languages but here we use sqlite3 module

DB_PATH = Path("database/voice_embeddings.db")

def get_connection():
    """Create a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    return conn

def create_table():
    """Create the user table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL,
            keywords TEXT
        )
    """)
    conn.commit()
    conn.close()



def insert_embedding(user_name, embedding, keywords):
    """
    Store the embedding of a user.
    embedding: np.ndarray
    """
    conn = get_connection()
    cursor = conn.cursor()
    emb_bytes = embedding.astype(np.float32).tobytes()
    cursor.execute("""
        INSERT OR REPLACE INTO users (user_name, embedding, keywords)
        VALUES (?, ?, ?)
    """, (user_name, emb_bytes, keywords))
    conn.commit()
    conn.close()

# Have not used so far so skip this function
def fetch_embedding(user_name):
    """Fetch a single embedding for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT embedding FROM users WHERE user_name = ?", (user_name,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32)



def fetch_all_embeddings():
    """Fetch all users and their embeddings and return as a list of tuples of user_name and embedding."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_name, embedding , keywords FROM users")
    rows = cursor.fetchall()
    conn.close()
    result = []
    for user_name, emb_bytes, keywords in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        result.append((user_name, emb, keywords))
    return result
