import sqlite3

conn = sqlite3.connect("database/central.db")
cursor = conn.cursor()

# Users table with preset
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    preset TEXT DEFAULT 'normal'
)
""")

# Face table
cursor.execute("""
CREATE TABLE IF NOT EXISTS face (
    user_id INTEGER,
    face_embedding BLOB,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
)
""")

# Voice table
cursor.execute("""
CREATE TABLE IF NOT EXISTS voice (
    user_id INTEGER,
    voice_embedding BLOB,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
)
""")

# Gesture table
cursor.execute("""
CREATE TABLE IF NOT EXISTS gesture (
    user_id INTEGER,
    gesture_label TEXT,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
)
""")

# Indexing
cursor.execute("CREATE INDEX IF NOT EXISTS idx_gesture_user_id ON gesture(user_id)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_gesture_label ON gesture(gesture_label)")

conn.commit()
conn.close()
print("Central Database Created Successfully with Indexing")
