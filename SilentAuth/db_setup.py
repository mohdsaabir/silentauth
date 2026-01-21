import sqlite3

# Connect to central database
conn = sqlite3.connect("database/central.db")
cursor = conn.cursor()

# -------- USER TABLE --------
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
)
""")

# -------- FACE TABLE --------
cursor.execute("""
CREATE TABLE IF NOT EXISTS face (
    user_id INTEGER,
    face_embedding BLOB,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
)
""")

# -------- VOICE TABLE --------
cursor.execute("""
CREATE TABLE IF NOT EXISTS voice (
    user_id INTEGER,
    voice_embedding BLOB,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
)
""")

# -------- GESTURE TABLE --------
cursor.execute("""
CREATE TABLE IF NOT EXISTS gesture (
    user_id INTEGER,
    gesture_label TEXT,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
)
""")

# -------- INDEXING FOR GESTURE TABLE --------
cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_gesture_user_id
ON gesture(user_id)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_gesture_label
ON gesture(gesture_label)
""")

# Commit and close
conn.commit()
conn.close()

print(" Central Database Created Successfully with Gesture Indexing")