import sqlite3

conn = sqlite3.connect('database/gesture_users.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    user_name TEXT PRIMARY KEY,
    gesture_name TEXT NOT NULL
)
''')

conn.commit()
conn.close()

print("Gesture database ready")
