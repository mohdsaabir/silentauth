import sqlite3
import os

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get(
    "SILENTAUTH_DB_PATH",
    os.path.abspath(os.path.join(BASE_DIR, "database", "central.db"))
)

def preset_selector(username):
    """
    Fetches the preset value for a given username from DB.
    Returns "normal" if no preset is found.
    """
    if not username:
        return "normal"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT preset FROM users WHERE name = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row and row[0]:
        return row[0]

    return "normal"
