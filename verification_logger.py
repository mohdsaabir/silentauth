import sqlite3
from datetime import datetime

def log_verification(username, success):

    conn = sqlite3.connect("database/central.db")
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if success:
        result = "SUCCESS"
    else:
        result = "FAILED"
        username = "unknown"

    cursor.execute(
        "INSERT INTO verification_logs (timestamp, username, result) VALUES (?,?,?)",
        (timestamp, username, result)
    )

    conn.commit()
    conn.close()