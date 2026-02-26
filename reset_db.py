import sqlite3
import os

DB_PATH = 'web/complaints.db'

def reset_database():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    tables = ['assignments', 'complaints', 'users']
    
    print("Resetting tables...")
    for table in tables:
        try:
            cursor.execute(f"DELETE FROM {table}")
            cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
            print(f"  - Cleared table: {table}")
        except sqlite3.OperationalError as e:
            print(f"  - Error clearing {table}: {e}")

    conn.commit()
    conn.close()
    print("\nDatabase reset complete. All data cleared.")
    print("You can now restart the app and sign up as a new user.")

if __name__ == "__main__":
    confirm = input("Are you sure you want to WIP ALL DATA? (y/N): ")
    if confirm.lower() == 'y':
        reset_database()
    else:
        print("Reset cancelled.")
