import hashlib
import sqlite3
import os
from database import get_db_connection

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    # Using a simple SHA-256 for now as per minimal change rule, 
    # but in production bcrypt would be preferred.
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password_hash: str, provided_password: str) -> bool:
    """Verify a stored password against one provided by user."""
    return stored_password_hash == hash_password(provided_password)

def create_user(username, password, role='student', profile_info=None):
    """Create a new user in the database."""
    password_hash = hash_password(password)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, profile_info) VALUES (?, ?, ?, ?)",
                (username, password_hash, role, profile_info)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            print(f"User {username} already exists")
            return None

def get_user_by_username(username):
    """Fetch user details by username."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None

def get_user_by_id(user_id):
    """Fetch user details by ID."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
