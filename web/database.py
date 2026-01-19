import sqlite3
import datetime
from typing import List, Dict, Optional, Any
import json

DB_PATH = 'web/complaints.db'

def get_db_connection():
    """Create a database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with the schema and run migrations"""
    schema = """
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT NOT NULL,
        student_id TEXT,
        complaint_text TEXT NOT NULL,
        predicted_issues TEXT,
        predicted_urgency TEXT,
        confidence_scores TEXT,
        routing_decision TEXT,
        routed_team TEXT,
        model_type TEXT,
        status TEXT DEFAULT 'SENT',
        admin_notes TEXT,
        final_status_time TEXT,
        timestamp TEXT NOT NULL
    );
    """
    with get_db_connection() as conn:
        conn.executescript(schema)
        
        # Migration: Add new columns if they don't exist
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT admin_notes FROM complaints LIMIT 1")
        except sqlite3.OperationalError:
            print("Migrating: Adding admin_notes column...")
            cursor.execute("ALTER TABLE complaints ADD COLUMN admin_notes TEXT")

        try:
            cursor.execute("SELECT final_status_time FROM complaints LIMIT 1")
        except sqlite3.OperationalError:
            print("Migrating: Adding final_status_time column...")
            cursor.execute("ALTER TABLE complaints ADD COLUMN final_status_time TEXT")
            
        conn.commit()

def add_complaint(data: Dict[str, Any]) -> int:
    """
    Add a new complaint to the database.
    """
    # Convert lists/dicts to JSON strings for storage
    issues = json.dumps(data.get('predicted_issues', []))
    scores = json.dumps(data.get('confidence_scores', {}))
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    query = """
    INSERT INTO complaints (
        student_name, student_id, complaint_text, 
        predicted_issues, predicted_urgency, confidence_scores,
        routing_decision, routed_team, model_type, status, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (
            data['student_name'],
            data.get('student_id', ''),
            data['complaint_text'],
            issues,
            data['predicted_urgency'],
            scores,
            data['routing_decision'],
            data['routed_team'],
            data.get('model_type', 'ML'),
            'SENT',
            timestamp
        ))
        conn.commit()
        return cursor.lastrowid

def get_all_complaints() -> List[Dict[str, Any]]:
    """Retrieve all complaints from the database ordered by timestamp desc"""
    query = "SELECT * FROM complaints ORDER BY id DESC"
    
    complaints = []
    with get_db_connection() as conn:
        rows = conn.execute(query).fetchall()
        for row in rows:
            # Convert row to dict
            item = dict(row)
            # Parse JSON fields
            try:
                item['predicted_issues'] = json.loads(item['predicted_issues'])
            except:
                item['predicted_issues'] = []
                
            try:
                item['confidence_scores'] = json.loads(item['confidence_scores'])
            except:
                item['confidence_scores'] = {}
                
            complaints.append(item)
            
    return complaints

def update_complaint_status(complaint_id: int, new_status: str, notes: Optional[str] = None) -> bool:
    """Update the status and notes of a complaint"""
    
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if notes is not None:
             query = "UPDATE complaints SET status = ?, admin_notes = ?, final_status_time = ? WHERE id = ?"
             cursor.execute(query, (new_status, notes, time_now, complaint_id))
        else:
             query = "UPDATE complaints SET status = ?, final_status_time = ? WHERE id = ?"
             cursor.execute(query, (new_status, time_now, complaint_id))
             
        conn.commit()
        return cursor.rowcount > 0

# Initialize DB on import if it doesn't exist
if __name__ == "__main__":
    init_db()
    print("Database initialized.")
