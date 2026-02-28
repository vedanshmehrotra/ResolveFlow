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
    print("Initializing Database...")
    schema = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'student',
        profile_info TEXT
    );
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        student_name TEXT NOT NULL,
        student_id TEXT,
        room_number TEXT,
        complaint_text TEXT NOT NULL,
        predicted_issues TEXT,
        predicted_urgency TEXT,
        confidence_scores TEXT,
        routing_decision TEXT,
        routed_team TEXT,
        model_type TEXT,
        assignment_source TEXT,
        status TEXT DEFAULT 'SENT',
        admin_notes TEXT,
        final_status_time TEXT,
        review_status TEXT DEFAULT 'PENDING',
        corrected_issues TEXT,
        corrected_urgency TEXT,
        corrected_team TEXT,
        review_notes TEXT,
        review_timestamp TEXT,
        rating INTEGER,
        feedback TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS assignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint_id INTEGER,
        team TEXT,
        issue_label TEXT,
        confidence REAL,
        status TEXT DEFAULT 'SENT',
        notes TEXT,
        timestamp TEXT,
        FOREIGN KEY(complaint_id) REFERENCES complaints(id)
    );
    """
    
    with get_db_connection() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(schema)
        
        # Migration: Add new columns if they don't exist
        cursor = conn.cursor()
        columns_to_add = [
            ("admin_notes", "TEXT"),
            ("final_status_time", "TEXT"),
            ("room_number", "TEXT"),
            ("assignment_source", "TEXT"),
            ("review_status", "TEXT DEFAULT 'PENDING'"),
            ("corrected_issues", "TEXT"),
            ("corrected_urgency", "TEXT"),
            ("corrected_team", "TEXT"),
            ("review_notes", "TEXT"),
            ("review_timestamp", "TEXT"),
            ("user_id", "INTEGER"),
            ("rating", "INTEGER"),
            ("feedback", "TEXT")
        ]
        
        for col_name, col_type in columns_to_add:
            try:
                cursor.execute(f"SELECT {col_name} FROM complaints LIMIT 1")
            except sqlite3.OperationalError:
                try:
                    print(f"Migrating: Adding {col_name} column...")
                    cursor.execute(f"ALTER TABLE complaints ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError as e:
                    print(f"Skipping {col_name}: {e}")
            
        conn.commit()
    print("Database Initialization Complete.")

def init_assignments_table():
    """Create (migrate) the assignments table if it doesn't exist"""
    # Note: Added sla_class
    schema = """
    CREATE TABLE IF NOT EXISTS assignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint_id INTEGER,
        team TEXT,
        issue_label TEXT,
        confidence REAL,
        status TEXT DEFAULT 'SENT',
        notes TEXT,
        timestamp TEXT,
        sla_class TEXT,
        FOREIGN KEY(complaint_id) REFERENCES complaints(id)
    );
    """
    with get_db_connection() as conn:
        conn.executescript(schema)
        # Migration: Add sla_class if missing
        try:
             conn.execute("SELECT sla_class FROM assignments LIMIT 1")
        except sqlite3.OperationalError:
             print("Migrating: Adding sla_class to assignments")
             conn.execute("ALTER TABLE assignments ADD COLUMN sla_class TEXT")

def create_assignment(complaint_id: int, team: str, issue: str, confidence: float, sla_class: str = 'P3') -> int:
    """Create a new team assignment for a complaint"""
    query = """
    INSERT INTO assignments (complaint_id, team, issue_label, confidence, status, timestamp, sla_class)
    VALUES (?, ?, ?, ?, 'SENT', ?, ?)
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (complaint_id, team, issue, confidence, timestamp, sla_class))
        conn.commit()
        return cursor.lastrowid

def get_assignments(team_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get assignments, optionally filtered by team"""
    query = """
    SELECT a.*, c.room_number, c.predicted_urgency, c.corrected_urgency, c.student_name 
    FROM assignments a
    JOIN complaints c ON a.complaint_id = c.id
    """
    args = []
    if team_filter:
        query += " WHERE a.team = ?"
        args.append(team_filter)
    
    query += " ORDER BY a.sla_class ASC, a.timestamp DESC"
    
    results = []
    with get_db_connection() as conn:
        rows = conn.execute(query, args).fetchall()
        for row in rows:
            results.append(dict(row))
    return results

def get_active_teams() -> List[str]:
    """Get list of teams that have active assignments"""
    query = "SELECT DISTINCT team FROM assignments ORDER BY team ASC"
    with get_db_connection() as conn:
        rows = conn.execute(query).fetchall()
        return [row['team'] for row in rows if row['team']]

def get_decision_queue() -> List[Dict[str, Any]]:
    """
    Get individual issues that need human review.
    Rule: Issue confidence between 0.65 and 0.84 AND no assignment exists.
    """
    from model import get_team_name
    
    # 1. Get detailed complaints
    query = "SELECT * FROM complaints WHERE status != 'RESOLVED' AND IFNULL(review_status, 'PENDING') = 'PENDING' ORDER BY timestamp DESC"
    
    candidates = []
    with get_db_connection() as conn:
        rows = conn.execute(query).fetchall()
        for row in rows:
             candidates.append(dict(row))
             
    final_queue = []
    
    final_queue_map = {}
    
    for c in candidates:
        try: 
            issues_json = c.get('predicted_issues')
            scores_json = c.get('confidence_scores')
            issues = json.loads(issues_json) if issues_json else []
            scores = json.loads(scores_json) if scores_json else {}
        except Exception as e:
            print(f"Error parsing JSON for complaint {c.get('id')}: {e}")
            issues = []
            scores = {}
            
        # Get existing assignments to filter out handled issues
        existing_assignments = get_assignments_map_for_complaint(c.get('id', 0))
        assigned_teams = {a['team'] for a in existing_assignments}
        
        # Check ignored issues from admin_notes/review_notes
        ignored_notes = (str(c.get('admin_notes') or '')) + (str(c.get('review_notes') or ''))
        
        # Complaint-level list for grouped display
        pending_for_this_complaint = []
        
        # If no issues detected but status is SENT, add as "Unclassified"
        if not issues and c.get('status') == 'SENT' and not existing_assignments:
             pending_for_this_complaint.append({
                 'issue': 'Unclassified',
                 'confidence': 0.0,
                 'suggested_team': 'Hostel Administration'
             })
        
        for issue in issues:
            conf = scores.get(issue, 0)
            team = get_team_name(issue)
            
            if f"IGNORED_ISSUE: {issue}" in ignored_notes:
                continue

            # LOGIC: 0.30 <= conf < 0.85 -> Queue
            # Fallback: if >= 0.85 but no assignment exists
            if (0.30 <= conf < 0.85 or conf >= 0.85) and team not in assigned_teams:
                pending_for_this_complaint.append({
                    'issue': issue,
                    'confidence': conf,
                    'suggested_team': team
                })
        
        if pending_for_this_complaint:
            item = dict(c)
            item['pending_issues'] = pending_for_this_complaint
            # For backward compatibility with simpler templates, set primary fields
            item['review_issue'] = pending_for_this_complaint[0]['issue']
            item['review_confidence'] = pending_for_this_complaint[0]['confidence']
            item['suggested_team'] = pending_for_this_complaint[0]['suggested_team']
            
            final_queue_map[c['id']] = item
            
    final_queue = list(final_queue_map.values())
    print(f"Decision queue built: {len(final_queue)} unique complaints")
    return final_queue

def get_assignments_map_for_complaint(complaint_id: int) -> List[Dict[str, Any]]:
    """Helper to get assignments for a specific ID quickly"""
    if not complaint_id: return []
    query = "SELECT * FROM assignments WHERE complaint_id = ?"
    try:
        with get_db_connection() as conn:
            rows = conn.execute(query, (complaint_id,)).fetchall()
            return [dict(r) for r in rows]
    except:
        return []

def update_assignment_status(assignment_id: int, status: str, notes: Optional[str] = None) -> bool:
    """Update assignment status"""
    query = "UPDATE assignments SET status = ?"
    args = [status]
    if notes:
        query += ", notes = ?"
        args.append(notes)
    
    query += " WHERE id = ?"
    args.append(assignment_id)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, args)
        
        # AUTO-SYNC LOGIC:
        # If this assignment was resolved, check if all others for this complaint are also resolved.
        if status.upper() == 'RESOLVED':
            # Get the complaint_id for this assignment
            res = cursor.execute("SELECT complaint_id FROM assignments WHERE id = ?", (assignment_id,)).fetchone()
            if res:
                cid = res[0]
                # Check all assignments for this CID
                all_assigns = cursor.execute("SELECT status FROM assignments WHERE complaint_id = ?", (cid,)).fetchall()
                if all(a[0].upper() == 'RESOLVED' for a in all_assigns):
                    # Auto-resolve parent complaint
                    update_complaint_status(cid, 'RESOLVED', "Automatically resolved as all assigned tasks are complete.")
        
        conn.commit()
        return cursor.rowcount > 0


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
        user_id, student_name, student_id, room_number, complaint_text, 
        predicted_issues, predicted_urgency, confidence_scores,
        routing_decision, routed_team, model_type, assignment_source, 
        status, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (
            data.get('user_id'),
            data['student_name'],
            data.get('student_id', ''),
            data.get('room_number', ''),
            data['complaint_text'],
            issues,
            data['predicted_urgency'],
            scores,
            data['routing_decision'],
            data['routed_team'],
            data.get('model_type', 'ML'),
            data.get('assignment_source', 'MODEL_ML'),
            'SENT',
            timestamp
        ))
        conn.commit()
        return cursor.lastrowid

def get_all_complaints(user_id: Optional[int] = None, hide_sensitive: bool = False) -> List[Dict[str, Any]]:
    """Retrieve complaints, optionally filtered by user_id"""
    query = "SELECT * FROM complaints"
    args = []
    if user_id:
        query += " WHERE user_id = ?"
        args.append(user_id)
        
    query += " ORDER BY id DESC"
    
    complaints = []
    with get_db_connection() as conn:
        rows = conn.execute(query, args).fetchall()
        for row in rows:
            # Convert row to dict
            item = dict(row)
            
            if hide_sensitive:
                # Remove sensitive fields for student view
                item['confidence_scores'] = {}
                item['routing_decision'] = "Pending" if item['status'] == 'SENT' else item['status']
                item['routed_team'] = "Privacy Masked" # Or just hide in UI
            # Parse JSON fields
            try:
                item['predicted_issues'] = json.loads(item['predicted_issues'])
            except:
                item['predicted_issues'] = []
                
            try:
                item['confidence_scores'] = json.loads(item['confidence_scores'])
            except:
                item['confidence_scores'] = {}
            
            # Parse corrected issues if present
            if item.get('corrected_issues'):
                try:
                    item['corrected_issues'] = json.loads(item['corrected_issues'])
                except:
                    item['corrected_issues'] = []
            
            # Fetch associated assignments for a complete progress view
            if hide_sensitive:
                item['assignments'] = get_assignments_map_for_complaint(item['id'])
                # Summarize status: If any assignment is NOT resolved, main status is 'IN PROGRESS'
                # but we respect 'RESOLVED' if explicitly set. 
                # For simplified status in UI:
                item['routing_decision'] = "Assigned" if item['assignments'] else "Pending Review"
                
            complaints.append(item)
            
    return complaints

def get_complaint_by_id(complaint_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a single complaint by ID"""
    query = "SELECT * FROM complaints WHERE id = ?"
    with get_db_connection() as conn:
        row = conn.execute(query, (complaint_id,)).fetchone()
        if row:
            item = dict(row)
            # Parse JSON
            try:
                item['predicted_issues'] = json.loads(item['predicted_issues'])
            except:
                item['predicted_issues'] = []
            try:
                item['confidence_scores'] = json.loads(item['confidence_scores'])
            except:
                item['confidence_scores'] = {}
            if item.get('corrected_issues'):
                try:
                    item['corrected_issues'] = json.loads(item['corrected_issues'])
                except:
                    item['corrected_issues'] = []
            return item
    return None

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

def override_complaint_routing(complaint_id: int, data: Dict[str, Any]) -> bool:
    """
    Override urgency, issues, and team manually.
    """
    query = """
    UPDATE complaints 
    SET 
        corrected_issues = ?,
        corrected_urgency = ?,
        corrected_team = ?,
        review_notes = ?,
        assignment_source = 'MANUAL',
        review_status = 'OVERRIDDEN',
        review_timestamp = ?
    WHERE id = ?
    """
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    corrected_issues_json = json.dumps(data.get('corrected_issues', []))
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (
            corrected_issues_json,
            data.get('corrected_urgency'),
            data.get('corrected_team'),
            data.get('review_notes'),
            timestamp,
            complaint_id
        ))
        conn.commit()
        return cursor.rowcount > 0

def update_complaint_feedback(complaint_id: int, rating: int, feedback: str):
    """Update rating and feedback for a complaint"""
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE complaints SET rating = ?, feedback = ? WHERE id = ?",
            (rating, feedback, complaint_id)
        )
        conn.commit()

def get_dashboard_stats() -> Dict[str, int]:
    """Get dashboard statistics from database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Total complaints
        total = cursor.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
        
        # Auto-routed = complaints that have at least one auto-assigned assignment
        auto_routed = cursor.execute("""
            SELECT COUNT(DISTINCT complaint_id) FROM assignments 
            WHERE confidence >= 0.85
        """).fetchone()[0]
        
        # Pending review = decision queue count (issues between 0.30 and 0.85 not yet resolved or fallback)
        pending = len(get_decision_queue())
        
        # Resolved = count of complaints where status is RESOLVED
        resolved = cursor.execute("SELECT COUNT(*) FROM complaints WHERE status = 'RESOLVED'").fetchone()[0]
        
        return {
            'total': total,
            'auto_routed': auto_routed,
            'pending_review': pending,
            'resolved': resolved
        }

# Initialize DB on import if it doesn't exist
if __name__ == "__main__":
    init_db()
    print("Database initialized.")

