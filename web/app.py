from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from database import (init_db, init_assignments_table, add_complaint, get_all_complaints, update_complaint_status, 
                      get_complaint_by_id, override_complaint_routing, create_assignment, 
                      get_assignments, update_assignment_status, get_decision_queue, get_db_connection, get_dashboard_stats)
from model import route_complaint, get_team_name, get_all_teams
from user_mgmt import get_user_by_id, get_user_by_username, verify_password

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
app.secret_key = 'super-secret-key-for-hostel-triage' # In production use environment variable

# Flask-Login Setup
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, user_data):
        self.id = user_data['id']
        self.username = user_data['username']
        self.role = user_data['role']

@login_manager.user_loader
def load_user(user_id):
    user_data = get_user_by_id(int(user_id))
    if user_data:
        return User(user_data)
    return None

@app.context_processor
def inject_user():
    return dict(user=current_user)

# Initialize DB on start
init_db()
init_assignments_table()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_data = get_user_by_username(username)
        if user_data and verify_password(user_data['password_hash'], password):
            user = User(user_data)
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for('admin')) # Changed from admin_dashboard to admin to match existing route
            return redirect(url_for('home'))
        return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username')
    password = request.form.get('password')
    role = request.form.get('role', 'student')
    
    if not username or not password:
        return render_template('login.html', error='All fields are required')
        
    try:
        from user_mgmt import create_user
        user_id = create_user(username, password, role=role)
        if user_id:
            return render_template('login.html', error='Account created! Please login.')
        else:
            return render_template('login.html', error='Username already exists')
    except Exception as e:
        return render_template('login.html', error=f'Signup failed: {str(e)}')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    """Render Student Portal with user's complaints"""
    # Fetch complaints only for this student
    complaints = get_all_complaints(user_id=current_user.id, hide_sensitive=True) if current_user.role != 'admin' else []
    return render_template('student.html', user=current_user, complaints=complaints, is_student=True)

@app.route('/admin')
@login_required
def admin():
    """Render Admin Dashboard (Read Only)"""
    if current_user.role != 'admin':
        return redirect(url_for('home'))
    complaints = get_all_complaints()
    # Fetch assignments for each complaint to show status
    assignments_map = {}
    all_assignments = get_assignments()
    for a in all_assignments:
        cid = a['complaint_id']
        if cid not in assignments_map: assignments_map[cid] = []
        assignments_map[cid].append(a)
    
    # Get DB-derived stats
    stats = get_dashboard_stats()
        
    return render_template('admin.html', complaints=complaints, assignments_map=assignments_map, stats=stats, is_student=False)

@app.route('/decision-queue')
@login_required
def decision_queue_page():
    """Render Triage Decision Panel"""
    if current_user.role != 'admin':
        return redirect(url_for('home'))
    priority_filter = request.args.get('priority')
    queue = get_decision_queue()
    
    # Filter by priority if selected
    if priority_filter and priority_filter.lower() in ['high', 'medium', 'low']:
        queue = [c for c in queue if c.get('predicted_urgency', '').lower() == priority_filter.lower()]
        
    return render_template('decision_panel.html', complaints=queue, active_priority=priority_filter, is_student=False, get_all_teams=get_all_teams)

@app.route('/assignments')
@login_required
def assignments_page():
    """Render Operations Inbox"""
    # For now, Ops also uses Admin/Student login, 
    # but we could add an 'ops' role later.
    if current_user.role != 'admin':
        return redirect(url_for('home'))
    team_filter = request.args.get('team')
    
    # Get Dynamic Teams
    from database import get_active_teams
    all_teams = get_active_teams()
    
    assignments = get_assignments(team_filter)
    TEAM_ISSUE_MAP = {
        'Electrical Team': 'Electrical Issue',
        'IT / Network Team': 'Network / Internet Issue',
        'Plumbing Team': 'Plumbing Issue',
        'Housekeeping Team': 'Housekeeping Issue',
        'Furniture & Infrastructure Team': 'Furniture Issue',
        'Mess Management': 'Mess Issue',
        'Hostel Administration': 'Admin Issue',
        'Accounts Department': 'Accounts Issue',
    }
    return render_template('assignments.html', assignments=assignments, active_team=team_filter, all_teams=all_teams, is_student=False, team_issue_map=TEAM_ISSUE_MAP)

@app.route('/api/triage/ignore', methods=['POST'])
def ignore_issue_route():
    """Mark a specific issue as ignored so it leaves the queue"""
    try:
        data = request.json
        complaint_id = data.get('id')
        issue = data.get('issue')
        
        if not complaint_id or not issue:
             return jsonify({'success': False, 'message': 'Missing data'}), 400
             
        # We assume the goal is to stop showing it in the queue.
        # Our queue logic filters out issues mentioned in admin_notes as "IGNORED_ISSUE: {issue}"
        
        # Append note
        note = f"\nIGNORED_ISSUE: {issue}"
        
        with get_db_connection() as conn:
             # concat to review_notes
             conn.execute("UPDATE complaints SET review_notes = COALESCE(review_notes, '') || ? WHERE id = ?", (note, complaint_id))
             conn.commit()
             
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error ignoring issue: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Helper for SLA
def get_sla_class(urgency: str) -> str:
    urgency = urgency.lower()
    if urgency == 'high': return 'P1'
    if urgency == 'medium': return 'P2'
    return 'P3' # Low

@app.route('/api/submit', methods=['POST'])
def submit_complaint():
    """Handle new complaint submission"""
    try:
        data = request.json
        text = data.get('complaint_text') or data.get('text', '')
        
        # 0. Input Gate Rule
        if not text or len(text.strip()) < 5:
             print("Invalid input logged:", text)
             return jsonify({'success': True, 'message': 'Input logged', 'data': {'id': -1, 'decision': 'IGNORED'}}), 200

        student_name = current_user.username
        student_id = data.get('student_id', '')
        room_number = data.get('room_number', '')
        model_type = data.get('model_type', 'ML') 
        
        # 1. Run Logic routing
        result = route_complaint(text, model_type)
        
        if result['error']:
            return jsonify({'success': False, 'message': result['message']}), 400
            
        # 2. Add ID/Name info to result for DB
        result['user_id'] = current_user.id
        result['student_name'] = student_name
        result['student_id'] = student_id
        result['room_number'] = room_number
        result['complaint_text'] = text
        result['assignment_source'] = f'MODEL_{model_type}'
        
        # 3. Persist Complaint
        complaint_id = add_complaint(result)
        
        # 4. Generate Assignments
        # SAFETY GATE: Confidence >= 0.85 AND Keyword Match -> Auto-Assign
        # Else -> Queue (if >= 0.65)
        
        KEYWORD_WHITELIST = {
            'electrical': ['spark', 'fuse', 'bulb', 'wiring', 'short', 'shock', 'power', 'light', 'switch'],
            'plumbing': ['leak', 'water', 'pipe', 'tap', 'drain', 'flood', 'flush', 'sink', 'shower', 'toilet'],
            'internet': ['wifi', 'router', 'network', 'disconnect', 'slow', 'internet', 'connection'],
            'cleanliness': ['dirty', 'smell', 'garbage', 'clean', 'stain', 'dust', 'mess', 'trash'],
            'noise': ['loud', 'music', 'shouting', 'noise', 'party', 'bang'],
            'furniture': ['desk', 'chair', 'bed', 'furniture', 'table', 'wardrobe', 'cupboard', 'shelf', 'broken', 'damaged']
        }
        
        assignments_created = []
        
        try:
            issues = result['predicted_issues']
            scores = result['confidence_scores']
            urgency = result['predicted_urgency']
            sla = get_sla_class(urgency)
            
            print("Detected issues:", issues)
            
            response_assignments = []
            
            for issue in issues:
                conf = scores.get(issue, 0)
                team = get_team_name(issue)
                
                # Check Keyword Signal
                keywords = KEYWORD_WHITELIST.get(issue.lower(), [])
                text_lower = text.lower()
                has_keyword = any(k in text_lower for k in keywords)
                
                print(f"Issue: {issue}, Conf: {conf}, Keyword Found: {has_keyword}")
                
                # Routing Logic:
                # 1. Auto-Assign: >= 0.75
                
                if conf >= 0.75:
                    create_assignment(complaint_id, team, issue, conf, sla)
                    assignments_created.append(team)
                    response_assignments.append({
                        "team": team,
                        "issue": issue,
                        "sla": sla,
                        "status": "SENT"
                    })
                elif conf >= 0.30:
                    # 2. Manual Review: 30-74%
                    # Do not create assignment. It will show in Decision Queue.
                    pass
                else:
                    # 3. Ignore: < 30%
                    pass
            
            print("Assignments created:", assignments_created)
            
            # Update status if assignments were made
            if assignments_created:
                 with get_db_connection() as conn:
                      conn.execute("UPDATE complaints SET review_status = 'PARTIAL' WHERE id = ?", (complaint_id,))
                      conn.commit()

        except Exception as ex:
            print(f"Assignment creation failed: {ex}")

        return jsonify({
            'success': True,
            'message': 'Complaint processed',
            'data': {
                'id': complaint_id,
                'decision': 'AUTO_ROUTE' if assignments_created else 'HUMAN_REVIEW',
                'assignments': response_assignments,
                'urgency': result['predicted_urgency']
            }
        })
        
    except Exception as e:
        print(f"Error processing complaint: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/complaints/rate', methods=['POST'])
@login_required
def rate_complaint():
    """Submit rating and feedback for a resolved complaint"""
    data = request.json
    complaint_id = data.get('complaint_id')
    rating = data.get('rating')
    feedback = data.get('feedback', '')
    
    if not complaint_id or not rating:
        return jsonify({'success': False, 'message': 'Missing data'}), 400
        
    try:
        from database import update_complaint_feedback
        update_complaint_feedback(complaint_id, rating, feedback)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
@app.route('/api/override/<int:complaint_id>', methods=['POST'])
@login_required
def override_complaint(complaint_id):
    if current_user.role != 'admin':
        return jsonify({'success': False}), 403
    data = request.json
    teams = data.get('teams', [])
    if not teams:
        return jsonify({'success': False, 'message': 'No teams provided'}), 400
        
    try:
        from database import update_complaint_status, create_assignment
        with get_db_connection() as conn:
            # Update complaint to OVERRIDDEN status directly in DB or use function
            conn.execute("UPDATE complaints SET review_status = 'OVERRIDDEN', routing_decision = 'MANUAL_OVERRIDE' WHERE id = ?", (complaint_id,))
            conn.commit()
            
        for team in teams:
            create_assignment(complaint_id, team, 'Manual Override', 1.0)
            
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/reject/<int:complaint_id>', methods=['POST'])
@login_required
def reject_complaint(complaint_id):
    if current_user.role != 'admin':
        return jsonify({'success': False}), 403
    try:
        with get_db_connection() as conn:
            conn.execute("UPDATE complaints SET review_status = 'REJECTED', routing_decision = 'REJECTED' WHERE id = ?", (complaint_id,))
            conn.commit()
        return jsonify({'success': True, 'id': complaint_id})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/assignment/<int:assignment_id>/note', methods=['POST'])
@login_required
def update_assignment_note(assignment_id):
    data = request.get_json()
    try:
        with get_db_connection() as conn:
            conn.execute("UPDATE assignments SET notes = ? WHERE id = ?", (data.get('notes', ''), assignment_id))
            conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/assignment/<int:assignment_id>/status', methods=['POST'])
@login_required
def update_assignment_status_route(assignment_id):
    """Update assignment status"""
    try:
        data = request.json
        new_status = data.get('status')
        notes = data.get('notes')
        
        if not new_status:
             return jsonify({'success': False, 'message': 'Missing data'}), 400
             
        from database import update_assignment_status
        success = update_assignment_status(assignment_id, new_status, notes)
        
        if success:
             return jsonify({'success': True})
        else:
             return jsonify({'success': False, 'message': 'Update failed'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/update_status', methods=['POST'])
def update_status():
    """Legacy Endpoint - Keep for backward compatibility or Admin override"""
    # This might now update the main complaint status, not individual assignments
    # We'll leave it as is for the complaint-level status.
    try:
        data = request.json
        complaint_id = data.get('id')
        new_status = data.get('status')
        notes = data.get('notes') 
        
        if not complaint_id or not new_status:
            return jsonify({'success': False, 'message': 'Missing data'}), 400
            
        success = update_complaint_status(complaint_id, new_status, notes)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Update failed'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
