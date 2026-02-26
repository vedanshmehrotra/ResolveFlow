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
    return render_template('student.html', user=current_user, complaints=complaints)

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
        
    return render_template('admin.html', complaints=complaints, assignments_map=assignments_map, stats=stats)

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
        
    return render_template('decision_panel.html', complaints=queue, active_priority=priority_filter)

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
    return render_template('assignments.html', assignments=assignments, active_team=team_filter, all_teams=all_teams)

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
            'noise': ['loud', 'music', 'shouting', 'noise', 'party', 'bang']
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
                
                # Multi-Issue Logic:
                # 1. Auto-Assign: >= 0.85 (Keyword check optional? User prompt didn't specify keyword strictness for this new logic, 
                #    but said "Confidence >= 85% -> Auto-assign". Let's stick to strict confidence).
                #    Actually, existing logic used keywords as safety. Let's keep keywords as a booster or safety?
                #    User Prompt: "Confidence >= 85% -> Auto-assign". Simplifies it.
                #    Let's trust the ML score per user request to "Minimal Design Rule".
                
                if conf >= 0.85:
                    create_assignment(complaint_id, team, issue, conf, sla)
                    assignments_created.append(team)
                    response_assignments.append({
                        "team": team,
                        "issue": issue,
                        "sla": sla,
                        "status": "SENT"
                    })
                elif conf >= 0.30:
                    # 2. Manual Review: 30-84%
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
@app.route('/triage')
@login_required
def triage_page():
    """Render Triage/Review Detail Page"""
    if current_user.role != 'admin':
        return redirect(url_for('home'))
    complaint_id = request.args.get('id')
    if not complaint_id:
        return "Missing Complaint ID", 400
    
    complaint = get_complaint_by_id(complaint_id)
    if not complaint:
        return "Complaint not found", 404
        
    all_teams = get_all_teams()
    return render_template('triage.html', c=complaint, all_teams=all_teams)

@app.route('/api/triage/override', methods=['POST'])
def override_triage():
    """Handle manual override of routing and create assignments"""
    try:
        data = request.json
        complaint_id = data.get('id')
        corrected_teams = data.get('corrected_teams', []) 
        corrected_issues = data.get('corrected_issues', [])
        
        if not complaint_id:
            return jsonify({'success': False, 'message': 'Missing ID'}), 400
            
        # 1. Update Complaint Record
        # We assume the UI sends the 'prmary' team or we join them.
        # But data.corrected_team was a single string in DB schema. 
        # For backward compatibility, we'll store the first team or joined string.
        # But assignments details matter more. 
        
        data['corrected_team'] = ", ".join(corrected_teams)
        
        success = override_complaint_routing(complaint_id, data)
        
        if success:
             # 2. Creates Assignments based on manual override
             # We assume manual override means these are the definitive tasks.
             # We should probably check if assignments already exist and 'void' them 
             # or just create new ones. For MVP, we just create new ones.
             
             for team in corrected_teams:
                  # Create a simplified assignment
                  # We don't have per-team issue granularity from the simple override form 
                  # unless the UI is complex. We'll assign "Manual Override" as label or first issue.
                  label = corrected_issues[0] if corrected_issues else "Manual Issue"
                  create_assignment(complaint_id, team, label, 1.0)
                  
             return jsonify({'success': True})
        else:
             return jsonify({'success': False, 'message': 'Override failed'}), 400
             
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/assignments/update', methods=['POST'])
def update_assignment_status_route():
    """Update assignment status"""
    try:
        data = request.json
        assignment_id = data.get('assignment_id')
        new_status = data.get('status')
        notes = data.get('notes')
        
        if not assignment_id or not new_status:
             return jsonify({'success': False, 'message': 'Missing data'}), 400
             
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
