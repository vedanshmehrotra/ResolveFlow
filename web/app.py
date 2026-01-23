from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from database import (init_db, init_assignments_table, add_complaint, get_all_complaints, update_complaint_status, 
                      get_complaint_by_id, override_complaint_routing, create_assignment, 
                      get_assignments, update_assignment_status, get_decision_queue)
from model import route_complaint, get_team_name

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Initialize DB on start
init_db()
init_assignments_table()

@app.route('/')
def home():
    """Render Student Portal"""
    return render_template('student.html')

@app.route('/admin')
def admin():
    """Render Admin Dashboard (Read Only)"""
    # Now this is just an overview/audit log
    complaints = get_all_complaints()
    # Fetch assignments for each complaint to show status
    assignments_map = {}
    all_assignments = get_assignments()
    for a in all_assignments:
        cid = a['complaint_id']
        if cid not in assignments_map: assignments_map[cid] = []
        assignments_map[cid].append(a)
        
    return render_template('admin.html', complaints=complaints, assignments_map=assignments_map)

@app.route('/decision-queue')
def decision_queue_page():
    """Render Triage Decision Panel"""
    queue = get_decision_queue()
    return render_template('decision_panel.html', complaints=queue)

@app.route('/assignments')
def assignments_page():
    """Render Operations Inbox"""
    team_filter = request.args.get('team')
    assignments = get_assignments(team_filter)
    return render_template('assignments.html', assignments=assignments, active_team=team_filter)

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
        text = data.get('text', '')
        
        # 0. Input Gate Rule
        if not text or len(text.strip()) < 5:
             print("Invalid input logged:", text)
             return jsonify({'success': True, 'message': 'Input logged', 'data': {'id': -1, 'decision': 'IGNORED'}}), 200

        student_name = data.get('student_name', 'Anonymous')
        student_id = data.get('student_id', '')
        room_number = data.get('room_number', '')
        model_type = data.get('model_type', 'ML') 
        
        # 1. Run Logic routing
        result = route_complaint(text, model_type)
        
        if result['error']:
            return jsonify({'success': False, 'message': result['message']}), 400
            
        # 2. Add ID/Name info to result for DB
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
                
                # STRICT SAFETY GATE
                if conf >= 0.85 and has_keyword:
                    create_assignment(complaint_id, team, issue, conf, sla)
                    assignments_created.append(team)
                    response_assignments.append({
                        "team": team,
                        "issue": issue,
                        "sla": sla,
                        "status": "SENT"
                    })
                # If < 0.85 OR (>= 0.85 but no keyword), it falls through to queue logic below
            
            print("Assignments created:", assignments_created)
            
            # Update status if assignments were made
            if assignments_created:
                 with get_db_connection() as conn:
                      conn.execute("UPDATE complaints SET review_status = 'ROUTED' WHERE id = ?", (complaint_id,))
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

@app.route('/triage')
def triage_page():
    """Render Triage/Review Detail Page"""
    complaint_id = request.args.get('id')
    if not complaint_id:
        return "Missing Complaint ID", 400
    
    complaint = get_complaint_by_id(complaint_id)
    if not complaint:
        return "Complaint not found", 404
        
    return render_template('triage.html', c=complaint)

@app.route('/api/triage/override', methods=['POST'])
def override_triage():
    """Handle manual override of routing and create assignments"""
    try:
        data = request.json
        print("Override payload:", data)
        complaint_id = data.get('id')
        corrected_teams = data.get('corrected_teams', []) # List of team names now
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
