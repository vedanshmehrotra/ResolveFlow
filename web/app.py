from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from database import init_db, add_complaint, get_all_complaints, update_complaint_status
from model import route_complaint

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Initialize DB on start
init_db()

@app.route('/')
def home():
    """Render Student Portal"""
    return render_template('student.html')

@app.route('/admin')
def admin():
    """Render Admin Dashboard"""
    complaints = get_all_complaints()
    return render_template('admin.html', complaints=complaints)

@app.route('/api/submit', methods=['POST'])
def submit_complaint():
    """Handle new complaint submission"""
    try:
        data = request.json
        text = data.get('text', '')
        student_name = data.get('student_name', 'Anonymous')
        student_id = data.get('student_id', '')
        model_type = data.get('model_type', 'ML') # ML or DL
        
        # 1. Run Logic routing
        result = route_complaint(text, model_type)
        
        if result['error']:
            return jsonify({'success': False, 'message': result['message']}), 400
            
        # 2. Add ID/Name info to result for DB
        result['student_name'] = student_name
        result['student_id'] = student_id
        result['complaint_text'] = text
        
        # 3. Persist
        complaint_id = add_complaint(result)
        
        return jsonify({
            'success': True,
            'message': 'Complaint processing successful',
            'data': {
                'id': complaint_id,
                'decision': result['routing_decision'],
                'team': result['routed_team'],
                'urgency': result['predicted_urgency']
            }
        })
        
    except Exception as e:
        print(f"Error processing complaint: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/update_status', methods=['POST'])
def update_status():
    """Update complaint status"""
    try:
        data = request.json
        complaint_id = data.get('id')
        new_status = data.get('status')
        notes = data.get('notes')  # Optional notes
        
        if not complaint_id or not new_status:
            return jsonify({'success': False, 'message': 'Missing data'}), 400
            
        success = update_complaint_status(complaint_id, new_status, notes)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Update failed'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    # Run from root dir so relative paths work
    app.run(debug=True, port=5000)
