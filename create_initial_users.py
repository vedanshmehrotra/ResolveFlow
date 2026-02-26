import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'web'))

from user_mgmt import create_user
from database import init_db

if __name__ == "__main__":
    init_db()
    
    # Create Admin
    admin_id = create_user("admin", "admin123", role="admin", profile_info="System Administrator")
    if admin_id:
        print(f"Admin created with ID: {admin_id}")
    else:
        print("Admin user already exists.")
        
    # Create a Test Student
    student_id = create_user("student1", "pass123", role="student", profile_info="Hostel Block A, Room 101")
    if student_id:
        print(f"Student created with ID: {student_id}")
    else:
        print("Student user already exists.")
