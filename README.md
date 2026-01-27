# 📧 Hostel Complaint Triage System

An AI-powered complaint routing system for hostel management, featuring multi-issue detection and intelligent team assignment.

## 🚀 Features

- **Multi-Issue Detection**: Identifies multiple issues in a single complaint (e.g., electrical + plumbing)
- **Confidence-Based Routing**:
  - ≥85% confidence → Auto-assigned to team
  - 65-84% confidence → Queued for manual review
  - <65% confidence → Ignored
- **Decision Queue**: Admin panel for reviewing uncertain assignments
- **Operations Inbox**: Team-wise task management with SLA tracking
- **Dark Mode UI**: Modern, responsive dark-themed interface

## 📁 Project Structure

```
Email_Triage_MiniProject/
├── web/                    # Flask Backend
│   ├── app.py              # Main Flask application
│   ├── database.py         # SQLite database operations
│   └── model.py            # ML model loading & inference
├── frontend/
│   └── templates/          # Jinja2 HTML templates
│       ├── student.html    # Student complaint form
│       ├── admin.html      # Admin dashboard
│       ├── decision_panel.html  # Triage queue
│       ├── assignments.html     # Operations inbox
│       └── triage.html     # Manual review page
├── models/                 # Trained ML models (.keras, .pkl)
├── data/                   # Training/test datasets
├── results/                # Evaluation metrics & plots
├── Phase 1-4*.py           # ML pipeline scripts
├── streamlit_app.py        # Alternative Streamlit interface
└── requirements.txt        # Python dependencies
```

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Email_Triage_MiniProject.git
cd Email_Triage_MiniProject

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Running the Application

### Flask Web App (Recommended)
```bash
cd web
python app.py
```
Visit `http://localhost:5000`

### Streamlit App (Alternative)
```bash
streamlit run streamlit_app.py
```

## 📊 ML Pipeline

Run in sequence to train models:
1. `Phase 1-Data Preperation.py` - Data loading & preprocessing
2. `Phase 2-Feature Engineering.py` - TF-IDF & tokenization
3. `Phase 3-ML Baseline Models.py` - Traditional ML (Logistic Regression, etc.)
4. `Phase 4-Deep Learning Models.py` - LSTM classifiers

## 🎯 Issue Categories

- Electrical, Plumbing, Internet/Network
- Furniture, Cleanliness, Food/Mess
- Noise, Billing, Other

## 📝 License

MIT License
