"""
HOSTEL COMPLAINT TRIAGE SYSTEM - ADMIN DASHBOARD
=================================================
Professional admin interface with per-issue routing logic

Author: Vedansh
Date: December 2025

KEY IMPROVEMENT:
- Changed from global routing (highest confidence wins)
- To per-issue routing (each issue evaluated independently)
- Prevents misrouting of multi-issue complaints
"""

import streamlit as st
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Hostel Complaint Triage System",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Admin Dashboard Style
CUSTOM_CSS = """<style>
    /* Reset & Base Styles */
    :root {
        --bg-color: #f8f9fa;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --primary-color: #3b82f6;
        --success-color: #22c55e;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }

    /* Global Overrides */
    .stApp {
        background-color: var(--bg-color);
    }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }

    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    p, label {
        color: var(--text-secondary) !important;
    }

    /* Admin Card Components */
    .admin-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
    }
    
    .admin-card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .admin-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--bg-color);
    }
    
    .admin-row:last-child {
        border-bottom: none;
    }
    
    .admin-label {
        font-weight: 500;
        color: var(--text-secondary);
    }
    
    .admin-value {
        font-weight: 600;
        color: var(--text-primary);
    }

    /* Routing Cards */
    .routing-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: transform 0.1s ease-in-out;
    }

    .routing-auto {
        border-left: 4px solid var(--primary-color);
    }

    .routing-review {
        border-left: 4px solid var(--warning-color);
    }

    .routing-title {
        font-weight: 700;
        color: var(--text-primary) !important;
        font-size: 1rem;
        display: block;
        margin-bottom: 0.25rem;
    }

    .routing-desc {
        color: var(--text-secondary) !important;
        font-size: 0.9rem;
    }
    
    .routing-badge-container {
        text-align: right;
        min-width: 120px;
    }

    /* Badges */
    .badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-high {
        background-color: #dcfce7;
        color: #15803d;
    }
    
    .badge-medium {
        background-color: #fef9c3;
        color: #a16207;
    }
    
    .badge-low {
        background-color: #fee2e2;
        color: #b91c1c;
    }

    /* Streamlit Component overrides */
    .stTextArea textarea {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px;
        color: var(--text-primary) !important;
        caret-color: var(--text-primary) !important;
    }
    
    .stTextInput input {
        color: var(--text-primary) !important;
        caret-color: var(--text-primary) !important;
    }
    
    .stButton button {
        background-color: var(--primary-color);
        color: white;
        font-weight: 500;
        border-radius: 6px;
        border: none;
        transition: background-color 0.2s;
    }
    
    .stButton button:hover {
        background-color: #2563eb;
    }

    /* Metric/Info display tweaks */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border-color);
        border-radius: 6px;
        background-color: var(--card-bg);
    }
</style>"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        with st.spinner('Downloading language data...'):
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt_tab', quiet=True)

download_nltk_data()
# Load models
@st.cache_resource
def load_models():
    try:
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        with open('models/issue_classifier_ml.pkl', 'rb') as f:
            issue_classifier_ml = pickle.load(f)

        with open('models/urgency_classifier_ml.pkl', 'rb') as f:
            urgency_classifier_ml = pickle.load(f)

        issue_classifier_lstm = load_model('models/issue_classifier_lstm.keras')
        urgency_classifier_lstm = load_model('models/urgency_classifier_lstm.keras')

        with open('models/tokenizer_dl.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        with open('data/category_mapping.json', 'r') as f:
            mappings = json.load(f)

        return {
            'tfidf': tfidf_vectorizer,
            'issue_ml': issue_classifier_ml,
            'urgency_ml': urgency_classifier_ml,
            'issue_lstm': issue_classifier_lstm,
            'urgency_lstm': urgency_classifier_lstm,
            'tokenizer': tokenizer,
            'categories': mappings['categories'],
            'urgency_labels': ['low', 'medium', 'high']
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()
def validate_complaint(text):
    if not text or text.strip() == "":
        return False, "error", "Please enter a complaint description"
    
    word_count = len(text.split())
    if word_count < 5:
        return False, "warning", f"Too short ({word_count} words). Minimum 5 words required"
    
    if word_count > 500:
        return False, "warning", f"Too long ({word_count} words). Maximum 500 words"
    
    spam_keywords = ['buy', 'sale', 'discount', 'click here', 'viagra', 
                     'casino', 'lottery', 'prize', 'congratulations']
    text_lower = text.lower()
    detected_spam = [kw for kw in spam_keywords if kw in text_lower]
    if detected_spam:
        return False, "spam", f"Spam keywords detected: {', '.join(detected_spam)}"
    
    if 'http://' in text or 'https://' in text or 'www.' in text:
        return False, "suspicious", "URLs detected - please remove links"
    
    if text.isupper() and len(text) > 50:
        return False, "warning", "Excessive capitalization detected"
    
    absurd_patterns = [
        'floor is lava', 'aliens', 'unicorn', 'dragon', 'magic', 'wizard',
        'flying', 'teleport', 'time travel', 
    ]
    if any(pattern in text_lower for pattern in absurd_patterns):
        return False, "absurd", "Unrealistic content detected - please describe a real issue"
    
    return True, "valid", "Input validated successfully"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s']", '', text)
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    important_words = {'not', 'no', 'need', 'urgent', 'very', 'extremely', 'asap', 'immediately'}
    stop_words = stop_words - important_words
    tokens = [w for w in tokens if w not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return ' '.join(tokens)

def get_keyword_fallbacks(text_lower, categories):
    """Enhanced keyword fallbacks for missing vocabulary"""
    fallback_issues = np.zeros(len(categories))
    
    # Electrical keywords - EXPANDED
    electrical_keywords = [
        'light', 'lights', 'bulb', 'bulbs', 'lamp', 'lamps', 
        'electric', 'electricity', 'power', 'outlet', 'switch',
        'ac', 'air conditioning', 'heater', 'heating', 'cooling',
        'fan', 'fans', 'ventilation', 'circuit', 'fuse', 'wiring',
        'flickering', 'giving heat', 'giving off heat', 'not cooling'
    ]
    if any(kw in text_lower for kw in electrical_keywords):
        if 'electrical_issue' in categories:
            idx = categories.index('electrical_issue')
            fallback_issues[idx] = 0.7
    
    # Noise keywords - EXPANDED
    noise_keywords = [
        'noisy', 'noise', 'loud', 'neighbor', 'neighbors', 'lobbymate', 
        'lobbymates', 'roommate', 'roommates', 'shouting', 'yelling',
        'music', 'party', 'disturbing', 'disturbance', 'too noisy',
        'making noise', 'sound', 'sounds'
    ]
    if any(kw in text_lower for kw in noise_keywords):
        if 'noise_issue' in categories:
            idx = categories.index('noise_issue')
            fallback_issues[idx] = 0.7
    
    # Cleanliness keywords
    cleanliness_keywords = [
        'mess', 'messy', 'dirty', 'filthy', 'unclean', 'not clean',
        'untidy', 'cluttered', 'trash', 'garbage', 'rubbish',
        'not being cleaned', 'cleaning', 'cleaned'
    ]
    if any(kw in text_lower for kw in cleanliness_keywords):
        if 'cleanliness_issue' in categories:
            idx = categories.index('cleanliness_issue')
            fallback_issues[idx] = 0.6
    
    return fallback_issues

def predict_ml(complaint_text, models):
    processed = preprocess_text(complaint_text)
    X = models['tfidf'].transform([processed])
    
    issue_proba = models['issue_ml'].predict_proba(X)
    
    issue_confidences = []
    
    # Check if output is a list (standard MultiOutput/OneVsRest behavior)
    if isinstance(issue_proba, list):
        for proba in issue_proba:
            proba_array = np.array(proba)
            try:
                if len(proba_array.shape) == 2 and proba_array.shape[1] > 1:
                    issue_confidences.append(float(proba_array[0][1]))
                else:
                    issue_confidences.append(float(proba_array.flat[0]))
            except:
                issue_confidences.append(0.0)
    # Handle case where output is a single array (e.g. ClassifierChain, specialized multilabel)
    # shape: (n_samples, n_classes) -> we want probas for n_classes
    elif hasattr(issue_proba, 'shape') or isinstance(issue_proba, np.ndarray):
        proba_array = np.array(issue_proba)
        # Assuming proba_array is (1, n_classes) and contains positive class probabilities
        if len(proba_array.shape) >= 2:
            issue_confidences = proba_array[0].tolist()
        else:
             issue_confidences = proba_array.tolist()
    
    issue_confidences = np.array(issue_confidences)
    
    text_lower = complaint_text.lower()
    fallback_issues = get_keyword_fallbacks(text_lower, models['categories'])
    
    for i in range(len(issue_confidences)):
        if issue_confidences[i] < 0.2 and fallback_issues[i] > 0:
            issue_confidences[i] = max(issue_confidences[i], fallback_issues[i] * 0.8)
    
    issue_pred = (issue_confidences > 0.3).astype(int)
    
    urgency_pred = models['urgency_ml'].predict(X)[0]
    urgency_proba = models['urgency_ml'].predict_proba(X)[0]
    
    return {
        'issues': issue_pred,
        'issue_confidences': issue_confidences.tolist(),
        'urgency': urgency_pred,
        'urgency_proba': urgency_proba,
        'urgency_confidence': urgency_proba.max(),
        'processed_text': processed
    }

def predict_lstm(complaint_text, models):
    MAX_SEQUENCE_LENGTH = 100
    
    processed = preprocess_text(complaint_text)
    sequence = models['tokenizer'].texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, 
                          padding='post', truncating='post')
    
    issue_proba = models['issue_lstm'].predict(padded, verbose=0)[0]
    
    text_lower = complaint_text.lower()
    fallback_issues = get_keyword_fallbacks(text_lower, models['categories'])
    
    for i in range(len(issue_proba)):
        if issue_proba[i] < 0.2 and fallback_issues[i] > 0:
            issue_proba[i] = max(issue_proba[i], fallback_issues[i] * 0.8)
    
    issue_pred = (issue_proba > 0.5).astype(int)
    
    urgency_proba = models['urgency_lstm'].predict(padded, verbose=0)[0]
    urgency_pred = urgency_proba.argmax()
    
    return {
        'issues': issue_pred,
        'issue_confidences': issue_proba,
        'urgency': urgency_pred,
        'urgency_proba': urgency_proba,
        'urgency_confidence': urgency_proba.max(),
        'processed_text': processed
    }

# ============================================================================
# PER-ISSUE ROUTING LOGIC (KEY FIX)
# ============================================================================

def get_per_issue_routing(categories, confidences):
    """
    Per-issue routing decisions - CORE FIX
    
    Each issue is evaluated independently based on its own confidence.
    No more "highest wins" - prevents misrouting of multi-issue complaints.
    
    Thresholds (raised for safety with synthetic data):
    - Auto-route: >= 85% (was 80%)
    - Human review: 65-84% (was 60-80%)
    - Ignore: < 65% (was <60%)
    """
    decisions = []
    
    for i, category in enumerate(categories):
        conf = confidences[i]
        
        if conf >= 0.85:
            # High confidence - auto-create task
            decisions.append({
                'category': category,
                'confidence': conf,
                'action': 'AUTO-ROUTE',
                'status': 'auto',
                'team': get_team_name(category),
                'description': f'High confidence - auto-create task for {get_team_name(category)}'
            })
        elif 0.65 <= conf < 0.85:
            # Medium confidence - flag for review
            decisions.append({
                'category': category,
                'confidence': conf,
                'action': 'HUMAN REVIEW',
                'status': 'review',
                'team': 'Triage Staff',
                'description': f'Medium confidence - flag for triage staff to verify'
            })
        # Below 65%: implicitly ignored (no decision created)
    
    return decisions

def get_team_name(category):
    """Map issue category to maintenance team"""
    team_mapping = {
        'electrical_issue': 'Electrical Team',
        'internet_issue': 'IT/Network Team',
        'plumbing_issue': 'Plumbing Team',
        'furniture_issue': 'Furniture/Maintenance Team',
        'cleanliness_issue': 'Housekeeping Team',
        'food_issue': 'Mess Management',
        'noise_issue': 'Hostel Administration',
        'billing_issue': 'Accounts Department'
    }
    return team_mapping.get(category, 'General Maintenance')
    


# ============================================================================
# MAIN APP
# ============================================================================

with st.spinner('Loading models...'):
    models = load_models()

# Header
st.title("Hostel Complaint Triage System")
st.markdown("Automated classification of maintenance complaints using Natural Language Processing and Machine Learning. The system analyzes complaint text to identify issue types, assess urgency, and recommend routing decisions.")

st.markdown("---")

# Configuration
st.markdown("## Configuration")
selected_model = st.radio(
    "Select classification model",
    ["Machine Learning (Logistic Regression + Naive Bayes)", "Deep Learning (LSTM Neural Network)"],
    label_visibility="collapsed"
)

st.markdown("---")

# Input
st.markdown("## Complaint Input")
complaint_text = st.text_area(
    "Enter maintenance complaint",
    height=150,
    placeholder="Example: The lights are flickering in my room and the AC is giving off heat instead of cooling. Also the wifi connection keeps dropping.",
    label_visibility="collapsed"
)

# Analyze button
if st.button("Analyze Complaint", type="primary"):
    if complaint_text:
        # Validate
        is_valid, status, message = validate_complaint(complaint_text)
        
        if status == "valid":
            st.success(f"✓ {message}")
        elif status in ["warning", "error", "spam", "suspicious", "absurd"]:
            if status == "warning":
                st.warning(f"⚠ {message}")
            else:
                st.error(f"✗ {message}")
            
            with st.expander("Why was this flagged?"):
                st.markdown(f"**Detection:** {message}")
                st.markdown("The validation layer catches spam, gibberish, and unrealistic content before classification. Flagged complaints are routed to staff for manual review.")
            st.stop()
        
        # Predict
        with st.spinner('Analyzing complaint...'):
            if "Machine Learning" in selected_model:
                results = predict_ml(complaint_text, models)
            else:
                results = predict_lstm(complaint_text, models)
        
        st.markdown("---")
        st.markdown("## Classification Results")
        
        # Get data
        issues_array = np.array(results['issues'])
        confidences_array = np.array(results['issue_confidences'])
        predicted_urgency = models['urgency_labels'][results['urgency']]
        urgency_confidence = results['urgency_confidence']
        
        # Get urgency confidence badge
        if urgency_confidence > 0.75:
            urgency_badge = '<span class="badge badge-high">HIGH CONFIDENCE</span>'
        elif urgency_confidence > 0.50:
            urgency_badge = '<span class="badge badge-medium">MEDIUM CONFIDENCE</span>'
        else:
            urgency_badge = '<span class="badge badge-low">LOW CONFIDENCE</span>'
        
        # Summary Card
        issues_found = []
        for i, category in enumerate(models['categories']):
            if i < len(issues_array) and issues_array[i] == 1:
                issues_found.append(category.replace('_', ' ').title())
        
        issues_text = ", ".join(issues_found) if issues_found else "None detected"
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

        st.markdown(f"""<div class="admin-card">
<div class="admin-card-header">Overview</div>
<div class="admin-row">
<span class="admin-label">Detected Issues</span>
<span class="admin-value">{issues_text}</span>
</div>
<div class="admin-row">
<span class="admin-label">Urgency Level</span>
<span class="admin-value">{predicted_urgency.upper()}</span>
</div>
<div class="admin-row">
<span class="admin-label">Urgency Confidence</span>
{urgency_badge}
</div>
</div>""", unsafe_allow_html=True)
        
    # Per-Issue Routing Decisions
    st.markdown("### Routing Decisions")
    st.caption("Each issue is evaluated independently — prevents misrouting of multi-issue complaints")

    routing_decisions = get_per_issue_routing(models['categories'], confidences_array)

    if not routing_decisions:
        st.warning("No issues detected with sufficient confidence (≥65%). Full manual review required.")
    else:
        for decision in routing_decisions:
            status_class = f"routing-{decision['status']}"

            conf_badge = (
                '<span class="badge badge-high">HIGH</span>'
                if decision['confidence'] >= 0.85
                else '<span class="badge badge-medium">MEDIUM</span>'
            )

            st.markdown(f"""
            <div class="routing-card {status_class}">
                <div>
                    <div class="routing-title">
                        {decision['category'].replace('_', ' ').title()}
                    </div>
                    <div class="routing-desc">
                        {decision['description']}
                    </div>
                    <div style="font-size:0.8125rem; color:#2a2a2a; margin-top:0.375rem;">
                        Assigned to: <strong>{decision['team']}</strong>
                    </div>
                </div>
                <div class="routing-badge-container">
                    <div style="font-weight:600; font-size:0.875rem;">
                        {decision['action']}
                    </div>
                    {conf_badge}
                    <div style="font-size:0.75rem; color:#2a2a2a; margin-top:0.25rem;">
                        {decision['confidence']:.0%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        
        # All Detected Issues (for reference)
        st.markdown("### All Detected Issues")
        st.caption("Full classification results with confidence scores")
        
        for i, category in enumerate(models['categories']):
            if i < len(confidences_array):
                conf = float(confidences_array[i])
                
                # Show all detected issues (above detection threshold 0.3)
                # This aligns with the "Overview" card
                if conf >= 0.30:
                    if conf >= 0.85:
                        conf_label = "High" 
                        conf_badge_class = "badge-high"
                    elif conf >= 0.65:
                        conf_label = "Medium"
                        conf_badge_class = "badge-medium"
                    else:
                        conf_label = "Low (Ignored)"
                        conf_badge_class = "badge-low"
                    
                    st.markdown(f"""
                    <div class="admin-card" style="padding: 0.75rem 1rem; margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600; color: #000000;">{category.replace('_', ' ').title()}</span>
                            <div>
                                <span class="badge {conf_badge_class}">{conf_label}</span>
                                <span style="font-size: 0.8125rem; color: #2a2a2a; margin-left: 0.5rem;">{conf:.0%}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.caption("Note: Issues with confidence <65% are technically detected but ignored for routing to prevent false positives.")
        
        # Technical Details
        with st.expander("View technical details"):
            st.markdown("**Processing Pipeline**")
            st.code(f"""1. Input validation (length, spam, absurdity checks)
2. Text normalization (lowercase, special character removal)
3. Tokenization: {len(results['processed_text'].split())} tokens extracted
4. Lemmatization and stopword filtering
5. Feature extraction: {'TF-IDF vectorization' if 'Machine Learning' in selected_model else 'Word embeddings'}
6. Multi-label classification
7. Per-issue routing decisions""")
            
            st.markdown("**Processed Text**")
            st.code(results['processed_text'])