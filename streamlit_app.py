"""
HOSTEL COMPLAINT TRIAGE SYSTEM - ADMIN DASHBOARD
=================================================
Professional admin interface for complaint classification

Author: Vedansh
Date: December 2025
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

# Custom CSS - FIXED VERSION
CUSTOM_CSS = """
<style>
    /* Import System Fonts */
    * {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", 
                     "Helvetica Neue", Arial, sans-serif;
    }
    
    /* FORCE GREY BACKGROUND EVERYWHERE */
    html, body {
        background-color: #f5f5f5 !important;
    }
    
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"],
    .main,
    .stApp {
        background-color: #f5f5f5 !important;
    }
    
    /* Page Container */
    .block-container {
        padding: 2rem 3rem;
        max-width: 1100px;
        background-color: transparent !important;
    }
    
    /* Headings - Black Text */
    h1 {
        font-size: 1.75rem;
        font-weight: 600;
        color: #000000 !important;
        margin-bottom: 0.75rem;
    }
    
    h2 {
        font-size: 1.125rem;
        font-weight: 600;
        color: #000000 !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 0.9375rem;
        font-weight: 600;
        color: #000000 !important;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Paragraph Text */
    p {
        color: #2a2a2a !important;
        font-size: 0.9375rem;
        line-height: 1.6;
    }
    
    /* Text Area */
    .stTextArea textarea {
        font-size: 0.9375rem;
        color: #1a1a1a !important;
        border: 1px solid #bdbdbd !important;
        border-radius: 3px;
        padding: 0.75rem;
        background-color: #ffffff !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #4a90e2 !important;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
    }
    
    /* Radio Buttons */
    .stRadio > label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #000000 !important;
    }
    
    .stRadio div[role="radiogroup"] {
        display: flex;
        gap: 0.75rem;
    }
    
    .stRadio div[role="radiogroup"] > label {
        background-color: #ffffff !important;
        border: 1px solid #d0d0d0 !important;
        border-radius: 3px;
        padding: 0.625rem 1rem;
        font-size: 0.875rem;
        color: #2a2a2a !important;
    }
    
    .stRadio div[role="radiogroup"] > label:hover {
        background-color: #f5f5f5 !important;
        border-color: #b0b0b0 !important;
    }
    
    .stRadio div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #4a90e2 !important;
        color: #ffffff !important;
        border-color: #4a90e2 !important;
        font-weight: 500;
    }
    
    /* Button */
    .stButton button {
        background-color: #4a90e2 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 3px;
        padding: 0.625rem 1.5rem;
        font-size: 0.9375rem;
        font-weight: 500;
        width: 100%;
    }
    
    .stButton button:hover {
        background-color: #357abd !important;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 3px;
        border-left: 3px solid !important;
        padding: 0.75rem 1rem;
        font-size: 0.9375rem;
    }
    
    /* Success Alert */
    .stSuccess {
        background-color: #f0f9f4 !important;
        border-left-color: #48a868 !important;
        color: #2d6a3e !important;
    }
    
    /* Warning Alert */
    .stWarning {
        background-color: #fef9f0 !important;
        border-left-color: #e8a02a !important;
        color: #8a5e1a !important;
    }
    
    /* Error Alert */
    .stError {
        background-color: #fef2f2 !important;
        border-left-color: #d93025 !important;
        color: #8a1e1a !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 3px;
        padding: 0.625rem 0.875rem;
        font-size: 0.875rem;
        font-weight: 500;
        color: #404040 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #f5f5f5 !important;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e0e0e0 !important;
        border-top: none !important;
        padding: 0.875rem;
        background-color: #ffffff !important;
    }
    
    /* Code Blocks */
    code {
        font-family: "SF Mono", Monaco, monospace;
        font-size: 0.8125rem;
        background-color: #f5f5f5 !important;
        color: #1a1a1a !important;
        padding: 0.125rem 0.375rem;
        border-radius: 2px;
    }
    
    .stCodeBlock {
        border: 1px solid #e0e0e0 !important;
        border-radius: 3px;
        background-color: #ffffff !important;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #d0d0d0;
    }
    
    /* Caption */
    .stCaption {
        font-size: 0.8125rem;
        color: #707070 !important;
        font-style: italic;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""

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
        'flying', 'teleport', 'time travel', 'stealing my socks'
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
    fallback_issues = np.zeros(len(categories))
    
    electrical_keywords = ['light', 'lights', 'bulb', 'bulbs', 'lamp', 'lamps', 
                          'electric', 'electricity', 'power', 'outlet', 'switch',
                          'ac', 'air conditioning', 'heater', 'heating', 'fan not working',
                          'circuit', 'fuse', 'wiring']
    if any(kw in text_lower for kw in electrical_keywords):
        if 'electrical_issue' in categories:
            idx = categories.index('electrical_issue')
            fallback_issues[idx] = 0.7
    
    noise_keywords = ['noisy', 'noise', 'loud', 'neighbor', 'neighbors', 'lobbymate', 
                     'lobbymates', 'roommate', 'roommates', 'shouting', 'yelling',
                     'music', 'party', 'disturbing', 'disturbance']
    if any(kw in text_lower for kw in noise_keywords):
        if 'noise_issue' in categories:
            idx = categories.index('noise_issue')
            fallback_issues[idx] = 0.7
    
    cleanliness_keywords = ['mess', 'messy', 'dirty', 'filthy', 'unclean', 'not clean',
                           'untidy', 'cluttered', 'trash', 'garbage', 'rubbish']
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
    for proba in issue_proba:
        proba_array = np.array(proba)
        try:
            if len(proba_array.shape) == 2 and proba_array.shape[1] > 1:
                issue_confidences.append(float(proba_array[0][1]))
            else:
                issue_confidences.append(float(proba_array.flat[0]))
        except:
            issue_confidences.append(0.0)
    
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

#==============================================================================
# MAIN APP
#==============================================================================

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
    placeholder="Example: The wifi in my room has been down for two days and I have an exam tomorrow. Please fix this as soon as possible.",
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
        
        # Get data - ensure arrays are properly converted and handle shape issues
        issues_array = np.array(results['issues']).flatten()
        confidences_array = np.array(results['issue_confidences']).flatten()
        predicted_urgency = models['urgency_labels'][results['urgency']]
        urgency_confidence = results['urgency_confidence']
        max_confidence = max(confidences_array) if len(confidences_array) > 0 else 0.0
        
        # Collect issue categories - with proper bounds checking
        issues_found = []
        for i, category in enumerate(models['categories']):
            if i < len(issues_array) and int(issues_array[i]) == 1:
                issues_found.append(category.replace('_', ' ').title())
        
        issues_text = ", ".join(issues_found) if issues_found else "None detected"
        
        # Determine routing
        if max_confidence > 0.80:
            routing_action = "AUTO-ROUTE"
            routing_bg = "#f0f9f4"
            routing_border = "#48a868"
            routing_text = "#2d6a3e"
            routing_desc = "High confidence — route to maintenance automatically"
        elif max_confidence > 0.60:
            routing_action = "HUMAN REVIEW"
            routing_bg = "#fef9f0"
            routing_border = "#e8a02a"
            routing_text = "#8a5e1a"
            routing_desc = "Medium confidence — flag for staff verification"
        else:
            routing_action = "MANUAL ASSIGNMENT"
            routing_bg = "#fef2f2"
            routing_border = "#d93025"
            routing_text = "#8a1e1a"
            routing_desc = "Low confidence — requires full human assessment"
        
        # Get confidence label
        if urgency_confidence > 0.75:
            conf_label = "High"
            conf_color = "#2e7d32"
            conf_bg = "#e8f5e9"
        elif urgency_confidence > 0.50:
            conf_label = "Medium"
            conf_color = "#e65100"
            conf_bg = "#fff3e0"
        else:
            conf_label = "Low"
            conf_color = "#c62828"
            conf_bg = "#ffebee"
        
        # Use Streamlit containers and columns instead of raw HTML
        with st.container():
            st.markdown("### Decision Summary")
            
            # Issue Type
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown('<div style="color:#6b7280; font-size:0.875rem;">Issue Type</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="font-weight:600; color:#000000; font-size:0.9375rem;">{issues_text}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Urgency
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown('<div style="color:#6b7280; font-size:0.875rem;">Urgency</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="font-weight:600; color:#000000; font-size:0.9375rem; text-transform:uppercase;">{predicted_urgency}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Confidence
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown('<div style="color:#6b7280; font-size:0.875rem;">Confidence</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<span style="display:inline-block; padding:0.25rem 0.75rem; border-radius:3px; font-size:0.75rem; font-weight:600; background:{conf_bg}; color:{conf_color};">{conf_label.upper()} CONFIDENCE</span>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Routing Decision
            st.markdown(f"""
            <div style="padding:1rem; background:{routing_bg}; border:1.5px solid {routing_border}; border-radius:4px; margin-top:0.5rem;">
                <div style="font-weight:600; font-size:0.9375rem; color:{routing_text}; margin-bottom:0.25rem;">{routing_action}</div>
                <div style="font-size:0.875rem; color:{routing_text}; margin-bottom:0.5rem;">{routing_desc}</div>
                <div style="font-size:0.8125rem; color:{routing_text}; opacity:0.8;">Classification confidence: {max_confidence:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Issue Categories List - using Streamlit columns for better control
        if issues_found:
            st.markdown("### Predicted Issue Categories")
            for i, category in enumerate(models['categories']):
                if i < len(issues_array) and int(issues_array[i]) == 1:
                    conf = float(confidences_array[i])
                    conf_label = "High" if conf > 0.75 else ("Medium" if conf > 0.50 else "Low")
                    
                    # Use columns instead of HTML for reliability
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{category.replace('_', ' ').title()}**")
                    with col2:
                        st.markdown(f'<div style="text-align:right; color:#000000; font-weight:600; font-size:0.875rem;">{conf_label} ({conf:.0%})</div>', unsafe_allow_html=True)
        
        st.caption("Note: Urgency is predicted based on linguistic patterns, not objective severity")
        
        # Technical Details - Hidden by default
        with st.expander("Technical details (for review only)"):
            st.markdown("""
            <ul style="margin:0.5rem 0; padding-left:1.5rem;">
                <li>Input validation (length, spam, absurdity checks)</li>
                <li>Text normalization (lowercase, special character removal)</li>
                <li>Tokenization: """ + str(len(results['processed_text'].split())) + """ tokens extracted</li>
                <li>Lemmatization and stopword filtering</li>
                <li>Feature extraction: """ + ('TF-IDF vectorization' if 'Machine Learning' in selected_model else 'Word embeddings') + """</li>
                <li>Classification</li>
                <li>Confidence scoring and routing decision</li>
            </ul>
            """, unsafe_allow_html=True)
            
            st.markdown("**Processed Text**")
            st.code(results['processed_text'])
            
            st.markdown("**Confidence Thresholds**")
            st.markdown("""
            <ul style="margin:0.5rem 0; padding-left:1.5rem;">
                <li>Auto-route: >80% confidence</li>
                <li>Human review: 60-80% confidence</li>
                <li>Manual assignment: <60% confidence</li>
            </ul>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a complaint description to analyze")

# Footer
st.markdown("---")
st.caption("Classification Model: ML (Logistic Regression + Naive Bayes) | DL (LSTM) • Trained on synthetic data • Linguistic urgency prediction • Confidence-based routing")