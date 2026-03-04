import os
import pickle
import json
import numpy as np
import re
import warnings
from typing import Dict, Any, Tuple, List

# Suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Note: Heavy imports (Tensorflow, NLTK) moved inside functions for memory efficiency (Zero-RAM strategy)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MAX_SEQUENCE_LENGTH = 100

# Global artifacts cache
_ml_artifacts = None
_dl_artifacts = None

def download_nltk_data():
    """Ensure required NLTK data is available"""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)

def load_ml_artifacts() -> Dict[str, Any]:
    """Load and cache lightweight ML artifacts (TF-IDF, LR/NB, Mappings)"""
    global _ml_artifacts
    if _ml_artifacts is not None:
        return _ml_artifacts

    download_nltk_data()
    
    print("Loading ML artifacts...")
    try:
        # Load mappings first
        with open(os.path.join(DATA_DIR, 'category_mapping.json'), 'r') as f:
            mappings = json.load(f)

        # Load ML specific artifacts
        with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        with open(os.path.join(MODELS_DIR, 'issue_classifier_ml.pkl'), 'rb') as f:
            issue_classifier_ml = pickle.load(f)

        with open(os.path.join(MODELS_DIR, 'urgency_classifier_ml.pkl'), 'rb') as f:
            urgency_classifier_ml = pickle.load(f)

        _ml_artifacts = {
            'tfidf': tfidf_vectorizer,
            'issue_ml': issue_classifier_ml,
            'urgency_ml': urgency_classifier_ml,
            'categories': mappings['categories'],
            'urgency_labels': ['low', 'medium', 'high']
        }
        print("ML artifacts loaded.")
        return _ml_artifacts
    except Exception as e:
        print(f"Error loading ML artifacts: {e}")
        raise e

def load_dl_artifacts() -> Dict[str, Any]:
    """Load and cache heavy DL models (LSTMs, Tokenizer) lazily"""
    global _dl_artifacts
    if _dl_artifacts is not None:
        return _dl_artifacts

    # Lazy imports for Tensorflow/Keras
    from tensorflow.keras.models import load_model as keras_load_model
    
    print("Loading DL models (Lazy Load)...")
    try:
        # Load DL specific artifacts
        issue_classifier_lstm = keras_load_model(os.path.join(MODELS_DIR, 'issue_classifier_lstm.keras'))
        urgency_classifier_lstm = keras_load_model(os.path.join(MODELS_DIR, 'urgency_classifier_lstm.keras'))

        with open(os.path.join(MODELS_DIR, 'tokenizer_dl.pkl'), 'rb') as f:
            tokenizer = pickle.load(f)

        _dl_artifacts = {
            'issue_lstm': issue_classifier_lstm,
            'urgency_lstm': urgency_classifier_lstm,
            'tokenizer': tokenizer
        }
        print("DL models loaded.")
        return _dl_artifacts
    except Exception as e:
        print(f"Error loading DL models: {e}")
        raise e

def load_models() -> Dict[str, Any]:
    """Deprecated: Use load_ml_artifacts or load_dl_artifacts instead.
    Provided for backward compatibility while refactoring.
    """
    ml = load_ml_artifacts()
    try:
        dl = load_dl_artifacts()
        return {**ml, **dl}
    except:
        return ml

def preprocess_text(text: str) -> str:
    """Clean and preprocess text for ML usage"""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
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

def validate_complaint(text: str) -> Tuple[bool, str, str]:
    """Validate input text quality"""
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
        'flying', 'teleport', 'time travel'
    ]
    if any(pattern in text_lower for pattern in absurd_patterns):
        return False, "absurd", "Unrealistic content detected"
    
    return True, "valid", "Valid"

def get_keyword_fallbacks(text_lower: str, categories: List[str]) -> np.ndarray:
    """Heuristic fallback for specific keywords"""
    fallback_issues = np.zeros(len(categories))
    
    # Electrical
    electrical_keywords = [
        'light', 'lights', 'bulb', 'bulbs', 'lamp', 'lamps', 
        'electric', 'electricity', 'power', 'outlet', 'switch',
        'ac', 'air conditioning', 'heater', 'heating', 'cooling',
        'fan', 'fans', 'ventilation', 'circuit', 'fuse', 'wiring',
        'flickering'
    ]
    if any(kw in text_lower for kw in electrical_keywords):
        if 'electrical_issue' in categories:
            fallback_issues[categories.index('electrical_issue')] = 0.8
    
    # Noise
    noise_keywords = [
        'noisy', 'noise', 'loud', 'neighbor', 'music', 'party', 
        'disturbing', 'shouting', 'yelling'
    ]
    if any(kw in text_lower for kw in noise_keywords):
        if 'noise_issue' in categories:
            fallback_issues[categories.index('noise_issue')] = 0.8
            
    # Cleanliness
    clean_keywords = ['mess', 'dirty', 'filthy', 'unclean', 'trash', 'garbage', 'rubbish', 'cleaning']
    if any(kw in text_lower for kw in clean_keywords):
        if 'cleanliness_issue' in categories:
            fallback_issues[categories.index('cleanliness_issue')] = 0.75
            
    # Furniture
    furniture_keywords = ['desk', 'chair', 'bed', 'furniture', 'table', 'wardrobe', 'cupboard', 'shelf', 'broken', 'damaged']
    if any(kw in text_lower for kw in furniture_keywords):
        if 'furniture_issue' in categories:
            fallback_issues[categories.index('furniture_issue')] = 0.8
            
    return fallback_issues

def get_team_name(category: str) -> str:
    """Map category to team"""
    team_mapping = {
        'electrical_issue': 'Electrical Team',
        'internet_issue': 'IT / Network Team',
        'plumbing_issue': 'Plumbing Team',
        'furniture_issue': 'Furniture & Infrastructure Team',
        'cleanliness_issue': 'Housekeeping Team',
        'food_issue': 'Mess Management',
        'noise_issue': 'Hostel Administration',
        'billing_issue': 'Accounts Department',
        'other_issue': 'Hostel Administration'
    }

    return team_mapping.get(category, 'Hostel Administration')

def get_all_teams() -> List[str]:
    """Return list of all unique teams"""
    return [
        'Electrical Team',
        'Plumbing Team', 
        'IT / Network Team',
        'Furniture & Infrastructure Team',
        'Housekeeping Team',
        'Mess Management',
        'Hostel Administration',
        'Accounts Department'
    ]

def predict_ml(text: str, models: Dict) -> Dict:
    """Run prediction using ML (TF-IDF + LR/NB) models"""
    processed = preprocess_text(text)
    X = models['tfidf'].transform([processed])
    
    # Issue Prediction
    issue_proba = models['issue_ml'].predict_proba(X)
    
    issue_confidences = []
    # Handle different sklearn outputs (list of arrays vs single array)
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
    elif hasattr(issue_proba, 'shape'):
        proba_array = np.array(issue_proba)
        # Check if multilabel output (n_samples, n_classes)
        if len(proba_array.shape) >= 2:
             issue_confidences = proba_array[0].tolist()
        else:
             issue_confidences = proba_array.tolist()
             
    issue_confidences = np.array(issue_confidences)
    
    # Keyword Fallback
    text_lower = text.lower()
    fallback_issues = get_keyword_fallbacks(text_lower, models['categories'])
    
    for i in range(len(issue_confidences)):
        if issue_confidences[i] < 0.2 and fallback_issues[i] > 0:
            issue_confidences[i] = max(issue_confidences[i], fallback_issues[i] * 0.8)
            
    issue_pred_indices = np.where(issue_confidences > 0.3)[0]
    
    # Urgency Prediction
    urgency_idx = models['urgency_ml'].predict(X)[0]
    urgency_proba = models['urgency_ml'].predict_proba(X)[0]
    
    return {
        'issue_confidences': issue_confidences,
        'issue_indices': issue_pred_indices,
        'urgency_idx': urgency_idx,
        'urgency_conf': urgency_proba.max(),
        'urgency_level': models['urgency_labels'][urgency_idx]
    }

def predict_lstm(text: str, models: Dict) -> Dict:
    """Run prediction using Deep Learning (LSTM) models"""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    processed = preprocess_text(text)
    sequence = models['tokenizer'].texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    # Issue Prediction
    issue_proba = models['issue_lstm'].predict(padded, verbose=0)[0]
    
    # Keyword Fallback
    text_lower = text.lower()
    fallback_issues = get_keyword_fallbacks(text_lower, models['categories'])
    
    for i in range(len(issue_proba)):
        if issue_proba[i] < 0.2 and fallback_issues[i] > 0:
            issue_proba[i] = max(issue_proba[i], fallback_issues[i] * 0.8)
            
    issue_pred_indices = np.where(issue_proba > 0.5)[0]
    
    # Urgency Prediction
    urgency_proba = models['urgency_lstm'].predict(padded, verbose=0)[0]
    urgency_idx = urgency_proba.argmax()
    
    return {
        'issue_confidences': issue_proba,
        'issue_indices': issue_pred_indices,
        'urgency_idx': urgency_idx,
        'urgency_conf': urgency_proba.max(),
        'urgency_level': models['urgency_labels'][urgency_idx]
    }

def route_complaint(text: str, model_type: str = 'ML') -> Dict[str, Any]:
    """
    Main function to process a complaint.
    
    Args:
        text: The complaint text
        model_type: 'ML' or 'DL' (Deep Learning)
        
    Returns:
        Dictionary with all routing outcome details
    """
    ml_models = load_ml_artifacts()
    
    # 1. Validation
    is_valid, status, message = validate_complaint(text)
    if not is_valid:
        return {
            "error": True,
            "message": message,
            "status": status
        }
        
    # 2. Prediction
    if model_type == 'DL':
        dl_models = load_dl_artifacts()
        # Merge for the prediction functions
        combined = {**ml_models, **dl_models}
        res = predict_lstm(text, combined)
    else:
        res = predict_ml(text, ml_models)
        
    # 3. Routing Logic
    categories = ml_models['categories']
    issue_confidences = res['issue_confidences']
    
    detected_issues = []
    primary_decision = {
        "action": "IGNORED",
        "status": "ignored",
        "team": "None",
        "confidence": 0.0
    }
    
    # Collect all detected issues
    confidence_map = {}
    
    # We will pick the "most confident" issue to drive the primary routing decision
    max_conf = 0.0
    best_category = None
    
    for i, conf in enumerate(issue_confidences):
        cat_name = categories[i]
        confidence_map[cat_name] = float(conf)
        
        # Consider it detected if > 30%
        if conf > 0.30:
            detected_issues.append(cat_name)
            
            if conf > max_conf:
                max_conf = conf
                best_category = cat_name

    # Determine Routing based on best category
    routing_desc = ""
    target_team = "None"
    routing_action = "IGNORED" # default
    
    if best_category:
        target_team = get_team_name(best_category)
        
        if max_conf >= 0.75:
            routing_action = "AUTO-ROUTE"
            routing_desc = f"High confidence ({max_conf:.0%}) - Auto-routed to {target_team}"
        elif max_conf >= 0.55:
            routing_action = "HUMAN REVIEW"
            routing_desc = f"Medium confidence ({max_conf:.0%}) - Flagged for review"
        else:
            routing_action = "IGNORED" 
            routing_desc = f"Low confidence ({max_conf:.0%}) - Ignored to prevent false positive"
            target_team = "None" # Reset team if ignored
            
    # Construct Result
    return {
        "error": False,
        "student_text": text,
        "predicted_issues": detected_issues,
        "predicted_urgency": res['urgency_level'],
        "urgency_confidence": float(res['urgency_conf']),
        "confidence_scores": confidence_map,
        "routing_decision": routing_action,
        "routing_description": routing_desc,
        "routed_team": target_team,
        "model_used": model_type
    }
