import joblib
import os
from utils import clean_text, assign_priority

MODELS_DIR = "models"
CATEGORY_MODEL_PATH = os.path.join(MODELS_DIR, "category_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

# Load artifacts globally
try:
    print(" Loading category model...")
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(" Model loaded successfully.")
except FileNotFoundError:
    print(" Models not found. Please run train.py first.")
    category_model = None
    vectorizer = None

def predict_ticket(text):
    
    if not category_model:
        return {"error": "Model not loaded"}
    
    # Preprocess
    cleaned_text = clean_text(text)
    
    # Vectorize
    vec_text = vectorizer.transform([cleaned_text])
    
    # Predict
    category_pred = category_model.predict(vec_text)[0]
    
    # Assign Priority
    priority_level = assign_priority(text, category_pred)
    
    return {
        "category": category_pred,
        "priority": priority_level
    }

if __name__ == "__main__":
    # Test Prediction
    sample_text = "My mouse is not working on my laptop."
    print(f" Sample Ticket: {sample_text}")
    result = predict_ticket(sample_text)
    print(f" Prediction: {result}")
