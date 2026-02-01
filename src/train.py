import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/processed/clean_tickets.csv"
MODELS_DIR = "models"
PRIORITY_MODEL_PATH = os.path.join(MODELS_DIR, "priority_model.pkl")

def train_model():
   
    print(" Starting model training (Category Only)...")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f" Processed data not found at {DATA_PATH}. Run preprocessing.py first.")

   
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=["clean_text"], inplace=True)
    
    print(f" Training on {len(df)} samples")

    X = df["clean_text"]
    y_category = df["Ticket Type"]

    # TF-IDF Vectorization
    print(" Vectorizing text...")
    # Increased max_features to capture important niche terms like 'vpn'
    vectorizer = TfidfVectorizer(max_features=10000)
    X_vec = vectorizer.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_cat_train, y_cat_test = train_test_split(
        X_vec, y_category, test_size=0.2, random_state=42
    )

    # --- Train Category Model ---
    print(" Training Ticket Type Model...")
    category_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    category_model.fit(X_train, y_cat_train)
    
    cat_pred = category_model.predict(X_test)
    cat_acc = accuracy_score(y_cat_test, cat_pred)
    print(f" Ticket Type Accuracy: {cat_acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_cat_test, cat_pred))

    # Save Models
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(category_model, os.path.join(MODELS_DIR, "category_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

    # Cleanup old priority model if it exists
    if os.path.exists(PRIORITY_MODEL_PATH):
        print(" Removing unused priority model...")
        os.remove(PRIORITY_MODEL_PATH)

    print(f" Model saved to {MODELS_DIR}/")

if __name__ == "__main__":
    train_model()
