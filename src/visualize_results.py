import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


DATA_PATH = "data/processed/clean_tickets.csv"
MODEL_PATH = "models/category_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
OUTPUT_PATH = "models/confusion_matrix.png"

def generate_visualization():
    print("📊 Generating Model Performance Visualization...")
    
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Data file not found.")
        
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=["clean_text"], inplace=True)
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Predict
    X = vectorizer.transform(df["clean_text"])
    y_true = df["Ticket Type"]
    y_pred = model.predict(X)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(df["Ticket Type"].unique())
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Support Ticket Classification')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
   
    plt.savefig(OUTPUT_PATH)
    print(f"✅ Confusion Matrix saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_visualization()
