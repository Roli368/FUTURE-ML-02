import pandas as pd
import os
from utils import clean_text

RAW_PATH = "data/raw/all_tickets.csv"
PROCESSED_DATA_DIR = "data/processed"
PROCESSED_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, "clean_tickets.csv")

def preprocess():
   
    print(" Starting preprocessing for IT Service Tickets...")
    
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f" Raw data not found at {RAW_PATH}")

   
    df = pd.read_csv(RAW_PATH)
    print(f" Loaded {len(df)} rows from {RAW_PATH}")
    print(f"Columns: {list(df.columns)}")

   
    if "Document" not in df.columns or "Topic_group" not in df.columns:
        raise ValueError(f" Unexpected columns. Expected 'Document' and 'Topic_group'. Found: {list(df.columns)}")

    # Rename for consistency
    df.rename(columns={"Topic_group": "Ticket Type"}, inplace=True)

    # Clean text
    print(" Cleaning text...")
    df["clean_text"] = df["Document"].apply(clean_text)
    
    # Ensure processed directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save processed data (Only needed columns)
    output_df = df[["clean_text", "Ticket Type"]]
    output_df.to_csv(PROCESSED_FILE_PATH, index=False)
    
    print(f" Preprocessing complete! Saved to {PROCESSED_FILE_PATH}")

if __name__ == "__main__":
    preprocess()
