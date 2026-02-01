import re
import string
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
custom_stop_words = {
    "hi", "hello", "regards", "kind", "best", "thank", "thanks", "sent", 
    "pm", "am", "subject", "from", "to", "cc", "date", "tuesday", "wednesday",
    "thursday", "friday", "monday", "dear"
}
stop_words.update(custom_stop_words)

def clean_text(text):
    """
    Cleans input text by removing special characters, numbers, and stopwords.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters and digits (keeping only a-z and spaces)
    text = re.sub(r"[^a-z\s]", "", text)
    
    # Remove stopwords
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    
    return " ".join(tokens)

def assign_priority(text, category):
    """
    Assigns a priority level (High, Medium, Low) based on text keywords and category.
    """
    text = text.lower()
    
    # Critical keywords -> HIGH
    # Added: 'leaver', 'starter', 'password' (blocking issues)
    high_keywords = [
        "urgent", "immediately", "critical", "down", "crash", "security", 
        "breach", "hack", "fire", "emergency", "fail", "broken",
        "leaver", "starter", "password", "locked"
    ]
    
    # Keywords -> MEDIUM
    medium_keywords = [
        "error", "issue", "bug", "slow", "access", "login", 
        "reset", "update", "performance", "wifi", "upgrade", "version"
    ]
    
    # Check for High Priority Keywords
    if any(keyword in text for keyword in high_keywords):
        return "High"
        
    # Check for High Priority Categories (if no keywords found)
    high_priority_categories = ["Access", "Security"] 
    if category in high_priority_categories:
        return "High"
    
    # Check for Medium Priority Keywords
    if any(keyword in text for keyword in medium_keywords):
        return "Medium"
        
    # Check for Medium Priority Categories (Hardware moved here)
    medium_priority_categories = ["Software", "System", "Hardware", "Administrative rights"]
    if category in medium_priority_categories:
        return "Medium"
        
    # Default to Low (HR Support, Purchase, etc unless 'leaver'/'starter' triggered High)
    return "Low"
