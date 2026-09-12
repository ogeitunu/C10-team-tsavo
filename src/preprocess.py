import re
import pandas as pd

def clean_text(text: str) -> str:
    """Cleans whitespace and normalizes text while preserving chemical/NPK codes."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def enrich_documents(docs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuses Title, Crop Category, Section/Source, and Body Text into a unified high-signal string schema:
    Schema = Title | Crop | Source | Text
    """
    df = docs_df.copy()
    df['title'] = df['title'].fillna('').apply(clean_text)
    df['crop'] = df['crop'].fillna('general').apply(clean_text)
    df['source'] = df['source'].fillna('').apply(clean_text)
    df['text'] = df['text'].fillna('').apply(clean_text)
    
    # Unified document schema string
    df['enriched_text'] = (
        "Title: " + df['title'] + 
        " | Crop: " + df['crop'] + 
        " | Source: " + df['source'] + 
        " | Text: " + df['text']
    )
    return df
