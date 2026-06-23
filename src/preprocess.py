import re
import string
import logging

logger = logging.getLogger("FakeShield")

def clean_text(text: str) -> str:
    """
    Complete text preprocessing pipeline for Fake News Detection.
    Includes lowercasing, special character removal, extra space removal.
    Note: Tokenization and lemmatization are implicitly handled 
    by the HuggingFace BERT tokenizer in the model pipeline, 
    but we do basic cleaning here.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 5. Remove numbers (optional, but often helpful in text classification)
    text = re.sub(r'\d+', '', text)
    
    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df, text_col='text'):
    """
    Apply cleaning to a dataframe column and handle missing values.
    """
    logger.info(f"Preprocessing dataframe of size {len(df)}")
    
    # Drop missing values
    df = df.dropna(subset=[text_col])
    
    # Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=[text_col])
    logger.info(f"Removed {initial_len - len(df)} duplicate records.")
    
    # Apply text cleaning
    df['cleaned_text'] = df[text_col].apply(clean_text)
    
    # Remove rows where cleaned text is empty
    df = df[df['cleaned_text'] != ""]
    
    logger.info(f"Final dataframe size: {len(df)}")
    return df
