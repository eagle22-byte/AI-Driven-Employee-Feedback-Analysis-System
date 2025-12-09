"""
Data Preprocessing Script for Employee Feedback Analysis
Handles text cleaning, normalization, and preparation for model training
"""

import pandas as pd
import re
import string
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def remove_emojis(text):
    """Remove emojis from text"""
    if pd.isna(text):
        return ""
    return emoji.replace_emoji(str(text), replace='')

def clean_text(text):
    """
    Comprehensive text cleaning function
    - Convert to lowercase
    - Remove emojis
    - Remove URLs
    - Remove special characters and punctuation
    - Remove numbers
    - Remove stopwords
    - Tokenize
    - Lemmatize
    """
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove emojis
    text = remove_emojis(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters and punctuation (keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords and lemmatize
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:  # Remove very short words
            lemmatized = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemmatized)
    
    # Join tokens back
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

def preprocess_dataset(csv_path='employee_reviews.csv', output_path='processed_data.csv'):
    """
    Load and preprocess the employee feedback dataset
    """
    print("Loading dataset...")
    # Try different encodings to handle special characters
    encodings = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
            print(f"Successfully loaded with {encoding} encoding")
            break
        except (UnicodeDecodeError, UnicodeError) as e:
            print(f"Failed to load with {encoding} encoding, trying next...")
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")
            continue
    
    if df is None:
        # Last resort: read with error handling (replace invalid characters)
        print("Trying with UTF-8 and error replacement...")
        try:
            # Read file as binary and decode with error handling
            with open(csv_path, 'rb') as f:
                content = f.read()
            # Decode with error replacement
            content_decoded = content.decode('utf-8', errors='replace')
            # Write to temporary string and read with pandas
            df = pd.read_csv(io.StringIO(content_decoded), low_memory=False)
            print("Successfully loaded with UTF-8 error replacement")
        except Exception as e:
            raise Exception(f"Could not load CSV file with any encoding. Last error: {e}")
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Combine summary and pros&cons into a single feedback column
    print("\nCombining feedback columns...")
    df['feedback'] = df['summary'].fillna('') + ' ' + df['pros&cons'].fillna('')
    df['feedback'] = df['feedback'].str.strip()
    
    # Remove rows with empty feedback
    df = df[df['feedback'].str.len() > 0]
    print(f"After removing empty feedback: {df.shape}")
    
    # Clean the feedback text
    print("\nCleaning text data (this may take a while)...")
    df['clean_feedback'] = df['feedback'].apply(clean_text)
    
    # Remove rows where cleaned feedback is empty
    df = df[df['clean_feedback'].str.len() > 0]
    print(f"After cleaning: {df.shape}")
    
    # Display sample
    print("\nSample of cleaned data:")
    print(df[['feedback', 'clean_feedback']].head(10))
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    df = preprocess_dataset()
    print("\nPreprocessing completed!")

