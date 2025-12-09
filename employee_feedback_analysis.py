"""
AI-Driven Employee Feedback Analysis: Fostering Workplace Harmony
Solution following the exact problem statement requirements
Uses TextBlob for sentiment analysis and CountVectorizer for theme extraction
"""

import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import re
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

print("="*70)
print("AI-DRIVEN EMPLOYEE FEEDBACK ANALYSIS")
print("Fostering Workplace Harmony")
print("="*70)

# ============================================================================
# TASK 1: Data Preparation and Loading
# ============================================================================
print("\n" + "="*70)
print("TASK 1: Data Preparation and Loading")
print("="*70)

def load_and_prepare_data(csv_path='employee_reviews.csv'):
    """
    Load employee feedback data from CSV file and prepare it for analysis.
    Cleans data by removing unnecessary characters, converting to lowercase,
    and tokenizing the feedback.
    """
    print("\n1. Loading employee feedback data from CSV file...")
    
    # Try different encodings to handle special characters
    encodings = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
            print(f"   ✓ Successfully loaded with {encoding} encoding")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"   Error with {encoding} encoding: {e}")
            continue
    
    if df is None:
        # Last resort: read with error handling
        print("   Trying with UTF-8 and error replacement...")
        import io
        try:
            with open(csv_path, 'rb') as f:
                content = f.read()
            content_decoded = content.decode('utf-8', errors='replace')
            df = pd.read_csv(io.StringIO(content_decoded), low_memory=False)
            print("   ✓ Successfully loaded with UTF-8 error replacement")
        except Exception as e:
            raise Exception(f"Could not load CSV file. Error: {e}")
    
    print(f"\n2. Dataset shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Combine summary and pros&cons into a single feedback column
    print("\n3. Combining feedback columns...")
    if 'summary' in df.columns and 'pros&cons' in df.columns:
        df['feedback'] = df['summary'].fillna('') + ' ' + df['pros&cons'].fillna('')
        df['feedback'] = df['feedback'].str.strip()
    elif 'feedback' not in df.columns:
        # If no feedback column exists, use the first text column
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            df['feedback'] = df[text_columns[0]].fillna('')
        else:
            raise ValueError("No feedback column found in the dataset")
    
    # Remove rows with empty feedback
    df = df[df['feedback'].str.len() > 0]
    print(f"   ✓ After removing empty feedback: {df.shape[0]} rows")
    
    # Clean the feedback text
    print("\n4. Cleaning text data...")
    print("   - Converting to lowercase")
    print("   - Removing unnecessary characters")
    print("   - Tokenizing feedback")
    
    def clean_text(text):
        """Clean text by removing unnecessary characters and converting to lowercase"""
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and punctuation (keep spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    df['cleaned_feedback'] = df['feedback'].apply(clean_text)
    df = df[df['cleaned_feedback'].str.len() > 0]
    
    print(f"   ✓ After cleaning: {df.shape[0]} rows")
    
    # Display first few rows
    print("\n5. First few rows of the data:")
    print("-" * 70)
    display_columns = ['feedback', 'cleaned_feedback']
    if 'overall-ratings' in df.columns:
        display_columns.insert(0, 'overall-ratings')
    
    print(df[display_columns].head(10).to_string())
    print("-" * 70)
    
    return df

# Load the data
df = load_and_prepare_data()

# ============================================================================
# TASK 2: Sentiment Analysis using TextBlob
# ============================================================================
print("\n" + "="*70)
print("TASK 2: Sentiment Analysis using TextBlob")
print("="*70)

print("\n1. Performing sentiment analysis on 'feedback' column using TextBlob...")

def get_sentiment_label(polarity):
    """
    Classify sentiment as positive, negative, or neutral based on polarity.
    - polarity > 0: positive
    - polarity < 0: negative
    - polarity == 0: neutral
    """
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Calculate sentiment polarity using TextBlob
print("2. Calculating sentiment polarity for each feedback...")
df['sentiment_polarity'] = df['cleaned_feedback'].apply(
    lambda text: TextBlob(text).sentiment.polarity if text else 0.0
)

# Classify sentiment based on polarity
print("3. Classifying sentiment as positive, negative, or neutral...")
df['sentiment'] = df['sentiment_polarity'].apply(get_sentiment_label)

# Display sentiment distribution
print("\n4. Sentiment distribution:")
print("-" * 70)
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)
print("\nSentiment percentages:")
print(df['sentiment'].value_counts(normalize=True) * 100)
print("-" * 70)

# Display first few rows with sentiment labels
print("\n5. First few rows of data with sentiment labels:")
print("-" * 70)
display_cols = ['feedback', 'sentiment_polarity', 'sentiment']
if 'overall-ratings' in df.columns:
    display_cols.insert(0, 'overall-ratings')

print(df[display_cols].head(10).to_string())
print("-" * 70)

# ============================================================================
# TASK 3: Key Themes Extraction using CountVectorizer
# ============================================================================
print("\n" + "="*70)
print("TASK 3: Key Themes Extraction using CountVectorizer")
print("="*70)

print("\n1. Extracting key themes from 'feedback' column using CountVectorizer...")

# Get stop words
stop_words = set(stopwords.words('english'))

# Use CountVectorizer to convert feedback to matrix of token counts
print("2. Converting feedback data into matrix of token counts...")
print("   - Excluding common stop words")
print("   - Using unigrams (single words)")

# Initialize CountVectorizer
# max_features: limit to top N most frequent words
# stop_words: exclude common stop words
# min_df: ignore terms that appear in less than 2 documents
vectorizer = CountVectorizer(
    max_features=100,  # Top 100 most common words
    stop_words='english',  # Exclude stop words
    min_df=2,  # Minimum document frequency
    ngram_range=(1, 1)  # Unigrams only
)

# Fit and transform the cleaned feedback
print("3. Fitting CountVectorizer on cleaned feedback...")
try:
    # Use cleaned_feedback for better results
    count_matrix = vectorizer.fit_transform(df['cleaned_feedback'])
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate word frequencies (sum across all documents)
    word_counts = count_matrix.sum(axis=0).A1
    
    # Create a dictionary of word: count
    word_freq_dict = dict(zip(feature_names, word_counts))
    
    # Sort by frequency (descending)
    sorted_words = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("4. Most common words in the feedback data:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Word':<25} {'Frequency':<15}")
    print("-" * 70)
    
    # Display top 20 most common words
    for i, (word, count) in enumerate(sorted_words[:20], 1):
        print(f"{i:<6} {word:<25} {count:<15}")
    
    print("-" * 70)
    
    # Extract top 10 HR-related keywords
    print("\n5. Top 10 HR-related keywords extracted:")
    print("-" * 70)
    
    # HR-related keywords to look for
    hr_keywords = [
        'salary', 'workload', 'culture', 'management', 'stress', 'growth',
        'team', 'support', 'bonus', 'environment', 'work', 'balance', 'benefit',
        'compensation', 'career', 'opportunity', 'manager', 'leadership',
        'colleague', 'workplace', 'employee', 'company', 'time', 'hour',
        'pressure', 'promotion', 'development', 'training', 'feedback',
        'review', 'performance', 'office', 'flexible', 'schedule'
    ]
    
    # Filter and get top HR-related keywords
    hr_found = [(word, count) for word, count in sorted_words if any(hr_term in word.lower() for hr_term in hr_keywords)]
    
    if hr_found:
        print(f"{'Rank':<6} {'HR Keyword':<25} {'Frequency':<15}")
        print("-" * 70)
        for i, (word, count) in enumerate(hr_found[:10], 1):
            print(f"{i:<6} {word:<25} {count:<15}")
    else:
        # If no HR keywords found, show top 10 overall
        print("No specific HR keywords found. Showing top 10 overall keywords:")
        print(f"{'Rank':<6} {'Word':<25} {'Frequency':<15}")
        print("-" * 70)
        for i, (word, count) in enumerate(sorted_words[:10], 1):
            print(f"{i:<6} {word:<25} {count:<15}")
    
    print("-" * 70)
    
except Exception as e:
    print(f"Error in theme extraction: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary and Conclusion
# ============================================================================
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
By leveraging sentiment analysis and NLP techniques, the project efficiently:
✓ Classifies employee feedback as positive, negative, or neutral
✓ Extracts key themes from the data

This automated analysis provides HR teams with valuable insights into employee 
experiences and concerns, enabling them to take targeted actions to improve 
workplace satisfaction and address potential issues. The streamlined approach 
saves time and resources while delivering actionable information for better 
decision-making and employee engagement.
""")

print("="*70)
print("Analysis Complete!")
print("="*70)

# Save results to CSV
output_file = 'feedback_analysis_results.csv'
df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")


