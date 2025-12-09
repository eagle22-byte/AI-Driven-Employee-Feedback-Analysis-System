"""
Dataset Preparation Script
- Converts ratings to sentiment labels
- Checks class balance
- Applies SMOTE if needed
"""

import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def convert_rating_to_sentiment(rating):
    """
    Convert rating to sentiment label:
    1-2 stars → negative
    3 stars → neutral
    4-5 stars → positive
    """
    if pd.isna(rating) or rating == 'none' or rating == '':
        return None
    
    try:
        rating = float(rating)
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:  # 4 or 5
            return 'positive'
    except (ValueError, TypeError):
        return None

def prepare_dataset(csv_path='processed_data.csv', output_path='final_dataset.csv'):
    """
    Prepare dataset with sentiment labels
    """
    print("Loading processed dataset...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    
    # Convert overall-ratings to sentiment
    print("\nConverting ratings to sentiment labels...")
    df['sentiment'] = df['overall-ratings'].apply(convert_rating_to_sentiment)
    
    # Remove rows with no sentiment
    df = df[df['sentiment'].notna()]
    print(f"After removing rows with no sentiment: {df.shape}")
    
    # Check class distribution
    print("\nClass distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    print(sentiment_counts)
    print(f"\nClass percentages:")
    print(df['sentiment'].value_counts(normalize=True) * 100)
    
    # Check if dataset is imbalanced
    total = len(df)
    min_class_count = sentiment_counts.min()
    max_class_count = sentiment_counts.max()
    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
    
    print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 2.0:
        print("WARNING: Dataset is imbalanced. Consider applying SMOTE during training.")
    else:
        print("OK: Dataset is relatively balanced.")
    
    # Save final dataset
    df.to_csv(output_path, index=False)
    print(f"\nFinal dataset saved to {output_path}")
    
    # Display sample
    print("\nSample of final dataset:")
    print(df[['feedback', 'clean_feedback', 'sentiment']].head(10))
    
    return df

if __name__ == "__main__":
    df = prepare_dataset()
    print("\nDataset preparation completed!")

