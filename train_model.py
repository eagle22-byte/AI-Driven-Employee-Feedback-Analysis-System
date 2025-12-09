"""
Model Training Script
- TF-IDF Vectorization
- Train Naive Bayes or Logistic Regression
- Save model and vectorizer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import os

def train_model(csv_path='final_dataset.csv', model_type='naive_bayes', use_smote=False):
    """
    Train sentiment classification model
    
    Parameters:
    - csv_path: Path to the final dataset
    - model_type: 'naive_bayes' or 'logistic_regression'
    - use_smote: Whether to use SMOTE for oversampling
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Remove rows with missing clean_feedback
    df = df[df['clean_feedback'].notna() & (df['clean_feedback'].str.len() > 0)]
    
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features and labels
    X = df['clean_feedback'].values
    y = df['sentiment'].values
    
    print(f"\nClass distribution:")
    print(pd.Series(y).value_counts())
    
    # Split dataset: 80% training, 20% testing
    print("\nSplitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # TF-IDF Vectorization
    print("\nApplying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,  # Minimum document frequency
        max_df=0.95  # Maximum document frequency
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    # Apply SMOTE if requested
    if use_smote:
        print("\nApplying SMOTE for oversampling...")
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)
        print(f"After SMOTE - Training set size: {len(X_train)}")
        print(f"Class distribution after SMOTE:")
        print(pd.Series(y_train).value_counts())
    
    # Train model
    print(f"\nTraining {model_type} model...")
    if model_type == 'naive_bayes':
        model = MultinomialNB(alpha=1.0)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    else:
        raise ValueError("model_type must be 'naive_bayes' or 'logistic_regression'")
    
    model.fit(X_train_tfidf, y_train)
    print("Model training completed!")
    
    # Predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluation
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save model and vectorizer
    print("\nSaving model and vectorizer...")
    model_filename = 'sentiment_model.pkl'
    vectorizer_filename = 'vectorizer.pkl'
    
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    
    print(f"Model saved to: {model_filename}")
    print(f"Vectorizer saved to: {vectorizer_filename}")
    
    # Manual test with example sentences
    print("\n" + "="*50)
    print("MANUAL TESTING")
    print("="*50)
    
    test_sentences = [
        "I feel management is supportive but workload is stressful!",
        "Salary is good but workload pressure is too high",
        "Great company culture and amazing benefits",
        "The work environment is okay, nothing special",
        "Terrible management and no work life balance"
    ]
    
    print("\nTesting with sample sentences:")
    for sentence in test_sentences:
        # Clean the sentence
        from preprocess_data import clean_text
        cleaned = clean_text(sentence)
        if cleaned:
            # Vectorize
            sentence_tfidf = vectorizer.transform([cleaned])
            # Predict
            prediction = model.predict(sentence_tfidf)[0]
            probability = model.predict_proba(sentence_tfidf)[0]
            
            print(f"\nOriginal: {sentence}")
            print(f"Cleaned: {cleaned}")
            print(f"Prediction: {prediction}")
            print(f"Probabilities: {dict(zip(model.classes_, probability))}")
    
    return model, vectorizer

if __name__ == "__main__":
    # Train with Naive Bayes (recommended for first time)
    print("Training Naive Bayes model...")
    model, vectorizer = train_model(model_type='naive_bayes', use_smote=False)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)

