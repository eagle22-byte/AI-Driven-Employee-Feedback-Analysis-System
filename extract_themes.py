"""
Theme Extraction Script
Extracts top HR-related keywords from employee feedback
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def extract_themes(csv_path='final_dataset.csv', top_n=10, create_wordcloud=True):
    """
    Extract top HR-related keywords from feedback
    
    Parameters:
    - csv_path: Path to the dataset
    - top_n: Number of top keywords to extract
    - create_wordcloud: Whether to generate a word cloud visualization
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Remove rows with missing clean_feedback
    df = df[df['clean_feedback'].notna() & (df['clean_feedback'].str.len() > 0)]
    
    print(f"Dataset shape: {df.shape}")
    
    # Combine all clean feedback
    all_feedback = ' '.join(df['clean_feedback'].astype(str))
    
    # Use TF-IDF to extract important keywords
    print("\nExtracting keywords using TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=5,  # Minimum document frequency
        max_df=0.95
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['clean_feedback'])
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate mean TF-IDF scores across all documents
    mean_scores = tfidf_matrix.mean(axis=0).A1
    
    # Get top keywords
    top_indices = mean_scores.argsort()[-top_n*2:][::-1]  # Get more to filter HR-related
    top_keywords = [(feature_names[i], mean_scores[i]) for i in top_indices]
    
    # Filter for HR-related keywords
    hr_keywords = [
        'salary', 'workload', 'culture', 'management', 'stress', 'growth',
        'team', 'support', 'bonus', 'environment', 'work', 'balance', 'benefit',
        'compensation', 'career', 'opportunity', 'manager', 'leadership',
        'colleague', 'workplace', 'employee', 'company', 'time', 'hour',
        'pressure', 'promotion', 'development', 'training', 'feedback',
        'review', 'performance', 'office', 'flexible', 'schedule'
    ]
    
    # Extract top HR-related keywords
    extracted_keywords = []
    for keyword, score in top_keywords:
        # Check if keyword contains any HR-related term
        if any(hr_term in keyword.lower() for hr_term in hr_keywords):
            extracted_keywords.append((keyword, score))
        if len(extracted_keywords) >= top_n:
            break
    
    # If we don't have enough, add top keywords by frequency
    if len(extracted_keywords) < top_n:
        print("\nUsing CountVectorizer for additional keywords...")
        count_vectorizer = CountVectorizer(
            max_features=100,
            ngram_range=(1, 1),
            min_df=10
        )
        count_matrix = count_vectorizer.fit_transform(df['clean_feedback'])
        word_counts = count_matrix.sum(axis=0).A1
        feature_names_count = count_vectorizer.get_feature_names_out()
        
        top_count_indices = word_counts.argsort()[-50:][::-1]
        for idx in top_count_indices:
            keyword = feature_names_count[idx]
            if keyword not in [k[0] for k in extracted_keywords]:
                extracted_keywords.append((keyword, word_counts[idx]))
            if len(extracted_keywords) >= top_n:
                break
    
    # Display results
    print("\n" + "="*50)
    print(f"TOP {top_n} HR-RELATED KEYWORDS")
    print("="*50)
    
    for i, (keyword, score) in enumerate(extracted_keywords[:top_n], 1):
        print(f"{i}. {keyword} (score: {score:.4f})")
    
    # Create word cloud if requested
    if create_wordcloud:
        print("\nGenerating word cloud...")
        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(all_feedback)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Employee Feedback Word Cloud', fontsize=16, pad=20)
            plt.tight_layout()
            
            wordcloud_path = 'wordcloud.png'
            plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
            print(f"Word cloud saved to: {wordcloud_path}")
            plt.close()
        except Exception as e:
            print(f"Could not generate word cloud: {e}")
            print("(This is optional and doesn't affect the main functionality)")
    
    # Save keywords to file
    keywords_file = 'extracted_keywords.txt'
    with open(keywords_file, 'w') as f:
        f.write("Top HR-Related Keywords Extracted from Employee Feedback\n")
        f.write("="*50 + "\n\n")
        for i, (keyword, score) in enumerate(extracted_keywords[:top_n], 1):
            f.write(f"{i}. {keyword} (score: {score:.4f})\n")
    
    print(f"\nKeywords saved to: {keywords_file}")
    
    return [keyword for keyword, _ in extracted_keywords[:top_n]]

if __name__ == "__main__":
    keywords = extract_themes(top_n=10, create_wordcloud=True)
    print("\nTheme extraction completed!")
    print(f"\nExtracted keywords: {keywords}")

