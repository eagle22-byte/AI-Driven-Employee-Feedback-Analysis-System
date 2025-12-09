"""
Feature-Rich Flask Backend API for Employee Feedback Analysis
Includes: Sentiment Analysis, Emotion Detection, AI Summaries, Department Analysis
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import json
import io
import csv
from preprocess_data import clean_text
import re
from collections import Counter
from textblob import TextBlob
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
print("Loading model and vectorizer...")
try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("Model and vectorizer loaded successfully!")
    model_loaded = True
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run train_model.py first to generate the model files.")
    model = None
    vectorizer = None
    model_loaded = False

# Load dataset for random sampling and department analysis
dataset_df = None
try:
    if os.path.exists('final_dataset.csv'):
        dataset_df = pd.read_csv('final_dataset.csv', encoding='latin-1', low_memory=False)
        print(f"Dataset loaded: {len(dataset_df)} rows")
        # Ensure feedback column exists
        if 'feedback' not in dataset_df.columns:
            if 'summary' in dataset_df.columns and 'pros&cons' in dataset_df.columns:
                dataset_df['feedback'] = dataset_df['summary'].fillna('') + ' ' + dataset_df['pros&cons'].fillna('')
    else:
        print("Warning: final_dataset.csv not found. Random sample and department analysis will not work.")
except Exception as e:
    print(f"Could not load dataset: {e}")

# Model evaluation metrics (from training)
MODEL_METRICS = {
    'accuracy': 0.7210,
    'precision': 0.6604,
    'recall': 0.7210,
    'f1_score': 0.6485,
    'confusion_matrix': {
        'negative': {'negative': 714, 'neutral': 90, 'positive': 987},
        'neutral': {'negative': 226, 'neutral': 108, 'positive': 2131},
        'positive': {'negative': 155, 'neutral': 73, 'positive': 8643}
    }
}

# Load extracted keywords
def load_keywords():
    """Load or extract keywords for theme detection"""
    keywords_file = 'extracted_keywords.txt'
    if os.path.exists(keywords_file):
        keywords = []
        with open(keywords_file, 'r') as f:
            for line in f:
                match = re.search(r'\d+\.\s+(\w+)', line)
                if match:
                    keywords.append(match.group(1).lower())
        return keywords[:10]
    else:
        return ['salary', 'workload', 'culture', 'management', 'stress', 
                'growth', 'team', 'support', 'bonus', 'environment']

hr_keywords = load_keywords()

def detect_emotions(text):
    """
    Detect emotions in text: joy, stress, anger, frustration, motivation, neutral
    Returns dictionary with emotion percentages
    """
    if not text:
        return {}
    
    text_lower = text.lower()
    
    # Emotion keywords
    emotion_keywords = {
        'joy': ['happy', 'great', 'amazing', 'wonderful', 'excellent', 'fantastic', 'love', 'enjoy', 'pleased', 'satisfied', 'delighted', 'excited', 'thrilled'],
        'stress': ['stress', 'stressful', 'pressure', 'overwhelming', 'anxious', 'worried', 'tension', 'burnout', 'exhausted', 'tired', 'drained'],
        'anger': ['angry', 'furious', 'mad', 'rage', 'hate', 'disgusted', 'annoyed', 'irritated', 'frustrated'],
        'frustration': ['frustrated', 'frustrating', 'disappointed', 'disappointing', 'upset', 'unhappy', 'dissatisfied', 'let down'],
        'motivation': ['motivated', 'inspired', 'encouraged', 'energized', 'enthusiastic', 'passionate', 'driven', 'ambitious', 'determined'],
        'neutral': ['okay', 'fine', 'average', 'normal', 'regular', 'standard', 'typical']
    }
    
    emotion_scores = {}
    total_matches = 0
    
    for emotion, keywords in emotion_keywords.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        emotion_scores[emotion] = matches
        total_matches += matches
    
    # Calculate percentages
    if total_matches > 0:
        emotion_percentages = {k: round((v / total_matches) * 100, 1) for k, v in emotion_scores.items() if v > 0}
    else:
        # Default to neutral if no emotions detected
        emotion_percentages = {'neutral': 100.0}
    
    # Normalize to 100%
    total = sum(emotion_percentages.values())
    if total > 0:
        emotion_percentages = {k: round((v / total) * 100, 1) for k, v in emotion_percentages.items()}
    
    return emotion_percentages

def generate_ai_summary(feedback, sentiment, themes, emotions):
    """
    Generate AI-based improvement summary for HR
    """
    sentiment_lower = sentiment.lower()
    
    # Build summary based on analysis
    summary_parts = []
    
    # Sentiment-based insights
    if sentiment_lower == 'positive':
        summary_parts.append("Employees express positive sentiment overall.")
    elif sentiment_lower == 'negative':
        summary_parts.append("Employees express concerns that require attention.")
    else:
        summary_parts.append("Mixed feedback indicates areas for improvement.")
    
    # Theme-based insights
    if themes:
        theme_str = ', '.join(themes[:3])
        if 'salary' in themes or 'compensation' in themes:
            summary_parts.append(f"Compensation-related themes ({theme_str}) are prominent.")
        if 'workload' in themes or 'stress' in themes:
            summary_parts.append(f"Workload and stress concerns ({theme_str}) suggest potential burnout risks.")
        if 'management' in themes:
            summary_parts.append(f"Management-related feedback ({theme_str}) highlights leadership areas.")
        if 'culture' in themes or 'environment' in themes:
            summary_parts.append(f"Workplace culture themes ({theme_str}) indicate organizational climate focus.")
    
    # Emotion-based insights
    if emotions:
        top_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else None
        if top_emotion == 'stress' and emotions.get('stress', 0) > 30:
            summary_parts.append("High stress levels detected - consider wellness programs and workload redistribution.")
        if top_emotion == 'joy' and emotions.get('joy', 0) > 40:
            summary_parts.append("Positive emotions indicate good workplace satisfaction.")
        if 'frustration' in emotions and emotions.get('frustration', 0) > 25:
            summary_parts.append("Frustration patterns suggest need for better communication and support systems.")
    
    # Recommendations
    recommendations = []
    if sentiment_lower == 'negative' or (emotions and emotions.get('stress', 0) > 30):
        recommendations.append("HR should explore workload redistribution and mental-health support policies.")
    if 'management' in themes:
        recommendations.append("Consider leadership training and management development programs.")
    if 'salary' in themes or 'compensation' in themes:
        recommendations.append("Review compensation structures and benefits packages.")
    if 'culture' in themes:
        recommendations.append("Focus on building positive workplace culture and team engagement.")
    
    # Combine summary
    summary = " ".join(summary_parts)
    if recommendations:
        summary += " " + " ".join(recommendations)
    
    if not summary:
        summary = "Analysis indicates standard workplace feedback patterns. Continue monitoring employee satisfaction."
    
    return summary

def extract_themes_from_text(text, keywords_list):
    """Extract themes from text based on keyword matching"""
    if not text:
        return []
    
    text_lower = text.lower()
    found_themes = []
    
    for keyword in keywords_list:
        if keyword in text_lower:
            found_themes.append(keyword)
    
    # Common HR-related terms
    common_hr_terms = {
        'salary': ['salary', 'pay', 'compensation', 'wage', 'income'],
        'workload': ['workload', 'work load', 'pressure', 'demand'],
        'culture': ['culture', 'environment', 'atmosphere'],
        'management': ['management', 'manager', 'leadership', 'boss'],
        'stress': ['stress', 'stressful', 'pressure', 'overwhelming'],
        'growth': ['growth', 'development', 'career', 'advancement'],
        'team': ['team', 'colleague', 'coworker', 'peer'],
        'support': ['support', 'supportive', 'help'],
        'benefit': ['benefit', 'perk', 'bonus', 'reward'],
        'balance': ['balance', 'work life', 'work-life']
    }
    
    for theme, terms in common_hr_terms.items():
        if any(term in text_lower for term in terms):
            if theme not in found_themes:
                found_themes.append(theme)
    
    return found_themes[:5]

def get_department_sentiment():
    """Calculate department-wise sentiment if job-title column exists"""
    if dataset_df is None or 'job-title' not in dataset_df.columns:
        return None
    
    try:
        # Extract department from job title (simple extraction)
        def extract_dept(title):
            if pd.isna(title):
                return 'Other'
            title_lower = str(title).lower()
            if any(x in title_lower for x in ['engineer', 'developer', 'programmer', 'software', 'technical']):
                return 'IT'
            elif any(x in title_lower for x in ['sales', 'account', 'business']):
                return 'Sales'
            elif any(x in title_lower for x in ['support', 'customer', 'service']):
                return 'Support'
            elif any(x in title_lower for x in ['manager', 'director', 'lead', 'head']):
                return 'Management'
            elif any(x in title_lower for x in ['hr', 'human resource', 'recruiter']):
                return 'HR'
            else:
                return 'Other'
        
        df_copy = dataset_df.copy()
        df_copy['department'] = df_copy['job-title'].apply(extract_dept)
        
        if 'sentiment' not in df_copy.columns:
            return None
        
        dept_sentiment = df_copy.groupby('department')['sentiment'].apply(
            lambda x: pd.Series({
                'positive': (x == 'positive').sum() / len(x) * 100,
                'neutral': (x == 'neutral').sum() / len(x) * 100,
                'negative': (x == 'negative').sum() / len(x) * 100,
                'total': len(x)
            })
        ).to_dict()
        
        # Format for frontend
        result = []
        for dept, stats in dept_sentiment.items():
            if isinstance(stats, dict):
                result.append({
                    'department': dept,
                    'positive_pct': round(stats.get('positive', 0), 1),
                    'neutral_pct': round(stats.get('neutral', 0), 1),
                    'negative_pct': round(stats.get('negative', 0), 1),
                    'total': stats.get('total', 0)
                })
        
        return result
    except Exception as e:
        print(f"Error calculating department sentiment: {e}")
        return None

@app.route('/')
def index():
    """Render the main UI page"""
    return render_template('index.html')

@app.route('/analyze_feedback', methods=['POST'])
def analyze_feedback():
    """Enhanced analyze endpoint with emotions and AI summary"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'feedback' not in data:
            return jsonify({'error': 'Missing "feedback" field in request body'}), 400
        
        feedback_text = data['feedback']
        
        if not feedback_text or len(feedback_text.strip()) == 0:
            return jsonify({'error': 'Feedback text cannot be empty'}), 400
        
        # Clean the feedback text
        cleaned_text = clean_text(feedback_text)
        
        if not cleaned_text or len(cleaned_text.strip()) == 0:
            return jsonify({
                'error': 'After cleaning, feedback text is empty. Please provide valid text.'
            }), 400
        
        # Vectorize
        text_tfidf = vectorizer.transform([cleaned_text])
        
        # Predict sentiment
        sentiment = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        prob_dict = dict(zip(model.classes_, probabilities))
        
        # Calculate polarity score using TextBlob
        polarity = TextBlob(cleaned_text).sentiment.polarity
        
        # Improve sentiment classification for mixed/negative feedback
        # Check for negative indicators that should override positive prediction
        text_lower = cleaned_text.lower()
        original_text_lower = feedback_text.lower()
        
        # Comprehensive negative indicators
        negative_indicators = [
            'too high', 'too much', 'too many', 'pressure', 'stressful', 'stress',
            'overwhelming', 'difficult', 'hard', 'bad', 'very bad', 'terrible', 'awful',
            'not good', 'not great', 'disappointed', 'frustrated', 'worst',
            'no work life balance', 'work life balance bad', 'burnout',
            'some negatives', 'negative', 'aggressive', 'caustic', 'toxic',
            'problem', 'issue', 'concern', 'worry', 'complaint', 'dissatisfied',
            'unhappy', 'poor', 'weak', 'lack', 'missing', 'fail', 'failure',
            'stress is high', 'high stress'
        ]
        
        # Strong negative phrases (check in original text to preserve context)
        strong_negative_phrases = [
            'caustic work', 'caustic work environment', 'caustic work environments',
            'toxic environment', 'toxic environments',
            'aggressive personality', 'aggressive personalities',
            'difficult to have balance', 'difficult to have a balance', 'very difficult', 'some negatives',
            'work life balance', 'no balance', 'poor management', 'bad culture',
            'caustic', 'aggressive', 'toxic', 'make for caustic'
        ]
        
        # Count negative indicators in both cleaned and original text
        negative_count = sum(1 for indicator in negative_indicators 
                           if indicator in text_lower or indicator in original_text_lower)
        
        # Check for strong negative phrases in original text (preserves context)
        has_strong_negative = any(phrase in original_text_lower for phrase in strong_negative_phrases)
        
        # Also check for partial matches (e.g., "very difficult to have balance" matches "difficult to have balance")
        if not has_strong_negative:
            for phrase in strong_negative_phrases:
                if len(phrase) > 10:  # Only for longer phrases
                    words = phrase.split()
                    if len(words) >= 2:
                        # Check if key words appear together (last 2-3 words)
                        key_words = words[-2:] if len(words) >= 2 else words
                        if all(word in original_text_lower for word in key_words):
                            has_strong_negative = True
                            break
        
        # If model predicted positive/neutral but we have negative indicators, adjust
        if sentiment in ['positive', 'neutral']:
            # STRONG OVERRIDE: If we have strong negative phrases, ALWAYS override to neutral/negative
            if has_strong_negative:
                # Very aggressive: strong negative phrases mean it's definitely not purely positive
                # Always override if we detect strong negative phrases
                if prob_dict.get('negative', 0) > prob_dict.get('neutral', 0):
                    sentiment = 'negative'
                else:
                    # Even if neutral prob is lower, if we have strong negatives, go neutral
                    sentiment = 'neutral'
            # If polarity is negative (< 0) and we have negative indicators, override
            elif polarity < 0 and negative_count >= 1:
                # Negative polarity + negative words = definitely negative
                sentiment = 'negative'
            # Multiple negative indicators
            elif negative_count >= 2:
                if prob_dict.get('negative', 0) > 0.15 or prob_dict.get('neutral', 0) > 0.3:
                    if prob_dict.get('negative', 0) > prob_dict.get('neutral', 0):
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
            # Single negative indicator with low polarity
            elif negative_count >= 1 and polarity < 0.15:
                if prob_dict.get('neutral', 0) > 0.25:
                    sentiment = 'neutral'
                elif prob_dict.get('negative', 0) > 0.2:
                    sentiment = 'negative'
            
            # FINAL OVERRIDE: If polarity is negative (< 0) and we have ANY negative indicators
            if polarity < 0 and (negative_count >= 1 or has_strong_negative):
                # Negative polarity + negative indicators = definitely negative
                if polarity < -0.15:
                    sentiment = 'negative'
                elif prob_dict.get('negative', 0) > prob_dict.get('neutral', 0):
                    sentiment = 'negative'
                elif prob_dict.get('neutral', 0) > 0.15:
                    sentiment = 'neutral'
        
        # ABSOLUTE FINAL OVERRIDE: If polarity is strongly negative, always negative (regardless of everything else)
        if polarity < -0.2:
            sentiment = 'negative'
        elif polarity < -0.1 and (negative_count >= 1 or has_strong_negative):
            sentiment = 'negative'
        
        # Extract themes
        themes = extract_themes_from_text(cleaned_text, hr_keywords)
        if not themes:
            themes = extract_themes_from_text(feedback_text.lower(), hr_keywords)
        
        # Detect emotions
        emotions = detect_emotions(cleaned_text)
        
        # Generate AI summary
        ai_summary = generate_ai_summary(feedback_text, sentiment, themes, emotions)
        
        response = {
            'sentiment': sentiment,
            'themes': themes,
            'confidence': {
                'positive': round(prob_dict.get('positive', 0), 3),
                'neutral': round(prob_dict.get('neutral', 0), 3),
                'negative': round(prob_dict.get('negative', 0), 3)
            },
            'polarity_score': round(polarity, 3),
            'emotions': emotions,
            'ai_summary': ai_summary,
            'cleaned_text': cleaned_text
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/random_sample', methods=['GET'])
def random_sample():
    """Get a random sample from the dataset"""
    if dataset_df is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    try:
        # Get random row
        random_row = dataset_df.sample(n=1).iloc[0]
        
        feedback = str(random_row.get('feedback', ''))
        if pd.isna(feedback) or feedback == '':
            feedback = str(random_row.get('summary', '')) + ' ' + str(random_row.get('pros&cons', ''))
        
        return jsonify({
            'feedback': feedback.strip(),
            'job_title': str(random_row.get('job-title', 'N/A')),
            'rating': str(random_row.get('overall-ratings', 'N/A'))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/department_sentiment', methods=['GET'])
def department_sentiment():
    """Get department-wise sentiment analysis"""
    result = get_department_sentiment()
    if result is None:
        return jsonify({'error': 'Department analysis not available'}), 404
    return jsonify({'departments': result})

@app.route('/model_metrics', methods=['GET'])
def model_metrics():
    """Get model evaluation metrics"""
    return jsonify(MODEL_METRICS)

@app.route('/export_report', methods=['POST'])
def export_report():
    """Export analysis report as CSV"""
    try:
        data = request.get_json()
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Timestamp', 'Original Feedback', 'Cleaned Text', 'Sentiment',
            'Positive Confidence', 'Neutral Confidence', 'Negative Confidence',
            'Polarity Score', 'Themes', 'Emotions', 'AI Summary'
        ])
        
        # Write data
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data.get('feedback', ''),
            data.get('cleaned_text', ''),
            data.get('sentiment', ''),
            data.get('confidence', {}).get('positive', 0),
            data.get('confidence', {}).get('neutral', 0),
            data.get('confidence', {}).get('negative', 0),
            data.get('polarity_score', 0),
            ', '.join(data.get('themes', [])),
            json.dumps(data.get('emotions', {})),
            data.get('ai_summary', '')
        ])
        
        # Create file response
        output.seek(0)
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'feedback_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'dataset_loaded': dataset_df is not None
    })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    
    print("\n" + "="*50)
    print("Starting Feature-Rich Flask Server...")
    print("="*50)
    print("Server: http://localhost:5000")
    print("API: http://localhost:5000/analyze_feedback")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
