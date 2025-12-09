AI-Driven Employee Feedback Analysis System

An NLP/Gen-AI based web application that automatically processes employee feedback and provides sentiment classification and workplace theme extraction for HR insights.

🎯 Features

Automated Sentiment Analysis: Classifies feedback as positive, negative, or neutral

Theme Extraction: Identifies key HR-related themes (salary, workload, culture, management, etc.)

Real-time Analysis: Web-based UI for instant feedback analysis

Comprehensive Preprocessing: Text cleaning, tokenization, lemmatization

Model Training: Trainable Naive Bayes or Logistic Regression models

Visualization: Word cloud generation for theme visualization

📋 Prerequisites

Python 3.8 or higher

pip package manager

🚀 Installation

Clone or download this repository

Install required packages:

pip install -r requirements.txt


Download NLTK data:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

📊 Dataset

The project uses employee_reviews.csv, containing:

Employee feedback (summary, pros&cons)

Star ratings (overall-ratings)

Job title, review dates, metadata

🔧 Usage
Step 1: Preprocess the Data
python preprocess_data.py


This will:

Combine summary + pros/cons

Clean text (lowercase, remove emojis/URLs/numbers/punctuation)

Remove stopwords, tokenize, lemmatize

Save output to processed_data.csv

Step 2: Prepare Dataset with Sentiment Labels
python prepare_dataset.py


Rating mapping:

1–2 → negative

3 → neutral

4–5 → positive

Output saved to final_dataset.csv.

Step 3: Train the Model
python train_model.py


This will:

Vectorize text using TF-IDF

Train Naive Bayes model

Evaluate performance

Save model (sentiment_model.pkl) and vectorizer (vectorizer.pkl)

To use Logistic Regression instead:

model, vectorizer = train_model(model_type='logistic_regression', use_smote=False)

Step 4: Extract Themes
python extract_themes.py


Generates:

Top HR themes

wordcloud.png

extracted_keywords.txt

Step 5: Run the Web Application
python app.py


Web UI: http://localhost:5000

API Endpoint: http://localhost:5000/analyze_feedback

🌐 API Usage
POST /analyze_feedback

Request:

{
  "feedback": "I feel management is supportive but workload is stressful!"
}


Response:

{
  "sentiment": "negative",
  "themes": ["management", "workload", "stress"],
  "confidence": {
    "positive": 0.123,
    "neutral": 0.234,
    "negative": 0.643
  }
}

GET /health

Checks if the model is loaded correctly.

📁 Project Structure
.
├── employee_reviews.csv
├── preprocess_data.py
├── prepare_dataset.py
├── train_model.py
├── extract_themes.py
├── app.py
├── templates/
│   └── index.html
├── requirements.txt
├── README.md
├── processed_data.csv
├── final_dataset.csv
├── sentiment_model.pkl
├── vectorizer.pkl
├── extracted_keywords.txt
└── wordcloud.png

🧪 Testing
Manual Testing

Examples tested automatically:

"Management is supportive but workload is stressful."

"Salary is good but pressure is high."

"Great company culture."

"Work environment is okay."

"Terrible management and no work-life balance."

Web UI Testing

Run app

Enter sample feedback

Click Analyze Feedback

View sentiment and themes

📈 Model Evaluation

Training script prints:

Accuracy

Precision

Recall

F1-score

Classification report

Confusion matrix

🎨 Features Implemented
Data Preprocessing

Lowercase conversion

Emoji removal

URL removal

Punctuation & number removal

Stopword removal

Tokenization

Lemmatization

Dataset Preparation

Ratings → sentiment mapping

Class balance check

SMOTE support

Model Training

TF-IDF vectorization

Naive Bayes & Logistic Regression

80/20 train-test split

Model persistence

Model Evaluation

Accuracy, precision, recall, F1

Confusion matrix

Manual test cases

Theme Extraction

Top keywords using TF-IDF

Word cloud generation

Backend API

Flask REST API

/analyze_feedback route

Error handling

/health route

Web UI

Modern Bootstrap interface

Real-time feedback analysis

Sentiment and theme visualization

🔍 Example Output

Input:

Salary is good but workload pressure is too high


Output:

Sentiment: negative

Themes: ["salary", "workload", "pressure"]

Confidence:

Positive: 0.15

Neutral: 0.20

Negative: 0.65

🛠️ Troubleshooting
Model Not Found

Run:

python train_model.py

NLTK Errors
nltk.download('punkt')

Port Already in Use

Change port:

app.run(debug=True, host='0.0.0.0', port=5001)

📝 Notes

Uses TF-IDF with unigrams + bigrams

Naive Bayes is default

Logistic Regression gives better accuracy

Themes extracted using keyword-based TF-IDF

🚀 Deployment

For Render/Railway/PythonAnywhere:

Ensure requirements.txt is complete

Upload model files

Update app run command:

app.run(debug=False, host='0.0.0.0', port=5000)

📄 License

This project is for educational purposes.

👥 Author

AI-Driven Employee Feedback Analysis System
Built for HR insights and employee engagement analysis.

Happy Analyzing! 🎉
