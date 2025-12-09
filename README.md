# 🤖 AI-Driven Employee Feedback Analysis System

A comprehensive NLP and Machine Learning-based web application that automatically processes employee feedback and provides sentiment classification, emotion detection, theme extraction, and AI-generated HR insights.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project analyzes employee feedback using advanced Natural Language Processing (NLP) techniques and Machine Learning models. It provides HR departments with actionable insights through:

- **Sentiment Analysis**: Classifies feedback as positive, negative, or neutral
- **Emotion Detection**: Identifies emotions like joy, stress, anger, frustration, motivation
- **Theme Extraction**: Extracts key workplace themes (salary, workload, culture, management, etc.)
- **AI-Generated Summaries**: Provides HR-focused insights and recommendations
- **Department Analysis**: Department-wise sentiment breakdown
- **Interactive Web UI**: Modern, responsive dashboard for real-time analysis

## ✨ Features

### Core Features
- ✅ **Automated Sentiment Classification** using trained ML models (Naive Bayes/Logistic Regression)
- ✅ **Multi-emotion Detection** (joy, stress, anger, frustration, motivation, neutral)
- ✅ **HR Theme Extraction** (salary, workload, culture, management, stress, growth, team, support, benefits, work-life balance)
- ✅ **AI-Powered HR Summaries** with actionable recommendations
- ✅ **Department-wise Sentiment Analysis** with visualizations
- ✅ **Real-time Web Interface** with dark mode support
- ✅ **Export Functionality** (CSV reports)
- ✅ **Model Performance Metrics** dashboard

### Technical Features
- ✅ **Text Preprocessing**: Lowercase conversion, emoji/URL removal, tokenization, lemmatization
- ✅ **TF-IDF Vectorization** with unigrams and bigrams
- ✅ **SMOTE Support** for handling imbalanced datasets
- ✅ **Model Persistence** using joblib/pickle
- ✅ **RESTful API** with Flask backend
- ✅ **Responsive Design** with Bootstrap 5 and Chart.js

## 📁 Project Structure

```
Gen-AI/
├── Gen/                          # Main project directory
│   ├── app.py                    # Flask backend API (main application)
│   ├── employee_feedback_analysis.py  # Core analysis functions
│   ├── preprocess_data.py        # Data preprocessing script
│   ├── prepare_dataset.py        # Dataset preparation with sentiment labels
│   ├── train_model.py           # Model training script
│   ├── extract_themes.py        # Theme extraction script
│   ├── run_pipeline.py          # Complete pipeline runner
│   ├── employee_reviews.csv      # Original dataset
│   ├── final_dataset.csv         # Generated: Dataset with sentiment labels
│   ├── processed_data.csv        # Generated: Cleaned data
│   ├── sentiment_model.pkl      # Generated: Trained model
│   ├── vectorizer.pkl           # Generated: TF-IDF vectorizer
│   ├── extracted_keywords.txt   # Generated: Top keywords
│   ├── wordcloud.png            # Generated: Word cloud visualization
│   ├── requirements.txt         # Python dependencies
│   ├── README.md                # Detailed project documentation
│   ├── QUICKSTART.md            # Quick start guide
│   ├── templates/
│   │   └── index.html           # Web UI frontend
│   └── .gitignore              # Git ignore rules
├── README.md                    # This file (root documentation)
└── .gitignore                   # Root-level git ignore
```

## 🚀 Installation

### Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **pip** package manager
- **Git** (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Surya333gyle/Gen-AI-.git
cd Gen-AI-
```

### Step 2: Navigate to Project Directory

```bash
cd Gen
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

The application will automatically download NLTK data on first run, but you can also download manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## 🏃 Quick Start

### Option 1: Run Complete Pipeline

```bash
cd Gen
python run_pipeline.py
```

This will:
1. Preprocess the data
2. Prepare the dataset
3. Train the model
4. Extract themes
5. Start the web application

### Option 2: Step-by-Step Execution

#### Step 1: Preprocess Data
```bash
python preprocess_data.py
```

#### Step 2: Prepare Dataset
```bash
python prepare_dataset.py
```

#### Step 3: Train Model
```bash
python train_model.py
```

#### Step 4: Extract Themes
```bash
python extract_themes.py
```

#### Step 5: Run Web Application
```bash
python app.py
```

The application will be available at: **http://localhost:5000**

## 📖 Usage Guide

### Web Interface

1. **Open the application**: Navigate to `http://localhost:5000`
2. **Enter feedback**: Type or paste employee feedback text
3. **Analyze**: Click "Analyze Feedback" button
4. **View results**: See sentiment, emotions, themes, and AI summary
5. **Export**: Download analysis report as CSV

### API Usage

#### Analyze Feedback

**Endpoint**: `POST /analyze_feedback`

**Request**:
```json
{
  "feedback": "I feel management is supportive but workload is stressful!"
}
```

**Response**:
```json
{
  "sentiment": "negative",
  "confidence": {
    "positive": 0.123,
    "neutral": 0.234,
    "negative": 0.643
  },
  "polarity_score": -0.456,
  "themes": ["management", "workload", "stress"],
  "emotions": {
    "stress": 45.5,
    "frustration": 30.2
  },
  "ai_summary": "Employees express concerns that require attention. Workload and stress concerns suggest potential burnout risks...",
  "cleaned_text": "feel management supportive workload stressful"
}
```

#### Get Random Sample

**Endpoint**: `GET /random_sample`

Returns a random feedback sample from the dataset.

#### Department Sentiment

**Endpoint**: `GET /department_sentiment`

Returns department-wise sentiment statistics.

#### Model Metrics

**Endpoint**: `GET /model_metrics`

Returns model performance metrics (accuracy, precision, recall, F1-score, confusion matrix).

#### Health Check

**Endpoint**: `GET /health`

Checks if the model is loaded and server is running.

## 🔧 Technologies Used

### Backend
- **Flask 3.0.0** - Web framework
- **Flask-CORS 4.0.0** - Cross-origin resource sharing
- **scikit-learn 1.3.2** - Machine learning library
- **NLTK 3.8.1** - Natural language processing
- **TextBlob 0.17.1** - Text processing and sentiment analysis
- **pandas 2.1.3** - Data manipulation
- **numpy 1.24.3** - Numerical computing
- **joblib 1.3.2** - Model persistence

### Frontend
- **Bootstrap 5.3.2** - CSS framework
- **Chart.js 4.4.0** - Data visualization
- **Bootstrap Icons** - Icon library

### Data Processing
- **imbalanced-learn 0.11.0** - Handling imbalanced datasets (SMOTE)
- **wordcloud 1.9.2** - Word cloud generation
- **matplotlib 3.8.2** - Plotting and visualization
- **emoji 2.8.0** - Emoji handling

## 📊 Model Performance

### Current Model Metrics

- **Accuracy**: 72.1%
- **Precision**: 66.0%
- **Recall**: 72.1%
- **F1-Score**: 64.9%

### Model Details

- **Algorithm**: Multinomial Naive Bayes (default) / Logistic Regression (optional)
- **Vectorization**: TF-IDF with unigrams and bigrams
- **Train-Test Split**: 80/20
- **Classes**: Positive, Neutral, Negative

### Improving Model Performance

1. **Use Logistic Regression**: Modify `train_model.py`:
   ```python
   model, vectorizer = train_model(model_type='logistic_regression', use_smote=True)
   ```

2. **Fine-tune Hyperparameters**: Adjust TF-IDF parameters, n-gram range, etc.

3. **Use SMOTE**: Enable SMOTE for imbalanced datasets:
   ```python
   model, vectorizer = train_model(use_smote=True)
   ```

## 🎨 Features Showcase

### Sentiment Analysis
- Classifies feedback into positive, negative, or neutral
- Provides confidence scores for each class
- Uses TextBlob for polarity scoring

### Emotion Detection
- Detects 6 emotions: joy, stress, anger, frustration, motivation, neutral
- Provides percentage breakdown of detected emotions
- Visual representation with animated progress bars

### Theme Extraction
- Extracts top 5 HR-related themes from feedback
- Uses keyword matching and TF-IDF analysis
- Common themes: salary, workload, culture, management, stress, growth, team, support, benefits, work-life balance

### AI-Generated Summaries
- Provides HR-focused insights
- Includes actionable recommendations
- Considers sentiment, themes, and emotions

### Department Analysis
- Department-wise sentiment breakdown
- Visual charts and tables
- Extracts department from job titles

## 🛠️ Troubleshooting

### Model Files Not Found

**Error**: `FileNotFoundError: sentiment_model.pkl`

**Solution**: Run the training script first:
```bash
python train_model.py
```

### NLTK Data Missing

**Error**: `LookupError: Resource punkt not found`

**Solution**: Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Port Already in Use

**Error**: `Address already in use`

**Solution**: Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port
```

### Large File Warning (GitHub)

If you see warnings about large files (>50MB), consider using Git LFS:
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
```

## 📝 Development Notes

### Code Structure

- **app.py**: Main Flask application with all API endpoints
- **employee_feedback_analysis.py**: Core analysis functions (if separate module)
- **preprocess_data.py**: Text cleaning and preprocessing
- **prepare_dataset.py**: Dataset preparation and sentiment labeling
- **train_model.py**: Model training and evaluation
- **extract_themes.py**: Theme extraction and visualization

### Adding New Features

1. **New Emotion Detection**: Add keywords to `emotion_keywords` dictionary in `app.py`
2. **New Themes**: Add terms to `common_hr_terms` dictionary in `app.py`
3. **New Endpoints**: Add routes to `app.py` following Flask conventions

## 🚀 Deployment

### Local Deployment

```bash
python app.py
```

### Production Deployment

For platforms like Render, Railway, or Heroku:

1. Update `app.py`:
   ```python
   app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
   ```

2. Ensure all dependencies are in `requirements.txt`

3. Include model files (`sentiment_model.pkl`, `vectorizer.pkl`) in deployment

4. Set environment variables if needed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is for educational purposes.

## 👥 Author

**AI-Driven Employee Feedback Analysis System**

Built for HR insights and employee engagement analysis.

---

## 📚 Additional Resources

- [Detailed Documentation](Gen/README.md) - Comprehensive guide in the Gen folder
- [Quick Start Guide](Gen/QUICKSTART.md) - Quick reference guide
- [GitHub Repository](https://github.com/Surya333gyle/Gen-AI-)

---

**Happy Analyzing! 🎉**

For questions or issues, please open an issue on GitHub.
