# AI-Driven Employee Feedback Analysis System

An NLP/Gen-AI based web application that automatically processes employee feedback and provides sentiment classification and workplace theme extraction for HR insights.

## 🎯 Features

- **Automated Sentiment Analysis**: Classifies feedback as positive, negative, or neutral
- **Theme Extraction**: Identifies key HR-related themes (salary, workload, culture, management, etc.)
- **Real-time Analysis**: Web-based UI for instant feedback analysis
- **Comprehensive Preprocessing**: Text cleaning, tokenization, lemmatization
- **Model Training**: Trainable Naive Bayes or Logistic Regression models
- **Visualization**: Word cloud generation for theme visualization

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🚀 Installation

1. **Clone or download this repository**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (will be downloaded automatically on first run, but you can also run):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

## 📊 Dataset

The project uses `employee_reviews.csv` which contains:
- Employee feedback in `summary` and `pros&cons` columns
- Overall ratings (1-5 stars) in `overall-ratings` column
- Additional metadata (job title, dates, etc.)

## 🔧 Usage

### Step 1: Preprocess the Data

Clean and normalize the text data:

```bash
python preprocess_data.py
```

This will:
- Combine `summary` and `pros&cons` into a single feedback column
- Clean text (lowercase, remove emojis/URLs/numbers/punctuation)
- Remove stopwords
- Apply tokenization and lemmatization
- Save processed data to `processed_data.csv`

### Step 2: Prepare Dataset with Sentiment Labels

Convert ratings to sentiment labels:

```bash
python prepare_dataset.py
```

This will:
- Convert ratings (1-2 = negative, 3 = neutral, 4-5 = positive)
- Check class balance
- Save final dataset to `final_dataset.csv`

### Step 3: Train the Model

Train the sentiment classification model:

```bash
python train_model.py
```

This will:
- Split data (80% train, 20% test)
- Apply TF-IDF vectorization
- Train Naive Bayes model (default)
- Evaluate model performance
- Save model to `sentiment_model.pkl` and vectorizer to `vectorizer.pkl`
- Run manual tests with example sentences

**Note**: To use Logistic Regression instead, modify `train_model.py`:
```python
model, vectorizer = train_model(model_type='logistic_regression', use_smote=False)
```

### Step 4: Extract Themes

Extract top HR-related keywords:

```bash
python extract_themes.py
```

This will:
- Extract top 10 HR-related keywords using TF-IDF
- Generate a word cloud visualization (`wordcloud.png`)
- Save keywords to `extracted_keywords.txt`

### Step 5: Run the Web Application

Start the Flask server:

```bash
python app.py
```

The application will be available at:
- **Web UI**: http://localhost:5000
- **API Endpoint**: http://localhost:5000/analyze_feedback

## 🌐 API Usage

### POST /analyze_feedback

Analyze employee feedback and get sentiment + themes.

**Request:**
```json
{
  "feedback": "I feel management is supportive but workload is stressful!"
}
```

**Response:**
```json
{
  "sentiment": "negative",
  "themes": ["management", "workload", "stress"],
  "confidence": {
    "positive": 0.123,
    "neutral": 0.234,
    "negative": 0.643
  }
}
```

### GET /health

Check if the model is loaded and server is running.

## 📁 Project Structure

```
.
├── employee_reviews.csv          # Original dataset
├── preprocess_data.py            # Data preprocessing script
├── prepare_dataset.py             # Dataset preparation with sentiment labels
├── train_model.py                # Model training script
├── extract_themes.py             # Theme extraction script
├── app.py                        # Flask backend API
├── templates/
│   └── index.html                # Web UI
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── processed_data.csv            # Generated: Cleaned data
├── final_dataset.csv             # Generated: Dataset with sentiment labels
├── sentiment_model.pkl           # Generated: Trained model
├── vectorizer.pkl                # Generated: TF-IDF vectorizer
├── extracted_keywords.txt        # Generated: Top keywords
└── wordcloud.png                 # Generated: Word cloud visualization
```

## 🧪 Testing

### Manual Testing

After training, the model is automatically tested with example sentences:
- "I feel management is supportive but workload is stressful!"
- "Salary is good but workload pressure is too high"
- "Great company culture and amazing benefits"
- "The work environment is okay, nothing special"
- "Terrible management and no work life balance"

### Web UI Testing

1. Open http://localhost:5000
2. Enter feedback text
3. Click "Analyze Feedback"
4. View sentiment classification and extracted themes

## 📈 Model Evaluation

The training script outputs:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision score
- **Recall**: Weighted recall score
- **F1-Score**: Weighted F1 score
- **Classification Report**: Per-class metrics
- **Confusion Matrix**: Classification breakdown

## 🎨 Features Implemented

✅ **Data Preprocessing**
- Lowercase conversion
- Emoji removal
- URL removal
- Special character and punctuation removal
- Number removal
- Stopword removal
- Tokenization
- Lemmatization

✅ **Dataset Preparation**
- Rating to sentiment conversion (1-2 = negative, 3 = neutral, 4-5 = positive)
- Class balance checking
- SMOTE support for imbalanced datasets

✅ **Model Training**
- TF-IDF vectorization
- Naive Bayes classifier
- Logistic Regression support
- 80/20 train-test split
- Model persistence (pickle/joblib)

✅ **Model Evaluation**
- Accuracy, Precision, Recall, F1-score
- Classification report
- Confusion matrix
- Manual testing with real sentences

✅ **Theme Extraction**
- Top 10 HR-related keywords
- TF-IDF based extraction
- Word cloud visualization

✅ **Backend API**
- Flask REST API
- POST /analyze_feedback endpoint
- Error handling
- Health check endpoint

✅ **Web UI**
- Modern, responsive design
- Real-time feedback analysis
- Sentiment visualization
- Theme display
- Example test cases

## 🔍 Example Output

**Input:**
```
"Salary is good but workload pressure is too high"
```

**Output:**
- **Sentiment**: negative
- **Themes**: ["salary", "workload", "pressure"]
- **Confidence**: 
  - Positive: 0.15
  - Neutral: 0.20
  - Negative: 0.65

## 🛠️ Troubleshooting

### Model files not found
If you see "Model not loaded" error:
1. Make sure you've run `train_model.py` first
2. Check that `sentiment_model.pkl` and `vectorizer.pkl` exist in the project directory

### NLTK data missing
If you see NLTK errors:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Port already in use
If port 5000 is busy, modify `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port
```

## 📝 Notes

- The model uses TF-IDF vectorization with unigrams and bigrams
- Default model is Naive Bayes (recommended for first-time use)
- For better accuracy, consider using Logistic Regression or fine-tuning DistilBERT
- Theme extraction uses keyword matching and can be enhanced with topic modeling

## 🚀 Deployment

For deployment to platforms like Render, Railway, or PythonAnywhere:

1. Ensure all dependencies are in `requirements.txt`
2. Set up environment variables if needed
3. Update `app.py` to use production settings:
   ```python
   app.run(debug=False, host='0.0.0.0', port=5000)
   ```
4. Make sure model files (`sentiment_model.pkl`, `vectorizer.pkl`) are included in deployment

## 📄 License

This project is for educational purposes.

## 👥 Author

AI-Driven Employee Feedback Analysis System
Built for HR insights and employee engagement analysis.

---

**Happy Analyzing! 🎉**

#   A I - D r i v e n - E m p l o y e e - F e e d b a c k - A n a l y s i s - S y s t e m  
 