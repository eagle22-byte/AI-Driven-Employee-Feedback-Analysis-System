# Quick Start Guide

## 🚀 Fast Setup (5 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Complete Pipeline
```bash
python run_pipeline.py
```

This will automatically:
1. ✅ Preprocess the data
2. ✅ Prepare dataset with sentiment labels
3. ✅ Train the model
4. ✅ Extract themes

**OR** run steps individually:
```bash
python preprocess_data.py
python prepare_dataset.py
python train_model.py
python extract_themes.py
```

### Step 3: Start the Web Server
```bash
python app.py
```

### Step 4: Open Browser
Navigate to: **http://localhost:5000**

### Step 5: Test the System
1. Enter feedback text in the text area
2. Click "Analyze Feedback"
3. View sentiment and themes

## 🧪 Test the API

In a separate terminal, run:
```bash
python test_api.py
```

This will test the API with 5 example sentences.

## 📊 Expected Output Files

After running the pipeline, you should have:
- ✅ `processed_data.csv` - Cleaned dataset
- ✅ `final_dataset.csv` - Dataset with sentiment labels
- ✅ `sentiment_model.pkl` - Trained model
- ✅ `vectorizer.pkl` - TF-IDF vectorizer
- ✅ `extracted_keywords.txt` - Top HR keywords
- ✅ `wordcloud.png` - Word cloud visualization (optional)

## ⚠️ Troubleshooting

### "Model not loaded" error
→ Run `python train_model.py` first

### NLTK data missing
→ The scripts will download automatically, or run:
```python
import nltk
nltk.download('all')
```

### Port 5000 already in use
→ Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 📝 Example Test Cases

Try these in the web UI:

1. **"I feel management is supportive but workload is stressful!"**
   - Expected: Negative sentiment, themes: management, workload, stress

2. **"Salary is good but workload pressure is too high"**
   - Expected: Negative sentiment, themes: salary, workload, pressure

3. **"Great company culture and amazing benefits"**
   - Expected: Positive sentiment, themes: culture, benefit

4. **"The work environment is okay, nothing special"**
   - Expected: Neutral sentiment

5. **"Terrible management and no work life balance"**
   - Expected: Negative sentiment, themes: management, balance

## 🎯 Next Steps

1. Review model accuracy in the training output
2. Check `extracted_keywords.txt` for themes
3. Customize the model (try Logistic Regression)
4. Deploy to production (Render, Railway, etc.)

---

**Need help?** Check the full README.md for detailed documentation.

