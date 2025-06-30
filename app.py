from flask import Flask, request, render_template, jsonify
import torch
from transformers import BertTokenizer, BertModel
import joblib
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Set device to CPU for compatibility
device = torch.device('cpu')

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()

# Load pre-trained models
logreg_model = joblib.load('logistic_regression_model.pkl')
calibrator_model = joblib.load('calibrator_model.pkl')

# Preprocessing function for tweets
def cleaner(tweet):
    soup = BeautifulSoup(tweet, 'lxml')
    souped = soup.get_text()
    re1 = re.sub(r"(@|http://|https://|www|\\x)\\S*", " ", souped)
    re2 = re.sub("[^A-Za-z]+", " ", re1)
    tokens = nltk.word_tokenize(re2)
    lower_case = [t.lower() for t in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t, 'v') for t in filtered_result]
    return ' '.join(lemmas)

# Function to generate BERT embeddings
def get_cls_embeddings_batch(texts):
    inputs = bert_tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embeddings

# Function to predict gender
def predict_gender(tweet):
    try:
        cleaned_tweet = cleaner(tweet)
        embedding = get_cls_embeddings_batch([cleaned_tweet])[0]
        prob = logreg_model.predict_proba([embedding])[:, 1][0]
        calibrated_prob = calibrator_model.predict_proba([[prob]])[:, 1][0]
        gender = 'M' if calibrated_prob > 0.65 else 'F'
        return gender, calibrated_prob
    except Exception as e:
        return None, str(e)

# Homepage route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Prediction route for form submission
@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form.get('tweet', '')
    if not tweet:
        return render_template('result.html', error="Please enter a tweet.", tweet=tweet)
    gender, prob = predict_gender(tweet)
    if gender is None:
        return render_template('result.html', error=f"Error: {prob}", tweet=tweet)
    return render_template('result.html', gender=gender, prob=prob, tweet=tweet)

# API route for programmatic access
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    tweet = data.get('tweet', '')
    if not tweet:
        return jsonify({'error': 'Tweet is required'}), 400
    gender, prob = predict_gender(tweet)
    if gender is None:
        return jsonify({'error': prob}), 500
    return jsonify({'gender': gender, 'probability': prob})

if __name__ == '__main__':
    app.run(debug=True)