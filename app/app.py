from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
import string

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def load_or_train_model():
    """Load existing model or train a new one"""
    global model, vectorizer
    
    model_path = 'models/sentiment_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    
    # Check if models exist
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print("Loading existing models...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("Models loaded successfully!")
    else:
        print("Training new model...")
        # Load cleaned data
        data_path = 'data/cleaned_data.csv'
        
        if not os.path.exists(data_path):
            print(f"Error: {data_path} not found!")
            return False
        
        df = pd.read_csv(data_path)
        
        # Assume the CSV has 'text' and 'sentiment' columns
        # Adjust column names based on your actual CSV structure
        text_column = df.columns[0]  # First column is text
        sentiment_column = df.columns[1] if len(df.columns) > 1 else 'sentiment'
        
        # Preprocess text
        df['cleaned_text'] = df[text_column].apply(preprocess_text)
        
        # Create sentiment labels if not present
        if sentiment_column not in df.columns:
            # Create dummy sentiment based on text length (replace with actual labels)
            df['sentiment'] = 'positive'
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], 
            df[sentiment_column], 
            test_size=0.2, 
            random_state=42
        )
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save models
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print("Model trained and saved successfully!")
    
    return True

def predict_sentiment(text):
    """Predict sentiment for given text"""
    if model is None or vectorizer is None:
        return None, None
    
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Transform text
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    # Get confidence score
    confidence = max(probability) * 100
    
    return prediction, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        sentiment, confidence = predict_sentiment(text)
        
        if sentiment is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence, 2),
            'text': text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load or train model on startup
    if load_or_train_model():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load or train model. Exiting...")