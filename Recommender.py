import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Concatenate
from tensorflow.keras.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
import gunicorn
import json
import heapq

app = Flask(__name__)

# Enhanced Data Preprocessing
class MovieDataProcessor:
    def __init__(self, movies_path, ratings_path):
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        # Advanced text cleaning pipeline
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = [self.lemmatizer.lemmatize(word) for word in text.split()]
        return ' '.join(words)
    
    def process_data(self):
        # Merge and enhance datasets
        merged = pd.merge(self.ratings, self.movies, on='movieId')
        
        # Feature engineering
        merged['weighted_rating'] = merged['rating'] * merged['timestamp'].apply(
            lambda x: 1 + np.log1p(x)/np.log1p(merged['timestamp'].max()))
        
        # Handle missing values
        merged['genres'] = merged['genres'].fillna('Unknown')
        merged['title'] = merged['title'].fillna('Unknown Title')
        
        # Text preprocessing
        merged['clean_genres'] = merged['genres'].apply(self.clean_text)
        merged['clean_title'] = merged['title'].apply(self.clean_text)
        
        return merged

# Hybrid LSTM-GRU Model
class HybridRecommender(Model):
    def __init__(self, num_users, num_movies, embedding_size=128):
        super().__init__()
        self.user_embedding = Embedding(num_users, embedding_size)
        self.movie_embedding = Embedding(num_movies, embedding_size)
        self.lstm = LSTM(64, return_sequences=True)
        self.gru = GRU(64)
        self.dense = Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        user_id, movie_id, text_features = inputs
        user_vec = self.user_embedding(user_id)
        movie_vec = self.movie_embedding(movie_id)
        concat = Concatenate()([user_vec, movie_vec, text_features])
        lstm_out = self.lstm(concat)
        gru_out = self.gru(lstm_out)
        return self.dense(gru_out)

# Real-time Processing Engine
class RecommendationEngine:
    def __init__(self, model, tfidf_vectorizer, movie_data):
        self.model = model
        self.tfidf = tfidf_vectorizer
        self.movies = movie_data
        self.content_matrix = self._create_content_matrix()
        
    def _create_content_matrix(self):
        tfidf_matrix = self.tfidf.fit_transform(
            self.movies['clean_genres'] + ' ' + self.movies['clean_title'])
        return tfidf_matrix
    
    def hybrid_recommend(self, user_id, top_n=10):
        # Collaborative filtering predictions
        collab_preds = self.model.predict(user_id)
        
        # Content-based similarity
        user_hist = self._get_user_history(user_id)
        content_sim = cosine_similarity(
            self.content_matrix[user_hist], 
            self.content_matrix)
        
        # Hybrid scoring
        combined_scores = 0.7 * collab_preds + 0.3 * content_sim.mean(axis=0)
        return heapq.nlargest(top_n, enumerate(combined_scores), key=lambda x: x[1])

# Flask API Endpoints
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.json
    user_id = data['user_id']
    recommendations = engine.hybrid_recommend(user_id)
    return jsonify({
        'user_id': user_id,
        'recommendations': [
            {
                'movie_id': int(self.movies.iloc[idx]['movieId']),
                'title': self.movies.iloc[idx]['title'],
                'score': float(score)
            } for idx, score in recommendations
        ]
    })

# Heroku Deployment Setup
if __name__ == '__main__':
    # Initialize components
    processor = MovieDataProcessor('movies.csv', 'ratings.csv')
    processed_data = processor.process_data()
    
    # Model training (would include proper train/test split in production)
    model = HybridRecommender(
        num_users=processed_data['userId'].nunique(),
        num_movies=processed_data['movieId'].nunique()
    )
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Initialize recommendation engine
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    engine = RecommendationEngine(model, tfidf, processed_data)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
