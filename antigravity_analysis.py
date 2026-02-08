import pandas as pd
import numpy as np
import os
from transformers import pipeline
import emoji
import re
from tqdm import tqdm

def clean_comment(text):
    # Remove special characters but keep emojis
    text = str(text)
    # Extract emojis
    emojis = "".join(c for c in text if c in emoji.EMOJI_DATA)
    # Basic cleaning
    text = re.sub(r'http\S+', '', text) # Remove URLs
    return text, emojis

def run_deep_analysis():
    print("Starting Anti-Gravity Deep Sentiment Analysis Engine...")
    
    # 1. Load Data
    file_path = os.path.join('datasets', 'social_media_comments.csv')
    df = pd.read_csv(file_path)
    
    # 2. Preprocessing
    print("Preprocessing data (cleaning and emoji extraction)...")
    cleaned_data = [clean_comment(c) for c in df['comment']]
    df['cleaned_comment'] = [c[0] for c in cleaned_data]
    df['emojis'] = [c[1] for c in cleaned_data]
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['comment'])
    print(f"Removed {initial_len - len(df)} duplicate comments.")

    # 3. Deep Sentiment Analysis using Transformers
    # We use a context-aware BERT model
    print("Loading BERT Sentiment Model (this may take a moment)...")
    sentiment_task = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Note for Korean: Use model="beomi/kcbert-base" for social media Korean text
    
    results = []
    print("Analyzing sentiments...")
    for comment in tqdm(df['cleaned_comment']):
        # Truncate to 512 tokens (BERT limit)
        res = sentiment_task(comment[:512])[0]
        # Normalize score: LABEL_0 (Negative) -> -score, LABEL_1 (Positive) -> +score
        score = res['score']
        if res['label'] == 'NEGATIVE':
            score = -score
        results.append(score)
    
    df['sentiment_score'] = results
    
    # Categorize
    df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

    # 4. Impact Score Calculation
    # Since original data doesn't have likes/replies, we'll synthesize for demo impact
    print("Calculating Impact Scores...")
    np.random.seed(42)
    df['likes'] = np.random.randint(0, 1500, size=len(df))
    df['replies'] = np.random.randint(0, 200, size=len(df))
    
    # Impact Score = Sentiment Score * (Likes + Replies)
    df['impact_score'] = df['sentiment_score'] * (df['likes'] + df['replies'])
    
    # 5. Keyword & Topic Association
    # Simplified approach: Tokenize and associate with sentiment
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("Extracting key topics and associations...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_comment'])
    keywords = vectorizer.get_feature_names_out()
    
    # Find which keyword is present in which comment
    def get_main_keyword(text):
        for word in keywords:
            if word in text.lower():
                return word
        return "Others"
    
    df['main_topic'] = df['cleaned_comment'].apply(get_main_keyword)

    # 6. Save Processed Data
    output_path = 'datasets/processed_sentiment_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Analysis complete! Data saved to {output_path}")

if __name__ == "__main__":
    run_deep_analysis()
