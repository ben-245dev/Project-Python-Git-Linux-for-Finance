import yfinance as yf
from textblob import TextBlob
import pandas as pd

def get_news_sentiment(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    news = ticker.news
    
    sentiment_score = 0
    count = 0
    news_data = []

    for item in news:
        title = item.get('title', '')
        if not title: continue
            
        # feeling from (-1 to +1)
        analysis = TextBlob(title)
        score = analysis.sentiment.polarity
        
        sentiment_score += score
        count += 1
        news_data.append({"Title": title, "Score": score})

    avg_sentiment = sentiment_score / count if count > 0 else 0
    
    # Interpretation
    if avg_sentiment > 0.1: label = "Positive 🟢"
    elif avg_sentiment < -0.1: label = "NNegative 🔴"
    else: label = "Neutral ⚪"
    
    return label, avg_sentiment, pd.DataFrame(news_data)