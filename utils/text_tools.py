import re
from textblob import TextBlob

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    return s.strip()

def get_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0

def count_keywords(series, n=20):
    words = " ".join(series.astype(str)).lower().split()
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]