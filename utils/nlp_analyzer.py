from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)

class NLPAnalyzer:
    def __init__(self):
        pass

    def analyze_text(self, text):
        """
        Analyze movie description for sentiment and keywords.
        Returns a dictionary with sentiment metrics and keywords.
        """
        if not text:
            return None
            
        try:
            blob = TextBlob(text)
            
            # Sentiment Analysis
            sentiment_score = blob.sentiment.polarity # -1.0 to 1.0
            subjectivity_score = blob.sentiment.subjectivity # 0.0 to 1.0
            
            # Keyword Extraction (Noun Phrases)
            keywords = list(blob.noun_phrases)
            
            # Classify Sentiment
            if sentiment_score > 0.1:
                sentiment_label = "Positive"
            elif sentiment_score < -0.1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
                
            return {
                "sentiment_score": sentiment_score,
                "subjectivity_score": subjectivity_score,
                "sentiment_label": sentiment_label,
                "keywords": keywords[:5], # Return top 5 noun phrases
                "word_count": len(text.split())
            }
        except Exception as e:
            logger.error(f"NLP Analysis failed: {e}")
            return None
