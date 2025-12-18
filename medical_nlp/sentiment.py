from transformers import pipeline

class MedicalSentiment:
    def __init__(self):
        """
        Initialize the Sentiment Analysis and Intent Detection pipelines.
        """
        print("Loading Sentiment models...")
        # Generic sentiment model
        self.sentiment_pipeline = pipeline(
            "text-classification", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Zero-shot classification for Intent
        self.intent_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the text.
        Returns: 'Anxious', 'Neutral', or 'Reassured' mapped from positive/negative.
        """
        # DistilBERT gives Positive/Negative. We need to map to the assignment labels.
        # This is a heuristic mapping since we don't have a custom medical sentiment model.
        result = self.sentiment_pipeline(text)[0]
        label = result['label']
        score = result['score']
        
        # Heuristic mapping
        if label == "NEGATIVE":
            # High confidence negative -> Anxious
            if score > 0.8:
                return "Anxious"
            else:
                return "Neutral" # Mild negative/concern
        else:
            # POSITIVE -> Reassured
            return "Reassured"

    def detect_intent(self, text, candidate_labels=None):
        """
        Detect patient intent using Zero-Shot Classification.
        """
        if candidate_labels is None:
            candidate_labels = ["Seeking reassurance", "Reporting symptoms", "Expressing concern", "Asking question"]
            
        result = self.intent_pipeline(text, candidate_labels)
        
        # Return the top label
        return result['labels'][0]
