from .ner import MedicalNER
from .sentiment import MedicalSentiment
from .soap import SOAPGenerator

class MedicalPipeline:
    def __init__(self, openai_api_key=None):
        self.ner = MedicalNER()
        self.sentiment = MedicalSentiment()
        self.soap = SOAPGenerator(api_key=openai_api_key)
        
    def process_conversation(self, transcript):
        """
        Run the full pipeline on a transcript.
        """
        print("\n--- Running Medical NLP Pipeline ---")
        
        # 1. NER Extraction (using scispacy)
        print("1. Extracting Entities (NER)...")
        entities = self.ner.extract_entities(transcript)
        keywords = self.ner.extract_keywords(transcript)
        
        # 2. General Summarization (using LLM)
        print("2. Generating Summary...")
        summary_json = self.soap.summarize_conversation(transcript)
        
        # 3. SOAP Note (using LLM)
        print("3. Generating SOAP Note...")
        soap_json = self.soap.generate_soap_note(transcript)
        
        # 4. Sentiment (using Transformers)
        # We can analyze the patient's lines specifically if we parse the dialogue,
        # but for this assignment, we might just analyze the whole text or a sample.
        # Let's assume we want to analyze the overall patient tone.
        print("4. Analyzing Sentiment...")
        sentiment = self.sentiment.analyze_sentiment(transcript[:500]) # Analyze first 500 chars as proxy or pass full text
        intent = self.sentiment.detect_intent(transcript)
        
        return {
            "ner_entities": entities,
            "keywords": keywords,
            "structured_summary": summary_json,
            "soap_note": soap_json,
            "sentiment_analysis": {
                "Sentiment": sentiment,
                "Intent": intent
            }
        }
