import spacy
import scispacy

class MedicalNER:
    def __init__(self, model_name="en_ner_bc5cdr_md"):
        """
        Initialize the Medical NER system.
        
        Args:
            model_name (str): The scispacy model to load. 
                              Default is 'en_ner_bc5cdr_md' (Disease and Chemical).
        """
        print(f"Loading NER model: {model_name}...")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise OSError(f"Model '{model_name}' not found. Please install it first.")
        
    def extract_entities(self, text):
        """
        Extract medical entities from the text.
        
        Returns:
            dict: Structured dictionary of entities (Symptoms, Diagnosis, Treatment, etc.)
                  Note: The bc5cdr model mainly extracts DISEASES and CHEMICALS.
        """
        doc = self.nlp(text)
        
        entities = {
            "Diseases_and_Symptoms": [],
            "Treatments_and_Medications": []
        }
        
        # scispacy entity labels:
        # DISEASE: Diseases, symptoms
        # CHEMICAL: Drugs, chemicals
        
        for ent in doc.ents:
            if ent.label_ == "DISEASE":
                if ent.text not in entities["Diseases_and_Symptoms"]:
                    entities["Diseases_and_Symptoms"].append(ent.text)
            elif ent.label_ == "CHEMICAL":
                if ent.text not in entities["Treatments_and_Medications"]:
                    entities["Treatments_and_Medications"].append(ent.text)
        
        return entities

    def extract_keywords(self, text):
        """
        Extract meaningful noun chunks or keywords.
        """
        doc = self.nlp(text)
        # Filter for interesting nouns/adjectives
        keywords = [
            chunk.text for chunk in doc.noun_chunks 
            if any(t.pos_ in ["NOUN", "PROPN"] for t in chunk)
        ]
        return list(set(keywords))
