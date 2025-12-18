import os
import json
from openai import OpenAI

class SOAPGenerator:
    def __init__(self, api_key=None):
        """
        Initialize OpenAI client.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("WARNING: No OpenAI API Key found. SOAP generation will fail.")
        
        self.client = OpenAI(api_key=self.api_key)

    def generate_soap_note(self, transcript):
        """
        Generate a structured SOAP note from the transcript using LLM.
        """
        system_prompt = """
        You are an expert medical scribe. 
        Your task is to convert the provided Doctor-Patient conversation transcript into a structured SOAP Note in JSON format.
        
        The JSON must have the following keys:
        - "Subjective": { "Chief_Complaint": "...", "History_of_Present_Illness": "..." }
        - "Objective": { "Physical_Exam": "...", "Observations": "..." }
        - "Assessment": { "Diagnosis": "...", "Severity": "..." }
        - "Plan": { "Treatment": "...", "Follow-Up": "..." }
        
        Be strictly professional, clinical, and concise. Do not guess information not present in the text.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Transcript:\n{transcript}"}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            return {"error": str(e)}

    def summarize_conversation(self, transcript):
        """
        Generate a general structured summary JSON as per assignment Part 1.
        """
        system_prompt = """
        You are a medical AI assistant. Extract the following details from the conversation in JSON format:
        - "Patient_Name"
        - "Symptoms" (List)
        - "Diagnosis"
        - "Treatment" (List)
        - "Current_Status"
        - "Prognosis"
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Transcript:\n{transcript}"}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            return {"error": str(e)}
