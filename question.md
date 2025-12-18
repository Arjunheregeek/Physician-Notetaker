👋 Welcome! The objective of this exercise is to build an AI system for **medical transcription, NLP-based summarization, and sentiment analysis**. 

Below is a sample physician patient conversation - 
Based on the conversation you have to build an **NLP pipeline** for extracting key medical details, and analyze **patient sentiment and intent.** 

> **Physician:** *Good morning, Ms. Jones. How are you feeling today?*
> 
> 
> **Patient:** *Good morning, doctor. I’m doing better, but I still have some discomfort now and then.*
> 
> **Physician:** *I understand you were in a car accident last September. Can you walk me through what happened?*
> 
> **Patient:** *Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.*
> 
> **Physician:** *That sounds like a strong impact. Were you wearing your seatbelt?*
> 
> **Patient:** *Yes, I always do.*
> 
> **Physician:** *What did you feel immediately after the accident?*
> 
> **Patient:** *At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.*
> 
> **Physician:** *Did you seek medical attention at that time?*
> 
> **Patient:** *Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home.*
> 
> **Physician:** *How did things progress after that?*
> 
> **Patient:** *The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.*
> 
> **Physician:** *That makes sense. Are you still experiencing pain now?*
> 
> **Patient:** *It’s not constant, but I do get occasional backaches. It’s nothing like before, though.*
> 
> **Physician:** *That’s good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?*
> 
> **Patient:** *No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.*
> 
> **Physician:** *And how has this impacted your daily life? Work, hobbies, anything like that?*
> 
> **Patient:** *I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.*
> 
> **Physician:** *That’s encouraging. Let’s go ahead and do a physical examination to check your mobility and any lingering pain.*
> 
> [**Physical Examination Conducted**]
> 
> **Physician:** *Everything looks good. Your neck and back have a full range of movement, and there’s no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.*
> 
> **Patient:** *That’s a relief!*
> 
> **Physician:** *Yes, your recovery so far has been quite positive. Given your progress, I’d expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.*
> 
> **Patient:** *That’s great to hear. So, I don’t need to worry about this affecting me in the future?*
> 
> **Physician:** *That’s right. I don’t foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you’re on track for a full recovery.*
> 
> **Patient:** *Thank you, doctor. I appreciate it.*
> 
> **Physician:** *You’re very welcome, Ms. Jones. Take care, and don’t hesitate to reach out if you need anything.*
> 

## **1. Medical NLP Summarization**

**Task:** Implement an NLP pipeline to **extract medical details** from the transcribed conversation.

### **📍 Deliverables:**

1. **Named Entity Recognition (NER):** Extract **Symptoms, Treatment, Diagnosis, Prognosis** using `spaCy` or `transformers`.
2. **Text Summarization:** Convert the transcript into a **structured medical report**.
3. **Keyword Extraction:** Identify **important medical phrases** (e.g., "whiplash injury," "physiotherapy sessions").

**📍 Sample Input (Raw Transcript):**

```
text
CopyEdit
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.

```

**📍 Expected Output (Structured Summary in JSON Format):**

```json
json
CopyEdit
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["10 physiotherapy sessions", "Painkillers"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery expected within six months"
}

```

**📍 Questions:**

- How would you handle **ambiguous or missing medical data** in the transcript?
- What **pre-trained NLP models** would you use for medical summarization?

## **2. Sentiment & Intent Analysis**

**Task:** Implement **sentiment analysis** to detect patient concerns and reassurance needs.

### **📍 Deliverables:**

1. **Sentiment Classification:** Use a **Transformer-based model** (e.g., `BERT`, `DistilBERT`) to classify **Patient Sentiment** as `Anxious`, `Neutral`, or `Reassured`.
2. **Intent Detection:** Identify **patient intent** (e.g., “Seeking reassurance,” “Reporting symptoms,” “Expressing concern”).

**📍 Sample Input (Patient’s Dialogue):**

```
text
CopyEdit
"I'm a bit worried about my back pain, but I hope it gets better soon."

```

**📍 Expected Output (JSON):**

```json
json
CopyEdit
{
  "Sentiment": "Anxious",
  "Intent": "Seeking reassurance"
}

```

**📍 Questions:**

- How would you fine-tune **BERT** for medical sentiment detection?
- What datasets would you use for training a **healthcare-specific** sentiment model?

## **3. SOAP Note Generation (Bonus)**

(A SOAP note is **a structured way for healthcare professionals to document patient information**. SOAP stands for Subjective, Objective, Assessment, and Plan.)

**Task:** Implement an **AI model that converts transcribed text into a structured SOAP note** format. (Note: This is a bonus section) 

### **📍 Deliverables:**

1. **Automated SOAP Note Generation** based on the conversation transcript.
2. **Logical Mapping** of **Subjective, Objective, Assessment, and Plan** sections.
3. **Text Structuring & Formatting** to ensure clinical readability.

**📍 Sample Input (Transcript):**

```
text
CopyEdit
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.

```

**📍 Expected Output (SOAP Note in JSON Format):**

```json
json
CopyEdit
{
  "Subjective": {
    "Chief_Complaint": "Neck and back pain",
    "History_of_Present_Illness": "Patient had a car accident, experienced pain for four weeks, now occasional back pain."
  },
  "Objective": {
    "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
    "Observations": "Patient appears in normal health, normal gait."
  },
  "Assessment": {
    "Diagnosis": "Whiplash injury and lower back strain",
    "Severity": "Mild, improving"
  },
  "Plan": {
    "Treatment": "Continue physiotherapy as needed, use analgesics for pain relief.",
    "Follow-Up": "Patient to return if pain worsens or persists beyond six months."
  }
}

```

**📍 Questions:**

- How would you train an NLP model to **map medical transcripts into SOAP format**?
- What **rule-based or deep-learning** techniques would improve the accuracy of SOAP note generation?

## **Submission Instructions -**

- **Python code (Jupyter Notebook / .py files)**
- **README.md** with setup instructions