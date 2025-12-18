# Medical NLP Assignment: Transcription, Summarization & Sentiment Analysis

## 📌 Project Overview
This project processes physician-patient conversations to extraction medical insights. It uses a **custom NLP pipeline** integrating:
1.  **Named Entity Recognition (NER)**: Uses `scispacy` (biomedical models) to extract diseases, drugs, and treatments.
2.  **Sentiment & Intent Analysis**: Uses Hugging Face `transformers` (DistilBERT & BART) to detect patient anxiety and intent.
3.  **SOAP Note Generation**: Uses **OpenAI GPT-4o** (via API) to generate clinical SOAP notes and summaries.

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.10+ (Recommended: 3.11)
- `uv` package manager (optional but recommended for speed) or standard `pip`.
- **OpenAI API Key**: Required for the Summarization and SOAP note generation modules.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd <repo-name>
    ```

2.  **Initialize Environment & Install Dependencies**:
    Using `uv` (Recommended):
    ```bash
    uv init
    uv venv --python 3.11
    source .venv/bin/activate
    uv pip install spacy scispacy transformers torch pandas openai numpy
    uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
    uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
    ```

### 🚀 Usage

1.  **Set your API Key**:
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

2.  **Run the Demo**:
    This runs the pipeline on the sample conversation from the assignment.
    ```bash
    python demo.py
    ```

---

## 🧠 Approach & Architecture

### 1. Medical NER (Named Entity Recognition)
We use **scispacy** (`en_ner_bc5cdr_md`), a model trained on the BC5CDR corpus, specifically optimized to recognize **Diseases** and **Chemicals**.
- **Input**: Raw text.
- **Output**: Structured list of detected medical entities.

### 2. Sentiment & Intent
We utilize **Transfer Learning**:
- **Sentiment**: `distilbert-base-uncased-finetuned-sst-2-english` provides sentiment scores. We apply a heuristic threshold to map simple "Negative" sentiment to "Anxious" in a medical context.
- **Intent**: `facebook/bart-large-mnli` (Zero-Shot Classification) maps the text to arbitrary labels like "Seeking reassurance" or "Reporting symptoms" without needing specific training data.

### 3. SOAP Note Generation (The "Modern" Way)
As recommended by the assignment ("If the company allows it, using an LLM... is the standard way"), we use **OpenAI GPT-4o**.
- We construct a **System Prompt** acting as a professional medical scribe.
- We force the output into a strict **JSON Schema** ensuring the `Subjective`, `Objective`, `Assessment`, and `Plan` sections are always present.

---

## 📚 Answers to Assignment Questions

### Part 1: Medical Summarization
**Q: How would you handle ambiguous or missing medical data?**
> **A:** In medical AI, hallucinations are dangerous. We follow a strict "do not guess" policy.
> 1.  **Explicit Flagging**: If a field (like "Diagnosis") is not explicitly mentioned, the system should return `null` or "Not Mentioned", rather than inferring it.
> 2.  **Human-in-the-Loop**: The UI should flag missing critical fields to the doctor for manual entry.
> 3.  **Confidence Scores**: We can output confidence scores for extracted entities; low confidence predictions should be flagged for review.

**Q: What pre-trained NLP models would you use for medical summarization?**
> **A:**
> - **Scientific/Clinical**: `BioBERT` or `ClinicalBERT` are standard for embeddings, but for generation (seq2seq), specifically fine-tuned T5 models (like `t5-base-clinical`) or specialized Llama-3 derivatives (like `MedLlama`) are best.
> - **General State-of-the-Art**: GPT-4o or Claude 3.5 Sonnet (as used here) currently outperform most smaller specialized models on zero-shot summarization tasks.

### Part 2: Sentiment Analysis
**Q: How would you fine-tune BERT for medical sentiment detection?**
> **A:**
> 1.  **Architecture**: Load a pre-trained `BertForSequenceClassification` model.
> 2.  **Freezing**: Freeze the lower encoding layers (embeddings) to retain general language knowledge.
> 3.  **Training**: Train only the final classification head (dense layer) on a labeled medical dataset.
> 4.  **Unfreezing (Optional)**: If enough data exists, slowly unfreeze upper layers with a very low learning rate to adapt to medical syntax.

**Q: What datasets would you use for training a healthcare-specific sentiment model?**
> **A:**
> - **MIMIC-III / MIMIC-IV**: Contains thousands of clinical notes (though sentiment labels might need to be derived).
> - **PubMedQA**: For general biomedical context.
> - **Consumer Health Question (CHQ) Corpus**: Excellent for patient-authored queries and sentiment.

### Part 3: SOAP Note Generation
**Q: How would you train an NLP model to map medical transcripts into SOAP format?**
> **A:**
> - **Task**: This is a Sequence-to-Sequence (Seq2Seq) task.
> - **Model**: Start with a strong summarizer like BART or T5.
> - **Data**: Create a dataset of (Transcript -> JSON SOAP Note) pairs.
> - **Fine-tuning**: Fine-tune the model to take the transcript as input and generate the JSON string as output.

**Q: What techniques improve accuracy?**
> **A:**
> - **Few-Shot Prompting**: (Used in this project) Giving the model examples of correct mappings in the prompt context.
> - **Constrained Decoding / JSON Mode**: Forcing the model to output valid JSON (as seen in the OpenAI API `response_format={"type": "json_object"}`).
> - **Rule-Based Post-Processing**: Using Regex to validate that keys like "Plan" or "Diagnosis" exist in the output.
