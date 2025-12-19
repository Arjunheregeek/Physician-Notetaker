# Medical NLP Pipeline

## 📋 Executive Summary
This project implements a robust **Medical Natural Language Processing (NLP) Pipeline** designed to automate the extraction of clinical insights from doctor-patient conversations. It automates three critical tasks:
1.  **Clinical Entity Extraction**: Identifying specific medical terms (Diseases, Chemicals).
2.  **Sentiment & Intent Analysis**: Understanding patient emotional state and underlying needs.
3.  **Automated Documentation**: Generating structured SOAP notes and summaries using State-of-the-Art LLMs.

---

## 🏗 System Architecture & Technology Stack

The pipeline is modularized into three core components, chosen for their specific strengths in the medical domain.

### 1. Named Entity Recognition (NER)
*   **Technology**: `scispacy` (en_ner_bc5cdr_md)
*   **Role**: Specialized Biomedical extraction.
*   **Rational**: Unlike distinct generic NER models (like standard Spacy), `scispacy` is specifically trained on the **BC5CDR corpus** (BioCreative V CDR), enabling it to distinguish between **Diseases** (e.g., "Whiplash") and **Chemicals/Treatments** (e.g., "Tylenol") with high precision.

### 2. Sentiment & Intent Analysis
*   **Technology**: Hugging Face Transformers (`distilbert`, `bart-large-mnli`)
*   **Role**: Mental health & interaction analysis.
*   **Rational**:
    *   **Sentiment**: We use a fine-tuned DistilBERT model to gauge patient anxiety levels.
    *   **Intent (Zero-Shot)**: We utilize `bart-large-mnli` for **Zero-Shot Classification**. This allows us to detect intents (e.g., "Seeking Reassurance") without requiring a labeled dataset of medical intents, solving the "cold start" problem in medical AI.

### 3. Clinical Generation (SOAP Notes)
*   **Technology**: OpenAI GPT-4o integration.
*   **Role**: Complex reasoning and structuring.
*   **Rational**: Structured clinical documentation requires deep semantic understanding of context (separating a patient's *past* history from the doctor's *future* plan). Large Language Models offer superior coherent reasoning for this "sorting" task compared to rule-based systems.

---

## ⚡️ Detailed Setup & Installation Guide

This project is built using Python 3.11+. Follow these steps to deploy the application locally.

### Prerequisites
*   **Python 3.10** or higher.
*   **OpenAI API Key** (Required for the SOAP generation module).

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd medical_nlp_pipeline
```

### Step 2: Configure Environment
1.  Create a `.env` file in the root directory.
2.  Add your OpenAI API key:
    ```bash
    OPENAI_API_KEY="sk-..."
    ```

### Step 3: Install Dependencies
We recommend using a virtual environment.

**Option A: Using Standard pip**
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install Core Dependencies
pip install -U pip setuptools wheel
pip install spacy scispacy transformers torch pandas openai numpy python-dotenv

# Install Specialized Medical Models (Direct Download)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
```

**Option B: Using uv (High Speed)**
```bash
uv pip install spacy scispacy transformers torch pandas openai numpy python-dotenv
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
```

---

## 🚀 Usage

### Running the Demo
The repository includes a `demo.py` script that processes a sample medical transcript validation.

```bash
python demo.py
```

**Expected Output:**
The script will output a JSON object containing:
1.  Extracted Entities (Diseases, Drugs).
2.  Sentiment Analysis (Anxious/Reassured).
3.  A fully formatted SOAP Note.
4.  A structured Clinical Summary.

---

## 🧠 Technical Deep Dive & Design Decisions
*( Addressing Assignment Requirements )*

### Part 1: Handling Data Ambiguity & Missing Info
In a clinical setting, **Precision** (correctness) trumps **Recall** (finding everything).
*   **Ambiguity**: We employ confidence thresholding. If the model is uncertain (<85% confidence), the entity is flagged as `Unknown` rather than guessing, which prevents hallucinations.
*   **Missing Data**: We strictly map missing information to `null` fields in the JSON schema. We do **not** use statistical imputation for clinical facts to ensure patient safety.

### Part 2: Customizing Architectures for Medicine
*   **Adaptation**: Standard BERT models trained on Wikipedia fail on medical jargon. We utilize **Domain Adaptation** principles by selecting models pre-trained on biomedical corpora (PubMed, MIMIC-III).
*   **Fine-Tuning**: For sentiment, we utilize a **Transfer Learning** approach: freezing the language-understanding base of BERT and training only a lightweight classification head on medical sentiment labels.

### Part 3: Seq2Seq for Clinical Notes
*   **Model Choice**: We treat SOAP note generation as a **Sequence-to-Sequence (Seq2Seq)** translation task.
*   **Accuracy Enforcement**: To ensure the output matches hospital formats, we utilize **Constrained Decoding** (JSON Mode), which mathematically restricts the model to only output tokens that form valid JSON structures.
