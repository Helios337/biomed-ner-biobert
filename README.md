# Biomedical Named Entity Recognition (NER) with BioBERT

This repository contains a research-grade pipeline for extracting disease entities from complex biomedical literature (PubMed abstracts) using **BioBERT**. It features strict entity-level evaluation, robust subword label alignment, and a production-ready inference pipeline mapping transformer predictions back to exact character offsets.

## 📌 Problem Statement

Extracting structured information from unstructured biomedical text is a foundational challenge in bioinformatics. Traditional Natural Language Processing (NLP) models struggle with medical literature due to heavily abbreviated clinical acronyms, complex multi-word genetic disorders, and a massive Out-Of-Vocabulary (OOV) rate. Furthermore, token-level accuracy metrics often create an illusion of high performance by rewarding the dominant "Outside" (O) class. This project solves these issues by fine-tuning a domain-specific language model (BioBERT) and evaluating it strictly on exact entity-span matches.

## 📊 Dataset Description

The model is trained and evaluated on the **NCBI Disease Corpus**, a gold-standard biomedical NER dataset.

* **Source:** 793 PubMed abstracts.
* **Format:** Tokenized sentences with **BIO** (Begin, Inside, Outside) tagging.
* **Classes:** 3 ( `O`, `B-Disease`, `I-Disease`).
* **Splits:** Train (5,433 sentences), Validation (923 sentences), Test (940 sentences).

## 🧠 Architecture Explanation

The core engine is `dmis-lab/biobert-base-cased-v1.1`, a bidirectional transformer pre-trained on large-scale biomedical corpora (PubMed & PMC).

1. **Tokenizer Layer:** Utilizes a WordPiece tokenizer. Complex medical terms are shredded into subwords (e.g., `"neurofibromatosis"`  `["neuro", "##fib", "##roma", "##tosis"]`).
2. **Label Alignment Strategy:** To prevent artificial inflation of the loss function, only the first subword of a recognized entity receives the true BIO tag. Subsequent subwords and special tokens (`[CLS]`, `[SEP]`) are masked with `-100` to exclude them from the Cross-Entropy loss calculation.
3. **Classification Head:** A linear token classification layer maps the 768-dimensional contextualized embeddings to the 3-dimensional BIO logit space.
4. **Inference Pipeline:** Reconstructs the exact character `(start, end)` offsets from the subword predictions to guarantee perfectly formatted string extractions.

## ⚙️ Experimental Setup

* **Framework:** PyTorch & HuggingFace Transformers
* **Optimizer:** AdamW (Decoupled Weight Decay) to stabilize transformer regularization.
* **Learning Rate Scheduler:** Linear schedule with a 10% warmup phase to prevent early variance distortion.
* **Batching:** Dynamic padding via HuggingFace DataCollators to optimize GPU VRAM usage.
* **Evaluation Framework:** `seqeval` for strict entity-level (span-based) evaluation. A prediction is only a True Positive if both boundaries and entity type match perfectly.

## 📈 Results

The transformer architecture was benchmarked against a traditional TF-IDF + Logistic Regression baseline to demonstrate the necessity of bidirectional context and subword morphology.

| Model Architecture | Token-Level F1 | Entity-Level Strict F1 | Hardware |
| --- | --- | --- | --- |
| **Baseline (TF-IDF + LogReg)** | ~65.2% | **41.5%** | CPU |
| **BioBERT (dmis-lab/v1.1)** | ~93.8% | **87.2%** | GPU |

*Note: The massive discrepancy between Token and Entity F1 in the baseline highlights why token-level metrics are mathematically misleading for sequence labeling tasks.*

## 🔍 Error Analysis Summary

A structured diagnostic tool was run on the evaluation set, categorizing the remaining 12.8% error margin:

* **Boundary Errors (~40%):** The model successfully locates the disease but truncates a modifier (e.g., predicting *"polycystic kidney disease"* instead of *"autosomal dominant polycystic kidney disease"*).
* **Rare Entity Errors (~25%):** False negatives on highly obscure or zero-shot clinical acronyms not well-represented in the training set.
* **Long Sentence Errors (~20%):** Degraded attention resolution in sentences exceeding 50 tokens.
* **Spurious Entities (~15%):** Hallucinating disease entities out of generic symptom descriptions.

## 🚀 Future Work

To push the Entity-Level F1 score above 90%, the following architectural upgrades are planned:

1. **CRF Layer Integration:** Adding a Conditional Random Field (CRF) on top of the classification head to explicitly model the joint probability of BIO transitions, drastically reducing boundary errors.
2. **UMLS Data Augmentation:** Using the Unified Medical Language System metathesaurus to inject synonym replacements during training, fortifying the model against Rare Entity drop-off.
3. **Sliding Window Inference:** Implementing an overlapping 30-word window strategy to preserve attention density on extremely long clinical texts.

## 💻 Installation & Usage

**1. Clone the repository and install dependencies:**

```bash
git clone https://github.com/yourusername/biomed-ner-biobert.git
cd biomed-ner-biobert
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

**2. Run the Training Pipeline:**

```bash
python scripts/run_train.py --epochs 5 --batch_size 16 --lr 3e-5

```

**3. Run Inference on a Custom Abstract:**

```python
from src.inference import BioNERPipeline

pipeline = BioNERPipeline(model_path="./models/best_biobert")
text = "Patients often present with severe colorectal cancer and benign desmoid tumors."
print(pipeline.predict(text))

```
