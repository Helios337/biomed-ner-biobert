import torch
import json
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BioNERPipeline:
    """
    Production-ready inference pipeline for Biomedical Named Entity Recognition.
    Handles raw text ingestion, subword merging, and JSON structuring.
    """
    def __init__(self, model_path: str = "dmis-lab/biobert-base-cased-v1.1"):
        logger.info(f"Loading NER Pipeline from: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer with use_fast=True to enable offset mapping
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Assuming standard BIO tags for Disease
        self.id2label = self.model.config.id2label
        if not self.id2label:
            # Fallback if config doesn't have it saved
            self.id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}

    def predict(self, text: str) -> str:
        """
        Takes a raw string, predicts entities, merges subwords, 
        and returns a formatted JSON string.
        """
        if not text.strip():
            return json.dumps({"text": text, "entities": []})

        # 1. Tokenize with offset mapping
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            return_offsets_mapping=True, 
            return_word_ids=True,
            truncation=True
        )
        
        # Extract metadata and move tensors to GPU
        offset_mapping = inputs.pop("offset_mapping")[0].numpy()
        word_ids = inputs.pop("word_ids")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Forward Pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        # 3. Align subwords to words
        # We take the predicted label of the FIRST subword of every word
        word_to_label = {}
        word_to_offsets = {}

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue  # Skip special tokens ([CLS], [SEP])
                
            start_char, end_char = offset_mapping[idx]
            if start_char == end_char:
                continue
                
            if word_idx not in word_to_label:
                # First subword of the word dictates the label
                word_to_label[word_idx] = self.id2label[predictions[idx]]
                word_to_offsets[word_idx] = [start_char, end_char]
            else:
                # Extend the character offset to encompass subsequent subwords
                word_to_offsets[word_idx][1] = end_char

        # 4. Stitch BIO tags into contiguous entities
        entities = self._extract_entities(text, word_to_label, word_to_offsets)

        # 5. Return structured JSON
        result = {
            "text": text,
            "entities": entities
        }
        return json.dumps(result, indent=4)

    def _extract_entities(self, text: str, word_to_label: Dict[int, str], word_to_offsets: Dict[int, List[int]]) -> List[Dict[str, Any]]:
        """
        Parses word-level BIO tags and offsets into exact string matches.
        """
        entities = []
        current_entity = None

        # Ensure we iterate in the correct order of words
        sorted_words = sorted(word_to_label.keys())

        for word_idx in sorted_words:
            tag = word_to_label[word_idx]
            start_char, end_char = word_to_offsets[word_idx]

            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = tag.split("-")[1]
                current_entity = {
                    "entity_type": entity_type,
                    "start_char": start_char,
                    "end_char": end_char,
                    "text": text[start_char:end_char]
                }
            
            elif tag.startswith("I-") and current_entity and current_entity["entity_type"] == tag.split("-")[1]:
                # Expand the current entity's boundary
                current_entity["end_char"] = end_char
                # Re-slice the original text to perfectly capture spaces/punctuation
                current_entity["text"] = text[current_entity["start_char"]:end_char]
                
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Catch trailing entity
        if current_entity:
            entities.append(current_entity)

        return entities

# --- Execution Example ---
if __name__ == "__main__":
    # In a real scenario, model_path would point to "./models/best_biobert"
    pipeline = BioNERPipeline()
    
    sample_abstract = (
        "We investigated the role of the APC gene in familial adenomatous polyposis. "
        "Patients often present with severe colorectal cancer and benign desmoid tumors. "
        "Treatment with non-steroidal anti-inflammatory drugs showed reduction in polyp burden."
    )
    
    logger.info("Running inference on sample abstract...\n")
    json_output = pipeline.predict(sample_abstract)
    print(json_output)
