"""Production-ready inference pipeline for Biomedical Named Entity Recognition."""

import torch
import json
import logging
from typing import List, Dict, Any, Optional
from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig

logger = logging.getLogger(__name__)


class BioNERPipeline:
    """
    Production-ready inference pipeline for Biomedical Named Entity Recognition.

    Handles raw text ingestion, subword merging, offset reconstruction,
    and JSON-structured output for downstream integration.

    Usage::

        pipeline = BioNERPipeline(model_path="./models/best_biobert")
        result = pipeline.predict("Patient presented with colorectal cancer.")
        print(result)
    """

    def __init__(self, model_path: str = "dmis-lab/biobert-base-cased-v1.1"):
        logger.info("Loading NER Pipeline from: %s", model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        config = BertConfig.from_pretrained(model_path, num_labels=3,
                                                         id2label={0: "O", 1: "B-Disease", 2: "I-Disease"},
                                                         label2id={"O": 0, "B-Disease": 1, "I-Disease": 2})
        self.model = BertForTokenClassification.from_pretrained(model_path, config=config)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label or {0: "O", 1: "B-Disease", 2: "I-Disease"}

    def predict(self, text: str) -> str:
        """
        Predict disease entities in a raw text string.

        Args:
            text: Raw biomedical text (e.g., PubMed abstract).

        Returns:
            JSON string with ``text`` and ``entities`` list. Each entity has
            ``entity_type``, ``start_char``, ``end_char``, and ``text`` fields.
        """
        if not text or not text.strip():
            return json.dumps({"text": text, "entities": []})

        return json.dumps(self._predict_structured(text), indent=4)

    def predict_structured(self, text: str) -> Dict[str, Any]:
        """
        Predict disease entities and return a Python dict instead of JSON.

        Args:
            text: Raw biomedical text.

        Returns:
            Dictionary with keys ``text`` and ``entities``.
        """
        return self._predict_structured(text)

    def _predict_structured(self, text: str) -> Dict[str, Any]:
        """Internal method: predict and return structured dict."""
        # 1. Tokenize with offset mapping (NOT return_word_ids — that's not a valid arg)
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
        )

        # Extract metadata before moving to device
        offset_mapping = tokenized.pop("offset_mapping")[0].numpy()
        inputs = {k: v.to(self.device) for k, v in tokenized.items()}

        # Get word_ids from the tokenizer output
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        # 2. Align subwords to words using offset mapping
        word_to_label: Dict[int, str] = {}
        word_to_offsets: Dict[int, List[int]] = {}

        # Re-tokenize to get word_ids (since we popped offset_mapping)
        word_ref = self.tokenizer(text, return_offsets_mapping=True, truncation=True)
        word_ids = word_ref.word_ids()

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue  # Skip special tokens

            start_char, end_char = offset_mapping[idx]
            if start_char == end_char:
                continue

            if word_idx not in word_to_label:
                word_to_label[word_idx] = self.id2label[predictions[idx]]
                word_to_offsets[word_idx] = [int(start_char), int(end_char)]
            else:
                word_to_offsets[word_idx][1] = int(end_char)

        # 3. Stitch BIO tags into contiguous entities
        entities = self._extract_entities(text, word_to_label, word_to_offsets)

        return {"text": text, "entities": entities}

    def _extract_entities(
        self,
        text: str,
        word_to_label: Dict[int, str],
        word_to_offsets: Dict[int, List[int]],
    ) -> List[Dict[str, Any]]:
        """Parse word-level BIO tags and offsets into exact string matches."""
        entities: List[Dict[str, Any]] = []
        current_entity: Optional[Dict[str, Any]] = None

        for word_idx in sorted(word_to_label.keys()):
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
                    "text": text[start_char:end_char],
                }
            elif tag.startswith("I-") and current_entity and current_entity["entity_type"] == tag.split("-")[1]:
                current_entity["end_char"] = end_char
                current_entity["text"] = text[current_entity["start_char"]:end_char]
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities


# --- Execution Example ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipeline = BioNERPipeline()

    sample_abstract = (
        "We investigated the role of the APC gene in familial adenomatous polyposis. "
        "Patients often present with severe colorectal cancer and benign desmoid tumors. "
        "Treatment with non-steroidal anti-inflammatory drugs showed reduction in polyp burden."
    )

    logger.info("Running inference on sample abstract...\n")
    print(pipeline.predict(sample_abstract))