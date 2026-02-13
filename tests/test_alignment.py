# tests/test_alignment.py

import pytest
from src.tokenizer_utils import BioTokenizer

def test_subword_label_alignment():
    """
    Tests that the BioTokenizer correctly assigns -100 to subwords 
    and special tokens, preserving the exact entity boundary.
    """
    tokenizer_utils = BioTokenizer("dmis-lab/biobert-base-cased-v1.1")
    
    # "neurofibromatosis" will be split into 4 subwords by BioBERT
    mock_batch = {
        "tokens": [["Severe", "neurofibromatosis", "observed"]],
        "ner_tags": [[0, 1, 0]] # O, B-Disease, O
    }
    
    result = tokenizer_utils.tokenize_and_align_labels(mock_batch)
    labels = result["labels"][0]
    
    # Expected logic:
    # [CLS] -> -100
    # Severe -> 0
    # neuro -> 1
    # ##fib -> -100
    # ##roma -> -100
    # ##tosis -> -100
    # observed -> 0
    # [SEP] -> -100
    
    assert labels[0] == -100, "[CLS] token must be ignored."
    assert labels[-1] == -100, "[SEP] token must be ignored."
    
    # Count how many valid labels exist (should exactly match the 3 original words)
    valid_labels = [l for l in labels if l != -100]
    assert len(valid_labels) == 3, f"Expected 3 valid labels, got {len(valid_labels)}."
    
    # The second valid label must be '1' (B-Disease)
    assert valid_labels[1] == 1, "Entity label was lost during subword splitting."
