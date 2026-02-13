import logging
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

class BioTokenizer:
    """
    Handles BioBERT WordPiece tokenization and robust BIO label alignment.
    """
    def __init__(self, model_checkpoint: str = "dmis-lab/biobert-base-cased-v1.1"):
        self.model_checkpoint = model_checkpoint
        logger.info(f"Loading tokenizer: {self.model_checkpoint}")
        
        # We MUST use use_fast=True to access the word_ids() alignment mapping
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            self.model_checkpoint, 
            use_fast=True
        )
        self.ignore_index = -100

    def tokenize_and_align_labels(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Tokenizes inputs and aligns BIO labels with WordPiece subwords.
        Designed to be used with HF datasets.map(batched=True).
        """
        # is_split_into_words=True tells the tokenizer our input is already tokenized into words
        tokenized_inputs = self.tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True,
            padding=False # Padding is usually handled dynamically during collation
        )

        labels = []
        # Iterate over the batch
        for i, label in enumerate(examples["ner_tags"]):
            # word_ids maps each subword back to the index of its original word
            # Example: [None, 0, 1, 1, 1, 2, None] 
            # None represents special tokens like [CLS] and [SEP]
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens ([CLS], [SEP]) get -100
                    label_ids.append(self.ignore_index)
                elif word_idx != previous_word_idx:
                    # First subword of a new word gets the actual label
                    label_ids.append(label[word_idx])
                else:
                    # Subsequent subwords of the same word get -100
                    label_ids.append(self.ignore_index)
                    
                previous_word_idx = word_idx
                
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def validate_alignment(self, original_words: List[str], original_labels: List[int], aligned_labels: List[int], word_ids: List[int]):
        """
        Failsafe validation to ensure the number of non-ignored labels 
        matches the original number of words.
        """
        valid_label_count = sum(1 for label in aligned_labels if label != self.ignore_index)
        assert valid_label_count == len(original_words), \
            f"Alignment Corrupted! Original words: {len(original_words)}, Valid labels: {valid_label_count}"

    def show_alignment_example(self, words: List[str], labels: List[str], label2id: Dict[str, int]):
        """Visualizes the before/after state of tokenization and alignment."""
        logger.info("\n--- Tokenization & Label Alignment Example ---")
        
        # Convert string labels to IDs for the example
        label_ids = [label2id[l] for l in labels]
        
        # Create a mock batch dictionary
        mock_batch = {"tokens": [words], "ner_tags": [label_ids]}
        
        # Process
        tokenized = self.tokenize_and_align_labels(mock_batch)
        
        aligned_tokens = self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
        aligned_label_ids = tokenized["labels"][0]
        
        # Validate
        self.validate_alignment(words, label_ids, aligned_label_ids, tokenized.word_ids(batch_index=0))
        
        print(f"{'Subword Token':<20} | {'Aligned Label ID':<18} | {'Explanation'}")
        print("-" * 65)
        for token, label_id in zip(aligned_tokens, aligned_label_ids):
            if label_id == self.ignore_index:
                explanation = "Ignored (-100)" if token.startswith("##") or token in ['[CLS]', '[SEP]'] else "Ignored (subword)"
            else:
                # Find the original label string for context
                label_str = list(label2id.keys())[list(label2id.values()).index(label_id)]
                explanation = f"Valid Label ({label_str})"
                
            print(f"{token:<20} | {str(label_id):<18} | {explanation}")
        print("-" * 65)

# --- Execution Example ---
if __name__ == "__main__":
    tokenizer_utils = BioTokenizer()
    
    # Mock complex biomedical sentence
    sample_words = ["Severe", "neurofibromatosis", "was", "observed", "."]
    sample_labels = ["O", "B-Disease", "O", "O", "O"]
    id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}
    label2id = {v: k for k, v in id2label.items()}
    
    tokenizer_utils.show_alignment_example(sample_words, sample_labels, label2id)
