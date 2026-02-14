import logging
from datasets import load_dataset, DatasetDict
from typing import Dict, List, Tuple

# Configure logging for reproducibility and debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NCBIDiseaseDataLoader:
    """
    Handles loading, parsing, and validating the NCBI Disease dataset.
    """
    def __init__(self):
        self.dataset_name = "ncbi_disease"
        # The NCBI dataset maps integers to BIO tags
        self.id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.dataset: DatasetDict = None

   def load_data(self) -> DatasetDict:
        """Loads the dataset and ensures splits are present."""
        logger.info(f"Downloading/Loading dataset: {self.dataset_name}")
        
        # ADD trust_remote_code=True HERE
        self.dataset = load_dataset(self.dataset_name, trust_remote_code=True)
        
        # Verify expected splits
        expected_splits = ["train", "validation", "test"]
        for split in expected_splits:
            if split not in self.dataset:
                raise ValueError(f"Missing required split: {split}")
                
        logger.info(f"Successfully loaded splits: {list(self.dataset.keys())}")
        return self.dataset
                
        logger.info(f"Successfully loaded splits: {list(self.dataset.keys())}")
        return self.dataset

    def validate_bio_sequence(self, split: str = "train") -> bool:
        """
        Iterates through a split to validate that no label corruption exists.
        Checks:
        1. Length match: len(tokens) == len(ner_tags)
        2. BIO constraints: 'I' must follow 'B' or 'I'.
        """
        logger.info(f"Validating BIO integrity for split: {split}...")
        data = self.dataset[split]
        
        for idx, row in enumerate(data):
            tokens = row['tokens']
            tags = row['ner_tags']
            
            # Constraint 1: Length match
            if len(tokens) != len(tags):
                logger.error(f"Length mismatch at index {idx}: {len(tokens)} tokens vs {len(tags)} tags.")
                return False
                
            # Constraint 2: BIO transitions
            for i, tag_id in enumerate(tags):
                current_tag = self.id2label[tag_id]
                if current_tag == "I-Disease":
                    if i == 0:
                        logger.error(f"Corruption at index {idx}: Sentence starts with I-Disease.")
                        return False
                    prev_tag = self.id2label[tags[i-1]]
                    if prev_tag == "O":
                        logger.error(f"Corruption at index {idx}: I-Disease follows O.")
                        return False
                        
        logger.info(f"Validation passed for {split}. No corruption detected.")
        return True

    def print_sample(self, split: str = "train", index: int = 0):
        """Prints a human-readable aligned sample from the dataset."""
        if self.dataset is None:
            self.load_data()
            
        sample = self.dataset[split][index]
        tokens = sample['tokens']
        tags = [self.id2label[tag_id] for tag_id in sample['ner_tags']]
        
        print(f"\n--- Sample from {split} dataset (Index {index}) ---")
        print(f"{'Token':<20} | {'BIO Label'}")
        print("-" * 35)
        for token, tag in zip(tokens, tags):
            print(f"{token:<20} | {tag}")
        print("-" * 35)


# --- Execution Example ---
if __name__ == "__main__":
    loader = NCBIDiseaseDataLoader()
    
    # 1. Load data
    dataset = loader.load_data()
    
    # 2. Print splits shapes
    for split in dataset.keys():
        print(f"Split '{split}' contains {len(dataset[split])} sentences.")
        
    # 3. Validate integrity
    loader.validate_bio_sequence("train")
    
    # 4. Print sample
    loader.print_sample("train", index=0)

