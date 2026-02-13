import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerFast
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BioNERDataset(Dataset):
    """
    PyTorch Dataset wrapper for the tokenized biomedical NER data.
    Ensures data is formatted correctly for the DataLoader.
    """
    def __init__(self, tokenized_dataset):
        """
        Args:
            tokenized_dataset: A Hugging Face dataset split (e.g., train) 
                               that has already been tokenized and aligned.
        """
        self.dataset = tokenized_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetches a single example. We only extract the fields required 
        by the transformer model.
        """
        item = self.dataset[idx]
        
        # We do not convert to tensors here. The DataCollator will handle 
        # the conversion to batched PyTorch tensors.
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"]
        }


def create_dataloaders(
    tokenized_datasets: Dict[str, Any], 
    tokenizer: PreTrainedTokenizerFast, 
    batch_size: int = 16
) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits.
    Implements dynamic padding for efficient VRAM usage.
    """
    logger.info(f"Creating DataLoaders with batch size: {batch_size}")
    
    # The collator handles dynamic padding. 
    # It pads input_ids with the tokenizer's pad_token_id (usually 0)
    # It pads labels with -100 so padded tokens are ignored in the loss computation.
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt" # Ensures the output is strictly PyTorch tensors
    )
    
    dataloaders = {}
    
    for split in ["train", "validation", "test"]:
        if split not in tokenized_datasets:
            continue
            
        dataset = BioNERDataset(tokenized_datasets[split])
        
        # Shuffle only the training data to break correlations between consecutive batches
        shuffle = True if split == "train" else False
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=data_collator,
            pin_memory=True # Speeds up host-to-device (CPU to GPU) memory transfers
        )
        logger.info(f"Initialized {split} dataloader with {len(dataloaders[split])} batches.")
        
    return dataloaders

# --- Execution Example ---
if __name__ == "__main__":
    # Assuming 'tokenizer' and 'tokenized_dataset_dict' are available from previous steps
    # mock_tokenizer = ...
    # mock_tokenized_data = {"train": [...], "validation": [...]}
    
    # dataloaders = create_dataloaders(mock_tokenized_data, mock_tokenizer, batch_size=16)
    #
    # To verify GPU compatibility in the training loop, you would do:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for batch in dataloaders["train"]:
    #     input_ids = batch["input_ids"].to(device)
    #     attention_mask = batch["attention_mask"].to(device)
    #     labels = batch["labels"].to(device)
    #     break # Just testing the first batch
    pass
