# scripts/train.py

import os
import yaml
import torch
import logging
import argparse
from src import (
    NCBIDiseaseDataLoader, 
    BioTokenizer, 
    BioNERModelFactory, 
    BioNERTrainer, 
    set_reproducibility
)
from src.dataset import create_dataloaders # Assuming dataset.py holds the DataLoader logic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config_path):
    # 1. Load Configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    set_reproducibility(config['seed'])
    
    # 2. Data Loading & Tokenization
    data_loader = NCBIDiseaseDataLoader()
    raw_dataset = data_loader.load_data()
    
    tokenizer_utils = BioTokenizer(config['model']['checkpoint'])
    
    logger.info("Tokenizing and aligning labels...")
    tokenized_datasets = raw_dataset.map(
        tokenizer_utils.tokenize_and_align_labels, 
        batched=True,
        remove_columns=raw_dataset["train"].column_names
    )
    
    dataloaders = create_dataloaders(
        tokenized_datasets, 
        tokenizer_utils.tokenizer, 
        batch_size=config['training']['batch_size']
    )

    # 3. Model Initialization
    factory = BioNERModelFactory(config['model']['checkpoint'])
    model = factory.create_model()
    
    # 4. Training
    trainer = BioNERTrainer(
        model=model,
        dataloaders=dataloaders,
        id2label=factory.id2label,
        device=factory.device,
        epochs=config['training']['epochs'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        max_grad_norm=config['training']['max_grad_norm'],
        patience=config['training']['patience'],
        output_dir=config['model']['output_dir']
    )
    
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BioNER Training")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
