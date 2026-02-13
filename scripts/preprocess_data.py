import os
from datasets import load_dataset
import json

def download_and_save():
    # We will use BC5CDR (Chemicals/Diseases) and NCBI-Disease
    datasets_to_fetch = {
        "bc5cdr": "tner/bc5cdr",
        "ncbi_disease": "ncbi/ncbi_disease"
    }

    raw_dir = "./data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    for name, hf_path in datasets_to_fetch.items():
        print(f"Fetching {name}...")
        dataset = load_dataset(hf_path)
        
        dataset_path = os.path.join(raw_dir, name)
        os.makedirs(dataset_path, exist_ok=True)
        
        for split in dataset.keys():
            # Save as JSON for human-readability and easy inspection
            output_file = os.path.join(dataset_path, f"{split}.json")
            dataset[split].to_json(output_file)
            print(f"  - Saved {split} to {output_file}")

if __name__ == "__main__":
    download_and_save()
