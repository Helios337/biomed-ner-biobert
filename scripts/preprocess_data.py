import os
from datasets import load_dataset

def download_biomed_data(output_dir="./data/raw"):
    """
    Downloads gold-standard NER datasets from Hugging Face and 
    saves them to the local raw data directory.
    """
    datasets_to_load = {
        "bc5cdr": "tner/bc5cdr",
        "ncbi_disease": "ncbi/ncbi_disease"
    }

    for name, hf_path in datasets_to_load.items():
        print(f"--- Downloading {name} ---")
        dataset = load_dataset(hf_path)
        
        # Create subdirectory for the specific dataset
        dataset_path = os.path.join(output_dir, name)
        os.makedirs(dataset_path, exist_ok=True)
        
        for split in dataset.keys():
            file_path = os.path.join(dataset_path, f"{split}.json")
            dataset[split].to_json(file_path)
            print(f"Saved {split} split to {file_path}")

if __name__ == "__main__":
    download_biomed_data()
