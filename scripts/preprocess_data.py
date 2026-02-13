import os
from datasets import load_dataset

def download_and_save():
    # Using the standard BioCreative V CDR dataset
    dataset_name = "tner/bc5cdr"
    raw_dir = os.path.abspath("./data/raw/bc5cdr")
    
    print(f"🚀 Starting download for {dataset_name}...")
    os.makedirs(raw_dir, exist_ok=True)

    try:
        # download the dataset
        dataset = load_dataset(dataset_name)
        
        for split in dataset.keys():
            output_file = os.path.join(raw_dir, f"{split}.json")
            # Convert to pandas then to json for a clean structure
            dataset[split].to_json(output_file)
            print(f"✅ Saved {split} split to: {output_file}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    download_and_save()
