import os
import sys
from datasets import load_dataset

def main():
    # 1. Define Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, "data", "raw", "bc5cdr")
    
    print(f"📂 Target Directory: {raw_data_dir}")
    os.makedirs(raw_data_dir, exist_ok=True)

    # 2. Download from Hugging Face
    try:
        print("Searching for 'tner/bc5cdr' on Hugging Face...")
        # we use 'tner/bc5cdr' because it's already in BIO format
        dataset = load_dataset("tner/bc5cdr")
        
        for split in dataset.keys():
            file_path = os.path.join(raw_data_dir, f"{split}.json")
            dataset[split].to_json(file_path)
            print(f"✅ Created: {file_path}")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        sys.exit(1)

    # 3. Final Verification
    files = os.listdir(raw_data_dir)
    print(f"\n🚀 Success! Files in raw/bc5cdr: {files}")

if __name__ == "__main__":
    main()

