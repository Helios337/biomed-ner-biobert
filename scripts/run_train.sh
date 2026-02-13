#!/bin/bash
# scripts/run_train.sh

# Exit immediately if a command exits with a non-zero status
set -e 

echo "Starting Biomedical NER Training Pipeline..."

# Optional: Set CUDA device if you have multiple GPUs (e.g., use GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Run the training script and pipe output to a log file while also printing to terminal
python scripts/train.py --config configs/base_config.yaml | tee logs/training_run.log

echo "Training pipeline finished."
