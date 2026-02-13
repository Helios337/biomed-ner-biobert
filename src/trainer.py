import os
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

logger = logging.getLogger(__name__)

def set_reproducibility(seed: int = 42):
    """Locks all sources of randomness for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic CuDNN behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} across all environments.")

class BioNERTrainer:
    """
    Research-grade training loop for Named Entity Recognition.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        id2label: Dict[int, str],
        device: torch.device,
        epochs: int = 5,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        patience: int = 2,
        output_dir: str = "./models/best_biobert"
    ):
        self.model = model
        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["validation"]
        self.id2label = id2label
        self.device = device
        
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.output_dir = output_dir
        self.ignore_index = -100

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Optimizer: AdamW
        # Exclude bias and LayerNorm weights from weight decay (Standard Transformer practice)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

        # 2. Scheduler: Linear with Warmup
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

    def _align_predictions(self, predictions: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Converts raw tensor predictions and labels back to list of strings (BIO tags).
        Dynamically strips out the -100 padding tokens.
        """
        preds = torch.argmax(predictions, dim=2).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        true_labels = []
        true_predictions = []

        for i in range(labels.shape[0]):
            true_label_seq = []
            true_pred_seq = []
            for j in range(labels.shape[1]):
                if labels[i, j] != self.ignore_index:
                    true_label_seq.append(self.id2label[labels[i, j]])
                    true_pred_seq.append(self.id2label[preds[i, j]])
            
            true_labels.append(true_label_seq)
            true_predictions.append(true_pred_seq)

        return true_predictions, true_labels

    def _evaluate(self) -> Dict[str, float]:
        """Runs validation and computes exact span-level metrics via Seqeval."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()

                preds, trues = self._align_predictions(outputs.logits, labels)
                all_preds.extend(preds)
                all_labels.extend(trues)

        # Calculate Entity-Level Metrics
        metrics = {
            "val_loss": total_loss / len(self.val_loader),
            "precision": precision_score(all_labels, all_preds),
            "recall": recall_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds)
        }
        return metrics

    def train(self):
        """Executes the training loop with Early Stopping."""
        logger.info("=== Starting Research-Grade Training ===")
        best_f1 = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            
            logger.info(f"\nEpoch {epoch}/{self.epochs}")
            progress_bar = tqdm(self.train_loader, desc="Training")

            for batch in progress_bar:
                self.optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_train_loss += loss.item()

                # Backward pass
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimization step
                self.optimizer.step()
                self.scheduler.step()
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(self.train_loader)
            
            # Validation phase
            val_metrics = self._evaluate()
            
            logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Validation (Entity-Level) -> Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")

            # Early Stopping & Checkpointing logic based purely on Strict F1
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                epochs_without_improvement = 0
                logger.info(f"🏆 New best F1 ({best_f1:.4f}). Saving checkpoint to {self.output_dir}...")
                self.model.save_pretrained(self.output_dir)
            else:
                epochs_without_improvement += 1
                logger.warning(f"No improvement. Early stopping patience: {epochs_without_improvement}/{self.patience}")

            if epochs_without_improvement >= self.patience:
                logger.error(f"🛑 Early stopping triggered at epoch {epoch}. Training halted.")
                break

        logger.info(f"Training Complete. Best Validation F1: {best_f1:.4f}")

# --- Execution Example ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # 1. Guarantee Reproducibility
    set_reproducibility(seed=42)
    
    # Assumptions: model, dataloaders, and id2label are properly instantiated from previous modules
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # trainer = BioNERTrainer(model, dataloaders, id2label, device)
    # trainer.train()
