import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

class BioNEREvaluator:
    """
    Comprehensive evaluation suite for Biomedical NER.
    Computes both token-level and strict entity-level metrics.
    """
    def __init__(self, id2label: Dict[int, str]):
        self.id2label = id2label
        self.label_list = [label for label in id2label.values()]
        self.ignore_index = -100

    def compute_metrics(self, true_labels: List[List[str]], pred_labels: List[List[str]]) -> Dict[str, Any]:
        """
        Computes both Entity-level (seqeval) and Token-level (sklearn) metrics.
        """
        # 1. Entity-Level Metrics (Strict Span Matching)
        entity_precision = precision_score(true_labels, pred_labels)
        entity_recall = recall_score(true_labels, pred_labels)
        entity_f1 = f1_score(true_labels, pred_labels)
        
        logger.info("\n=== Entity-Level Classification Report (Seqeval) ===")
        print(seqeval_report(true_labels, pred_labels))

        # 2. Token-Level Metrics
        # Flatten lists for scikit-learn
        flat_true = [tag for seq in true_labels for tag in seq]
        flat_pred = [tag for seq in pred_labels for tag in seq]

        # Calculate macro token metrics, ignoring 'O' to see actual entity token performance
        labels_to_evaluate = [l for l in self.label_list if l != 'O']
        tok_p, tok_r, tok_f1, _ = precision_recall_fscore_support(
            flat_true, flat_pred, labels=labels_to_evaluate, average='macro', zero_division=0
        )

        logger.info(f"\n=== Token-Level Metrics (Macro, excluding 'O') ===")
        logger.info(f"Precision: {tok_p:.4f} | Recall: {tok_r:.4f} | F1: {tok_f1:.4f}")

        return {
            "entity_f1": entity_f1,
            "entity_precision": entity_precision,
            "entity_recall": entity_recall,
            "token_f1": tok_f1,
            "flat_true": flat_true,
            "flat_pred": flat_pred
        }

    def plot_confusion_matrix(self, flat_true: List[str], flat_pred: List[str], save_path: str = "confusion_matrix.png"):
        """Plots a token-level confusion matrix to visualize class confusion."""
        cm = confusion_matrix(flat_true, flat_pred, labels=self.label_list)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_list, yticklabels=self.label_list)
        plt.title('Token-Level Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")

    def plot_learning_curves(self, train_losses: List[float], val_losses: List[float], val_f1s: List[float], save_path: str = "learning_curves.png"):
        """Visualizes training and validation progression."""
        epochs = range(1, len(train_losses) + 1)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot Losses
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(epochs, train_losses, 'r--', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.legend(loc='upper left')

        # Plot F1 on secondary axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Entity F1 Score', color='tab:blue')
        ax2.plot(epochs, val_f1s, 'b-', marker='o', label='Val Entity F1')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.legend(loc='upper right')

        plt.title('Training Loss and Validation Entity F1')
        fig.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Learning curves saved to {save_path}")

    def extract_entities(self, tokens: List[str], tags: List[str]) -> List[Dict[str, str]]:
        """Helper to stitch BIO tags back into human-readable entity spans."""
        entities = []
        current_entity = []
        current_type = None

        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                if current_entity:
                    entities.append({"entity": " ".join(current_entity), "type": current_type})
                current_entity = [token]
                current_type = tag.split("-")[1]
            elif tag.startswith("I-") and current_entity and current_type == tag.split("-")[1]:
                # Remove '##' for WordPiece subwords
                clean_token = token.replace("##", "") if token.startswith("##") else token
                if token.startswith("##"):
                    current_entity[-1] += clean_token
                else:
                    current_entity.append(clean_token)
            else:
                if current_entity:
                    entities.append({"entity": " ".join(current_entity), "type": current_type})
                    current_entity = []
                    current_type = None
                    
        # Catch entity at the very end
        if current_entity:
            entities.append({"entity": " ".join(current_entity), "type": current_type})
            
        return entities
