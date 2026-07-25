"""Factory for BioBERT model instantiation and configuration."""

import torch
import logging
from transformers import AutoConfig, BertForTokenClassification

logger = logging.getLogger(__name__)


class BioNERModelFactory:
    """
    Factory class for instantiating and configuring BioBERT for token classification.

    Usage::

        factory = BioNERModelFactory()
        model = factory.create_model()
        factory.print_architecture(model)
    """

    def __init__(self, model_checkpoint: str = "dmis-lab/biobert-base-cased-v1.1"):
        self.model_checkpoint = model_checkpoint

        # NCBI Disease label space
        self.id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(self.id2label)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self) -> BertForTokenClassification:
        """
        Load pretrained BioBERT and attach a token classification head.

        Returns:
            Model ready for training or inference on the configured device.
        """
        logger.info("Initializing model from checkpoint: %s", self.model_checkpoint)
        logger.info("Targeting device: %s", self.device)

        config = AutoConfig.from_pretrained(
            self.model_checkpoint,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        # BioBERT v1.1 config lacks model_type — set explicitly for Transformers>=4.45
        if not hasattr(config, "model_type") or config.model_type is None:
            config.model_type = "bert"

        model = BertForTokenClassification.from_pretrained(
            self.model_checkpoint, config=config
        )
        model.to(self.device)
        logger.info("Model successfully loaded and moved to device.")
        return model

    def print_architecture(self, model: BertForTokenClassification) -> None:
        """Print the model architecture and verify the classification head."""
        print("\n=== BioBERT Token Classification Architecture ===")
        print(model)

        print("\n=== Classification Head Details ===")
        print(f"Input Features (d): {model.classifier.in_features}")
        print(f"Output Classes (K): {model.classifier.out_features}")

        assert (
            model.classifier.out_features == self.num_labels
        ), f"Expected {self.num_labels} outputs, got {model.classifier.out_features}."


# --- Execution Example ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    factory = BioNERModelFactory()
    model = factory.create_model()
    factory.print_architecture(model)