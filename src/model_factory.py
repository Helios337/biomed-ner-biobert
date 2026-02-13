import torch
import logging
from transformers import AutoConfig, AutoModelForTokenClassification

logger = logging.getLogger(__name__)

class BioNERModelFactory:
    """
    A factory class to instantiate, configure, and prepare BioBERT 
    for token classification in a research environment.
    """
    def __init__(self, model_checkpoint: str = "dmis-lab/biobert-base-cased-v1.1"):
        self.model_checkpoint = model_checkpoint
        
        # Define the exact label space for the NCBI Disease dataset
        self.id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(self.id2label)
        
        # Detect GPU availability automatically
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self) -> AutoModelForTokenClassification:
        """
        Loads the pre-trained weights and initializes the classification head.
        """
        logger.info(f"Initializing model from checkpoint: {self.model_checkpoint}")
        logger.info(f"Targeting device: {self.device}")
        
        # 1. Load the configuration
        # Passing label mappings explicitly ensures the model's output layer 
        # is sized perfectly (K=3) and metadata is preserved in the saved model.
        config = AutoConfig.from_pretrained(
            self.model_checkpoint,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # 2. Instantiate the model with the customized config
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint,
            config=config
        )
        
        # 3. Move model to the correct hardware accelerator
        model.to(self.device)
        logger.info("Model successfully loaded and moved to device.")
        
        return model

    def print_architecture(self, model: AutoModelForTokenClassification):
        """Prints the model blueprint, highlighting the classification head."""
        print("\n=== BioBERT Token Classification Architecture ===")
        print(model)
        
        # Isolate and print just the classifier head to verify dimensions
        print("\n=== Classification Head Details ===")
        print(f"Expected Input Features (d): {model.classifier.in_features}")
        print(f"Output Classes (K): {model.classifier.out_features}")
        
        assert model.classifier.out_features == self.num_labels, \
            f"Architecture mismatch! Expected {self.num_labels} outputs, got {model.classifier.out_features}."

# --- Execution Example ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    factory = BioNERModelFactory()
    biobert_ner = factory.create_model()
    
    # Verify architecture
    factory.print_architecture(biobert_ner)
