import logging
import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import classification_report as seqeval_report

logger = logging.getLogger(__name__)

class BaselineNER:
    """
    TF-IDF + Logistic Regression baseline for biomedical NER.
    Uses an n-gram context window to simulate local sequence awareness.
    """
    def __init__(self, context_window: int = 1):
        self.context_window = context_window
        # Use character n-grams to handle OOV medical terms partially
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=50000)
        self.classifier = LogisticRegression(
            multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=1000, 
            class_weight='balanced', # Crucial: 'O' class dominates heavily
            n_jobs=-1
        )
        self.id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}

    def _prepare_token_features(self, sentences: List[List[str]], labels: List[List[int]] = None) -> Tuple[List[str], List[str], List[int]]:
        """
        Flattens sentences into individual tokens and creates a "context string" 
        for each token to be passed to TF-IDF.
        """
        context_strings = []
        flat_labels = []
        sentence_lengths = []

        for i, sentence in enumerate(sentences):
            sentence_lengths.append(len(sentence))
            
            for j, token in enumerate(sentence):
                # Create a context window: [prev_word, current_word, next_word]
                start = max(0, j - self.context_window)
                end = min(len(sentence), j + self.context_window + 1)
                
                context = " ".join(sentence[start:end])
                context_strings.append(context)
                
                if labels is not None:
                    # Map integer label to BIO string immediately for easier evaluation later
                    flat_labels.append(self.id2label[labels[i][j]])

        return context_strings, flat_labels, sentence_lengths

    def train(self, train_sentences: List[List[str]], train_labels: List[List[int]]):
        """Trains the TF-IDF vectorizer and Logistic Regression classifier."""
        logger.info("Preparing training data features...")
        X_text, y_train, _ = self._prepare_token_features(train_sentences, train_labels)

        logger.info("Fitting TF-IDF Vectorizer...")
        X_train = self.vectorizer.fit_transform(X_text)

        logger.info("Training Logistic Regression...")
        self.classifier.fit(X_train, y_train)
        logger.info("Baseline training complete.")

    def evaluate(self, test_sentences: List[List[str]], test_labels: List[List[int]]):
        """Evaluates using both token-level and strict entity-level metrics."""
        logger.info("Preparing test data features...")
        X_text, y_test_flat, sentence_lengths = self._prepare_token_features(test_sentences, test_labels)

        logger.info("Predicting...")
        X_test = self.vectorizer.transform(X_text)
        y_pred_flat = self.classifier.predict(X_test)

        # 1. Token-Level Evaluation (Scikit-Learn)
        logger.info("\n=== Baseline: Token-Level Classification Report ===")
        print(classification_report(y_test_flat, y_pred_flat, labels=["B-Disease", "I-Disease"]))

        # 2. Reconstruct sequences for Entity-Level Evaluation
        y_test_seqs = []
        y_pred_seqs = []
        pointer = 0
        
        for length in sentence_lengths:
            y_test_seqs.append(y_test_flat[pointer: pointer + length])
            y_pred_seqs.append(list(y_pred_flat[pointer: pointer + length]))
            pointer += length

        # 3. Entity-Level Evaluation (Seqeval)
        logger.info("\n=== Baseline: Entity-Level Classification Report (Seqeval) ===")
        print(seqeval_report(y_test_seqs, y_pred_seqs))
        
        return seqeval_f1(y_test_seqs, y_pred_seqs)

# --- Execution Example ---
if __name__ == "__main__":
    # Mock usage assuming dataset is parsed into lists of lists
    # baseline = BaselineNER(context_window=1)
    # baseline.train(train_sentences, train_labels)
    # baseline.evaluate(test_sentences, test_labels)
    pass
