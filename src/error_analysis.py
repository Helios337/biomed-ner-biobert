import logging
from typing import List, Dict, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

class NERErrorAnalyzer:
    """
    Diagnostic tool for extracting and categorizing NER failure modes.
    """
    def __init__(self, train_entities: List[str] = None, rare_threshold: int = 5, long_sentence_threshold: int = 40):
        self.rare_threshold = rare_threshold
        self.long_sentence_threshold = long_sentence_threshold
        
        # Build a frequency map of training entities to detect "Rare Entity" errors
        self.train_entity_counts = defaultdict(int)
        if train_entities:
            for ent in train_entities:
                self.train_entity_counts[ent.lower()] += 1

        self.error_counts = {
            "Boundary Errors": 0,
            "Rare Entity Errors (FN)": 0,
            "Spurious Entities (FP)": 0,
            "Missed Entities (FN)": 0,
            "Long Sentence Errors": 0,
            "Label Confusion": 0
        }
        self.detailed_errors = []

    def _extract_spans(self, tags: List[str]) -> Set[Tuple[int, int, str]]:
        """Converts BIO tags to a set of (start_idx, end_idx, type) tuples."""
        spans = set()
        start = -1
        current_type = None
        
        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if start != -1:
                    spans.add((start, i - 1, current_type))
                start = i
                current_type = tag.split("-")[1]
            elif tag == "O" or (tag.startswith("I-") and current_type != tag.split("-")[-1]):
                if start != -1:
                    spans.add((start, i - 1, current_type))
                    start = -1
                    current_type = None
                    
        if start != -1:
            spans.add((start, len(tags) - 1, current_type))
            
        return spans

    def analyze_predictions(self, sentences: List[List[str]], true_tags: List[List[str]], pred_tags: List[List[str]]):
        """Categorizes errors for a full evaluation set."""
        logger.info("Starting structured error analysis...")
        
        for tokens, trues, preds in zip(sentences, true_tags, pred_tags):
            true_spans = self._extract_spans(trues)
            pred_spans = self._extract_spans(preds)
            
            is_long_sentence = len(tokens) > self.long_sentence_threshold
            sentence_text = " ".join(tokens)
            
            # Analyze False Negatives and Boundary Errors
            for t_span in true_spans:
                t_start, t_end, t_type = t_span
                t_text = " ".join(tokens[t_start:t_end + 1])
                
                # Check for overlap
                overlapping_preds = [p for p in pred_spans if not (p[1] < t_start or p[0] > t_end)]
                
                if not overlapping_preds:
                    # Missed completely (FN)
                    if self.train_entity_counts[t_text.lower()] < self.rare_threshold:
                        self.error_counts["Rare Entity Errors (FN)"] += 1
                        error_type = "Rare Entity (FN)"
                    else:
                        self.error_counts["Missed Entities (FN)"] += 1
                        error_type = "Missed (FN)"
                        
                    self._log_error(error_type, t_text, "N/A", sentence_text, is_long_sentence)
                    
                else:
                    for p_span in overlapping_preds:
                        p_start, p_end, p_type = p_span
                        p_text = " ".join(tokens[p_start:p_end + 1])
                        
                        if t_start != p_start or t_end != p_end:
                            self.error_counts["Boundary Errors"] += 1
                            self._log_error("Boundary Error", t_text, p_text, sentence_text, is_long_sentence)
                        elif t_type != p_type:
                            self.error_counts["Label Confusion"] += 1
                            self._log_error(f"Label Confusion ({t_type} vs {p_type})", t_text, p_text, sentence_text, is_long_sentence)

            # Analyze False Positives (Spurious)
            for p_span in pred_spans:
                p_start, p_end, p_type = p_span
                overlapping_trues = [t for t in true_spans if not (t[1] < p_start or t[0] > p_end)]
                
                if not overlapping_trues:
                    p_text = " ".join(tokens[p_start:p_end + 1])
                    self.error_counts["Spurious Entities (FP)"] += 1
                    self._log_error("Spurious (FP)", "N/A", p_text, sentence_text, is_long_sentence)

    def _log_error(self, error_type: str, true_ent: str, pred_ent: str, sentence: str, is_long: bool):
        """Records error instances for final reporting."""
        if is_long:
            self.error_counts["Long Sentence Errors"] += 1
            
        self.detailed_errors.append({
            "Type": error_type,
            "True": true_ent,
            "Pred": pred_ent,
            "Sentence": sentence
        })

    def print_report(self):
        """Displays the categorized error frequencies."""
        print("\n=== Structured Error Analysis Report ===")
        print(f"{'Error Category':<30} | {'Frequency'}")
        print("-" * 45)
        for category, count in self.error_counts.items():
            print(f"{category:<30} | {count}")
        print("-" * 45)
        
        print("\n--- Sample Boundary Error ---")
        boundary_errors = [e for e in self.detailed_errors if e["Type"] == "Boundary Error"]
        if boundary_errors:
            sample = boundary_errors[0]
            print(f"Sentence: {sample['Sentence']}")
            print(f"True Entity: '{sample['True']}'")
            print(f"Predicted:   '{sample['Pred']}'")

# --- Execution Example ---
if __name__ == "__main__":
    # Mock data execution
    # analyzer = NERErrorAnalyzer(train_entities=["breast cancer", "APC mutation"])
    # analyzer.analyze_predictions(test_tokens, test_true_tags, test_pred_tags)
    # analyzer.print_report()
    pass
