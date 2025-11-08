"""
Evaluation module for computing Mean Recall@K and other metrics.
"""

import pandas as pd
from typing import List, Dict
import re


def normalize_url(url: str) -> str:
    """Normalize URL by extracting assessment identifier."""
    if not url:
        return ""
    match = re.search(r"product-catalog/view/([^/]+)", url)
    if match:
        return match.group(1).lower().strip("/")
    return url.lower()


class Evaluator:
    """Evaluator for recommendation system."""

    def __init__(self, train_file: str = None):
        """Initialize evaluator."""
        self.train_file = train_file
        self.ground_truth = None
        if train_file:
            self.load_ground_truth(train_file)

    def load_ground_truth(self, train_file: str):
        """Load ground truth from training file."""
        df = pd.read_csv(train_file)
        self.ground_truth = {}
        for _, row in df.iterrows():
            query = row["Query"]
            url = row["Assessment_url"]
            if pd.notna(query) and pd.notna(url):
                normalized_url = normalize_url(url)
                if query not in self.ground_truth:
                    self.ground_truth[query] = []
                if normalized_url not in self.ground_truth[query]:
                    self.ground_truth[query].append(normalized_url)

    def recall_at_k(
        self, predictions: List[str], ground_truth: List[str], k: int = 10
    ) -> float:
        """Calculate Recall@K for a single query."""
        if not ground_truth:
            return 0.0
        top_k_predictions = [normalize_url(url) for url in predictions[:k]]
        matches = len(set(top_k_predictions).intersection(set(ground_truth)))
        return matches / len(ground_truth)

    def mean_recall_at_k(
        self, all_predictions: Dict[str, List[str]], k: int = 10
    ) -> float:
        """Calculate Mean Recall@K across all queries."""
        if not self.ground_truth:
            raise ValueError("Ground truth not loaded")
        recalls = []
        for query, predictions in all_predictions.items():
            if query in self.ground_truth:
                recall = self.recall_at_k(predictions, self.ground_truth[query], k)
                recalls.append(recall)
        return sum(recalls) / len(recalls) if recalls else 0.0
