"""
Script to evaluate the recommendation system using the training dataset.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.hybrid_retriever import HybridRetriever
from src.evaluator import Evaluator, normalize_url


def main():
    """Run evaluation on training dataset."""
    print("=" * 80)
    print("SHL ASSESSMENT RECOMMENDATION SYSTEM - EVALUATION")
    print("=" * 80)
    print()

    # Initialize retriever
    print("Loading retriever...")
    retriever = HybridRetriever(data_dir="data")
    print("✓ Retriever loaded")
    print()

    # Initialize evaluator
    train_file = "/Users/sakshampoply/Downloads/Gen_AI Dataset/Train-Set-Table 1.csv"
    print(f"Loading training data from: {train_file}")
    evaluator = Evaluator(train_file)
    print(f"✓ Loaded {len(evaluator.ground_truth)} unique queries")
    print()

    # Get predictions for all queries
    print("Generating predictions...")
    all_predictions = {}

    for i, (query, ground_truth_urls) in enumerate(evaluator.ground_truth.items(), 1):
        print(f"[{i}/{len(evaluator.ground_truth)}] Processing: {query[:80]}...")

        try:
            # Use semantic-focused weights: FAISS=0.5, BM25=0.3, metadata=0.2, no hard constraints
            results = retriever.retrieve(
                query,
                top_k=10,
                faiss_weight=0.5,
                bm25_weight=0.3,
                metadata_weight=0.2,
                enforce_constraints=False,
            )
            predicted_urls = [r["url"] for r in results]
            all_predictions[query] = predicted_urls

            # Calculate recall for this query
            recall = evaluator.recall_at_k(predicted_urls, ground_truth_urls, k=10)
            # Normalize predicted URLs for match counting
            normalized_predicted = [normalize_url(url) for url in predicted_urls]
            matches = len(
                set(normalized_predicted).intersection(set(ground_truth_urls))
            )
            print(
                f"  Recall@10: {recall:.4f} ({matches}/{len(ground_truth_urls)} matches)"
            )

        except Exception as e:
            print(f"  Error: {e}")
            all_predictions[query] = []

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Calculate Mean Recall@K
    mean_recall = evaluator.mean_recall_at_k(all_predictions, k=10)
    print(f"Mean Recall@10: {mean_recall:.4f}")
    print()

    # Detailed per-query results
    print("Per-Query Results:")
    print("-" * 80)
    for i, (query, ground_truth_urls) in enumerate(evaluator.ground_truth.items(), 1):
        if query in all_predictions:
            predicted_urls = all_predictions[query]
            recall = evaluator.recall_at_k(predicted_urls, ground_truth_urls, k=10)
            normalized_predicted = [normalize_url(url) for url in predicted_urls]
            matches = len(
                set(normalized_predicted).intersection(set(ground_truth_urls))
            )

            print(f"\nQuery {i}:")
            print(f"  {query[:100]}...")
            print(f"  Recall@10: {recall:.4f}")
            print(f"  Matches: {matches}/{len(ground_truth_urls)}")
            print(f"  Predicted: {len(predicted_urls)} URLs")

    print()
    print("=" * 80)

    # Save predictions to CSV for reference
    output_file = "evaluation_predictions.csv"
    rows = []
    for query, urls in all_predictions.items():
        for url in urls:
            rows.append({"Query": query, "Assessment_url": url})

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to: {output_file}")


if __name__ == "__main__":
    main()
