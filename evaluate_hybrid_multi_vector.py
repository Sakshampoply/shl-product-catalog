"""
Evaluate Hybrid Multi-Vector approach (no runtime embedding API calls).
"""

from src.hybrid_multi_vector_retriever import HybridMultiVectorRetriever
from src.evaluator import Evaluator


def main():
    print("=" * 80)
    print("HYBRID MULTI-VECTOR EVALUATION")
    print("=" * 80)
    print("\nApproach: Query Expansion + Balanced Scoring")
    print("No runtime API calls - uses pre-computed Gemini embeddings\n")

    # Initialize retriever
    retriever = HybridMultiVectorRetriever(data_dir="data", gemini_dir="data/gemini")

    # Initialize evaluator
    train_file = "/Users/sakshampoply/Downloads/Gen_AI Dataset/Train-Set-Table 1.csv"
    evaluator = Evaluator(train_file)

    # Store predictions
    all_predictions = []

    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    # Process each query
    for query_idx, (query, ground_truth_urls) in enumerate(
        evaluator.ground_truth.items(), 1
    ):
        print(f"\n[Query {query_idx}/{len(evaluator.ground_truth)}]")
        print(f"Query: {query[:80]}...")

        # Retrieve
        results = retriever.retrieve(query, top_k=10, pool_size=50)

        # Extract URLs
        predicted_urls = [r["url"] for r in results]

        # Evaluate
        recall = evaluator.recall_at_k(predicted_urls, ground_truth_urls, k=10)
        print(
            f"Recall@10: {recall:.2%} ({int(recall * len(ground_truth_urls))}/{len(ground_truth_urls)} matches)"
        )

        # Store prediction
        all_predictions.append(
            {
                "Query Number": query_idx,
                "Query": query,
                "Predicted URLs": "; ".join(predicted_urls),
                "Ground Truth Count": len(ground_truth_urls),
                "Recall@10": recall,
            }
        )

    # Calculate mean recall
    mean_recall = sum(p["Recall@10"] for p in all_predictions) / len(all_predictions)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nMean Recall@10: {mean_recall:.4f} ({mean_recall*100:.2f}%)")

    # Per-query summary
    print("\nPer-Query Results:")
    print("-" * 60)
    for pred in all_predictions:
        print(f"Query {pred['Query Number']}: {pred['Recall@10']:.2%}")
    print("-" * 60)
    print(f"Mean Recall@10: {mean_recall:.2%}")

    # Compare to baselines
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("-" * 60)
    print(f"Baseline (sentence-transformers):           21.33%")
    print(f"Gemini embeddings only:                     25.89%")
    print(f"Gemini + basic query expansion:             28.67%")
    print(f"Hybrid Multi-Vector (balanced scoring):     {mean_recall*100:.2f}%")
    print("=" * 80)

    # Save predictions
    import pandas as pd

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv("hybrid_multi_vector_predictions.csv", index=False)
    print("\n✓ Predictions saved to: hybrid_multi_vector_predictions.csv")


if __name__ == "__main__":
    main()
