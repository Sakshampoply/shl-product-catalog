"""
Evaluate Multi-Vector Retrieval V2 (with query deconstruction and LLM reranking).
"""

import os
import time
from src.multi_vector_retriever_v2 import MultiVectorRetrieverV2
from src.evaluator import Evaluator


def main():
    # Load API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Initialize retriever
    print("=" * 80)
    print("MULTI-VECTOR RETRIEVAL V2 - EVALUATION")
    print("=" * 80)
    print("\nApproach: Query Deconstruction → Multi-Vector Search → LLM Reranking")
    print("This ensures balanced recommendations (technical + behavioral tests)\n")

    retriever = MultiVectorRetrieverV2(
        api_key=api_key, data_dir="data", gemini_dir="data/gemini"
    )

    # Initialize evaluator with training data
    train_file = "/Users/sakshampoply/Downloads/Gen_AI Dataset/Train-Set-Table 1.csv"
    evaluator = Evaluator(train_file)

    # Store predictions
    all_predictions = []

    print("\n" + "=" * 80)
    print("RUNNING EVALUATION ON TRAINING SET")
    print("=" * 80)

    # Process each query
    for query_idx, (query, ground_truth_urls) in enumerate(
        evaluator.ground_truth.items(), 1
    ):
        print(f"\n[Query {query_idx}/{len(evaluator.ground_truth)}]")
        print(f"Ground Truth: {len(ground_truth_urls)} assessments")

        # Retrieve with multi-vector approach
        results = retriever.retrieve(
            query, top_k=10, candidates_per_facet=10, use_deconstruction=True
        )

        # Extract URLs
        predicted_urls = [r["url"] for r in results]

        # Evaluate
        recall = evaluator.recall_at_k(predicted_urls, ground_truth_urls, k=10)
        print(f"✓ Recall@10: {recall:.2%}")

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

        # Add delay between queries to avoid rate limits
        if query_idx < len(evaluator.ground_truth):
            print("  Waiting 2s before next query...")
            time.sleep(2)

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
    print("COMPARISON TO PREVIOUS APPROACHES:")
    print("-" * 60)
    print(f"Baseline (sentence-transformers):    21.33%")
    print(f"Gemini embeddings only:               25.89%")
    print(f"Current best (Gemini + expansion):    28.67%")
    print(f"Multi-Vector V2 (this run):           {mean_recall*100:.2f}%")
    print("=" * 80)

    # Save predictions
    import pandas as pd

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv("multi_vector_v2_predictions.csv", index=False)
    print("\n✓ Predictions saved to: multi_vector_v2_predictions.csv")


if __name__ == "__main__":
    main()
