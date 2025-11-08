"""
Script to generate predictions on the test dataset.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.hybrid_retriever import HybridRetriever


def main():
    """Generate predictions for test dataset."""
    print("=" * 80)
    print("SHL ASSESSMENT RECOMMENDATION SYSTEM - TEST PREDICTIONS")
    print("=" * 80)
    print()

    # Initialize retriever
    print("Loading retriever...")
    retriever = HybridRetriever(data_dir="data")
    print("✓ Retriever loaded")
    print()

    # Load test dataset
    test_file = "/Users/sakshampoply/Downloads/Gen_AI Dataset/Test-Set-Table 1.csv"
    print(f"Loading test data from: {test_file}")
    df = pd.read_csv(test_file)

    # Get unique queries
    queries = df["Query"].dropna().unique().tolist()
    print(f"✓ Found {len(queries)} unique queries")
    print()

    # Generate predictions
    print("Generating predictions...")
    all_predictions = []

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Processing: {query[:80]}...")

        try:
            # Use semantic-focused configuration for best performance
            results = retriever.retrieve(
                query,
                top_k=10,
                faiss_weight=0.5,
                bm25_weight=0.3,
                metadata_weight=0.2,
                enforce_constraints=False,
            )

            # Add predictions for this query
            for result in results:
                all_predictions.append(
                    {"Query": query, "Assessment_url": result["url"]}
                )

            print(f"  ✓ Generated {len(results)} recommendations")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print()
    print("=" * 80)

    # Save predictions to CSV
    output_file = "test_predictions.csv"
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(output_file, index=False)

    print(f"✓ Test predictions saved to: {output_file}")
    print(f"  Total rows: {len(predictions_df)}")
    print(f"  Queries: {len(queries)}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
