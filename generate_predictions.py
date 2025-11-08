"""
Script to generate predictions on the test dataset using Multi-Vector V2.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.multi_vector_retriever_v2 import MultiVectorRetrieverV2


def main():
    """Generate predictions for test dataset."""
    print("=" * 80)
    print("SHL ASSESSMENT RECOMMENDATION SYSTEM - TEST PREDICTIONS")
    print("Multi-Vector V2 with Query Deconstruction + LLM Reranking")
    print("=" * 80)
    print()

    # Initialize retriever
    print("Loading Multi-Vector Retriever V2...")
    retriever = MultiVectorRetrieverV2(data_dir="data", gemini_dir="data/gemini")
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
            # Use multi-vector retrieval
            results = retriever.retrieve(query, top_k=10)

            # Add predictions for this query
            for result in results:
                all_predictions.append(
                    {"Query": query, "Assessment_url": result["url"]}
                )

            print(f"  ✓ Generated {len(results)} recommendations")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()

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
