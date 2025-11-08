"""
Multi-Vector Retriever V2: Uses query deconstruction + semantic search + LLM reranking.
This version uses pre-computed Gemini embeddings to avoid runtime API quota issues.
"""

import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any
import json
import os
import time
from pathlib import Path

from src.query_deconstructor_v2 import QueryDeconstructorV2
from src.llm_reranker_v2 import LLMRerankerV2


def retry_with_backoff(func, max_retries=3, initial_delay=1.0):
    """
    Retry function with exponential backoff for rate limit errors.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds

    Returns:
        Function result
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            if (
                "429" in error_str
                or "ResourceExhausted" in error_str
                or "rate" in error_str.lower()
            ):
                if attempt < max_retries - 1:
                    # Extract retry delay from error message if available
                    import re

                    retry_match = re.search(r"retry in (\d+\.?\d*)", error_str)
                    if retry_match:
                        delay = float(retry_match.group(1)) + 1  # Add 1 second buffer
                    else:
                        delay = initial_delay * (2**attempt)  # Exponential backoff

                    print(
                        f"  Rate limit hit, waiting {delay:.1f}s before retry (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    raise
            else:
                raise
    return None


class MultiVectorRetrieverV2:
    """
    Multi-vector retrieval with balanced recommendations.

    Pipeline:
    1. Deconstruct query into facets (technical, behavioral, experience, etc.)
    2. Run separate semantic searches for each facet using Gemini embeddings
    3. Merge candidates from all searches (ensuring diversity)
    4. LLM re-ranks merged candidates for final balanced top-K
    """

    def __init__(
        self,
        api_key: str = None,
        data_dir: str = "data",
        gemini_dir: str = "data/gemini",
    ):
        """
        Initialize multi-vector retriever.

        Args:
            api_key: Google API key
            data_dir: Directory with catalog records
            gemini_dir: Directory with Gemini embeddings and FAISS index
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.gemini_dir = Path(gemini_dir)

        # Load catalog records
        with open(self.data_dir / "catalog_records.json", "r") as f:
            self.records = json.load(f)

        # Load Gemini FAISS index and embeddings
        self.faiss_index = faiss.read_index(str(self.gemini_dir / "faiss_gemini.index"))
        self.embeddings = np.load(self.gemini_dir / "embeddings_gemini.npy")

        # Initialize components
        self.deconstructor = QueryDeconstructorV2(api_key)
        self.reranker = LLMRerankerV2(api_key)

        print(f"✓ Loaded Multi-Vector Retriever V2")
        print(f"  - {len(self.records)} assessments")
        print(f"  - Gemini embeddings (3072-dim)")
        print(f"  - Query deconstruction + LLM reranking enabled")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidates_per_facet: int = 10,
        use_deconstruction: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve assessments using multi-vector approach.

        Args:
            query: Original recruiter query
            top_k: Final number of results to return
            candidates_per_facet: Number of candidates per facet search
            use_deconstruction: If False, uses single expanded query

        Returns:
            List of top-K assessment records after LLM reranking
        """
        print(f"\n{'='*80}")
        print(f"Query: {query[:100]}...")
        print(f"{'='*80}")

        # Step 1: Deconstruct query into facets (or use single query)
        if use_deconstruction:
            print("\n[Step 1] Deconstructing query into search facets...")
            search_queries = self.deconstructor.deconstruct(query)
            print(f"  Generated {len(search_queries)} search queries:")
            for i, sq in enumerate(search_queries, 1):
                print(f"    {i}. {sq}")
        else:
            print("\n[Step 1] Using single query (deconstruction disabled)")
            search_queries = [query]

        # Step 2: Multi-vector semantic search
        print(f"\n[Step 2] Running semantic searches for each facet...")
        all_candidates = []
        seen_urls = set()

        for i, search_query in enumerate(search_queries, 1):
            print(f"  Search {i}/{len(search_queries)}: '{search_query}'")

            # Add delay between searches to avoid rate limits (except first search)
            if i > 1:
                time.sleep(0.5)  # 500ms delay between searches

            candidates = self._semantic_search(search_query, candidates_per_facet)

            # Track which facet matched each candidate
            for candidate in candidates:
                url = candidate["url"]
                if url not in seen_urls:
                    candidate["matched_facet"] = i
                    candidate["matched_query"] = search_query
                    all_candidates.append(candidate)
                    seen_urls.add(url)

        print(
            f"\n  Merged {len(all_candidates)} unique candidates from {len(search_queries)} facets"
        )

        # Show distribution
        facet_counts = {}
        for c in all_candidates:
            facet = c.get("matched_facet", 0)
            facet_counts[facet] = facet_counts.get(facet, 0) + 1
        print(f"  Distribution: {facet_counts}")

        # Step 3: LLM Re-ranking for balanced selection
        print(
            f"\n[Step 3] LLM re-ranking {len(all_candidates)} candidates to select top {top_k}..."
        )
        reranked = self.reranker.rerank(query, all_candidates, top_k)

        # Show final distribution
        final_facet_counts = {}
        for c in reranked:
            facet = c.get("matched_facet", 0)
            final_facet_counts[facet] = final_facet_counts.get(facet, 0) + 1
        print(f"  Final top-{top_k} distribution: {final_facet_counts}")

        print(f"\n{'='*80}\n")

        return reranked

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform semantic search using Gemini embeddings with retry logic.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of assessment records
        """

        def _get_embedding():
            return genai.embed_content(
                model="gemini-embedding-001",
                content=query,
                task_type="retrieval_query",
            )

        # Get query embedding with retry logic
        result = retry_with_backoff(_get_embedding, max_retries=3, initial_delay=1.0)
        query_embedding = np.array([result["embedding"]], dtype="float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        # Build results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.records):
                record = self.records[idx].copy()
                record["score"] = float(distance)
                record["rank"] = i + 1
                results.append(record)

        return results
