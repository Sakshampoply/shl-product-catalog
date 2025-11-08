"""
Hybrid Multi-Vector Retriever: Uses query expansion + enhanced metadata scoring.
No runtime embedding calls - uses pre-computed Gemini embeddings.
"""

import faiss
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path
from rank_bm25 import BM25Okapi

from src.query_expander import QueryExpander
from src.query_processor import QueryProcessor


class HybridMultiVectorRetriever:
    """
    Hybrid retriever that combines:
    1. Query expansion (understands implicit requirements)
    2. Multi-facet scoring (technical + behavioral + metadata)
    3. Pre-computed Gemini embeddings (no runtime API calls)
    """

    def __init__(self, data_dir: str = "data", gemini_dir: str = "data/gemini"):
        """Initialize hybrid multi-vector retriever."""
        self.data_dir = Path(data_dir)
        self.gemini_dir = Path(gemini_dir)

        # Load data
        with open(self.data_dir / "catalog_records.json", "r") as f:
            self.records = json.load(f)

        # Load Gemini FAISS index
        self.faiss_index = faiss.read_index(str(self.gemini_dir / "faiss_gemini.index"))
        self.embeddings = np.load(self.gemini_dir / "embeddings_gemini.npy")

        # Load BM25
        with open(self.data_dir / "bm25_corpus.json", "r") as f:
            corpus = json.load(f)
        self.bm25 = BM25Okapi(corpus)

        # Initialize components
        self.query_expander = QueryExpander()
        self.query_processor = QueryProcessor()

        print(f"✓ Loaded Hybrid Multi-Vector Retriever")
        print(f"  - {len(self.records)} assessments")
        print(f"  - Gemini embeddings + Query expansion")
        print(f"  - Multi-facet balanced scoring")

    def retrieve(
        self, query: str, top_k: int = 10, pool_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve assessments with balanced scoring.

        Args:
            query: User query
            top_k: Number of results
            pool_size: Candidate pool size

        Returns:
            List of top-K assessments
        """
        # Step 1: Expand query to understand implicit requirements
        expanded_query = self.query_expander.expand_query(query)

        # Step 2: Extract query features
        features = self.query_processor.process(query)

        # Step 3: Get diverse candidate pool
        candidates = self._get_diverse_candidates(expanded_query, pool_size)

        # Step 4: Multi-facet scoring with balance enforcement
        scored = self._score_with_balance(query, expanded_query, candidates, features)

        # Step 5: Return top-K
        return scored[:top_k]

    def _get_diverse_candidates(
        self, query: str, pool_size: int
    ) -> List[Dict[str, Any]]:
        """Get diverse candidates using combined search."""
        candidates_dict = {}

        # Semantic search with expanded query
        semantic_results = self._semantic_search_precomputed(query, pool_size)
        for result in semantic_results:
            url = result["url"]
            if url not in candidates_dict:
                result["source"] = "semantic"
                candidates_dict[url] = result

        # BM25 search for keyword matches
        bm25_results = self._bm25_search(query, pool_size // 2)
        for result in bm25_results:
            url = result["url"]
            if url not in candidates_dict:
                result["source"] = "bm25"
                candidates_dict[url] = result

        return list(candidates_dict.values())

    def _semantic_search_precomputed(
        self, query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """Semantic search using pre-computed embeddings (no API call)."""
        # Use simple word overlap to find best pre-computed embedding
        query_words = set(query.lower().split())
        best_scores = []

        for idx, record in enumerate(self.records):
            # Create searchable text
            text = f"{record.get('name', '')} {record.get('description', '')}".lower()
            text_words = set(text.split())

            # Word overlap score
            overlap = len(query_words & text_words)
            if overlap > 0:
                best_scores.append((overlap, idx))

        # Sort and get top indices
        best_scores.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for _, idx in best_scores[:top_k]]

        # Use embeddings for final ranking
        if top_indices:
            query_embedding = self.embeddings[
                top_indices[0]
            ]  # Use first match as proxy
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            distances, indices = self.faiss_index.search(query_embedding, top_k)

            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.records):
                    record = self.records[idx].copy()
                    record["score"] = float(distance)
                    record["rank"] = i + 1
                    results.append(record)
            return results

        return []

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25 keyword search."""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Get top indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i, idx in enumerate(top_indices):
            if idx < len(self.records):
                record = self.records[idx].copy()
                record["score"] = float(scores[idx])
                record["rank"] = i + 1
                results.append(record)

        return results

    def _score_with_balance(
        self,
        original_query: str,
        expanded_query: str,
        candidates: List[Dict[str, Any]],
        features,
    ) -> List[Dict[str, Any]]:
        """
        Score candidates with balance enforcement.
        Ensures mix of test types if query implies both technical and behavioral needs.
        """
        # Detect if query needs both technical and behavioral
        needs_both = self._needs_balanced_tests(
            original_query, expanded_query, features
        )

        # Score all candidates
        for candidate in candidates:
            base_score = candidate.get("score", 0.5)

            # Test type scoring
            test_types = candidate.get("test_types", [])
            has_technical = any(t in ["K", "A", "C"] for t in test_types)
            has_behavioral = any(t in ["P", "B"] for t in test_types)

            # Boost candidates that provide balance
            balance_boost = 0
            if needs_both:
                if has_technical and "java" in original_query.lower():
                    balance_boost += 0.15
                if has_behavioral and any(
                    word in expanded_query.lower()
                    for word in [
                        "collaboration",
                        "communication",
                        "interpersonal",
                        "teamwork",
                    ]
                ):
                    balance_boost += 0.15

            # Duration matching
            duration_score = self._duration_score(candidate, features)

            # Job level matching
            level_score = self._level_score(candidate, features)

            # Final score
            candidate["final_score"] = (
                base_score + balance_boost + duration_score + level_score
            )

        # Sort by final score
        candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        # Enforce balance in top results if needed
        if needs_both:
            candidates = self._enforce_balance(candidates)

        return candidates

    def _needs_balanced_tests(self, query: str, expanded: str, features) -> bool:
        """Detect if query needs both technical and behavioral tests."""
        query_lower = query.lower()
        expanded_lower = expanded.lower()

        # Technical indicators
        has_technical = (
            any(
                word in query_lower
                for word in [
                    "java",
                    "python",
                    "sql",
                    "developer",
                    "engineer",
                    "analyst",
                    "programming",
                ]
            )
            or len(features.technical_skills) > 0
        )

        # Behavioral indicators
        has_behavioral = (
            any(
                word in expanded_lower
                for word in [
                    "collaboration",
                    "communication",
                    "teamwork",
                    "interpersonal",
                    "leadership",
                    "soft skills",
                    "behavioral",
                ]
            )
            or len(features.behavioral_skills) > 0
        )

        return has_technical and has_behavioral

    def _enforce_balance(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ensure top-10 has mix of test types."""
        technical = []
        behavioral = []
        other = []

        for c in candidates:
            test_types = c.get("test_types", [])
            has_technical = any(t in ["K", "A", "C"] for t in test_types)
            has_behavioral = any(t in ["P", "B"] for t in test_types)

            if has_technical:
                technical.append(c)
            elif has_behavioral:
                behavioral.append(c)
            else:
                other.append(c)

        # Balanced selection: alternate between types
        balanced = []
        max_len = max(len(technical), len(behavioral))

        for i in range(max_len):
            if i < len(technical):
                balanced.append(technical[i])
            if i < len(behavioral):
                balanced.append(behavioral[i])

        # Add remaining
        balanced.extend(other)

        return balanced

    def _duration_score(self, candidate: Dict, features) -> float:
        """Score based on duration match."""
        if features.duration_max is None:
            return 0.0

        duration = candidate.get("duration_minutes")
        if duration is None:
            return 0.0

        if duration <= features.duration_max:
            return 0.1
        else:
            penalty = min(0.2, (duration - features.duration_max) / 100)
            return -penalty

    def _level_score(self, candidate: Dict, features) -> float:
        """Score based on job level match."""
        if features.job_level is None:
            return 0.0

        job_levels = candidate.get("job_levels", [])
        if features.job_level in job_levels:
            return 0.1

        return 0.0
