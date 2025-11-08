"""
Hybrid Retrieval System combining FAISS semantic search and BM25 keyword search.
Supports both Sentence Transformers and Gemini embeddings.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dataclasses import asdict
from dotenv import load_dotenv

from .query_processor import QueryProcessor, QueryFeatures
from .query_expander import QueryExpander

# Try to import Gemini
try:
    import google.generativeai as genai

    load_dotenv()
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False


class HybridRetriever:
    """Hybrid retrieval combining FAISS semantic search and BM25 keyword search."""

    # Test type descriptions for better matching
    TEST_TYPE_DESCRIPTIONS = {
        "K": "Knowledge Skills Technical Coding Programming Development",
        "P": "Personality Behavior Behavioral Soft Skills Leadership Communication",
        "S": "Simulation Practical Hands-on Scenario Based Assessment",
        "A": "Ability Aptitude General Abilities",
        "B": "Behavior Behavioral Work Style Workplace",
        "C": "Cognitive Reasoning Logical Analytical Problem Solving",
        "D": "Development Growth Learning Career Development",
        "E": "Evaluation Assessment Center Development Center",
    }

    def __init__(
        self,
        data_dir: str = "data",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gemini: bool = True,  # Use Gemini by default if available
        use_query_expansion: bool = True,  # Use query expansion by default
    ):
        """
        Initialize the hybrid retriever.

        Args:
            data_dir: Directory containing index files
            model_name: Name of the sentence transformer model (fallback)
            use_gemini: Whether to use Gemini embeddings if available
            use_query_expansion: Whether to expand queries with Gemini 2.5 Flash
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        self.use_query_expansion = use_query_expansion and GEMINI_AVAILABLE

        # Load data
        self.records = self._load_records()
        self.metadata = self._load_metadata()

        # Try to load Gemini index first, fall back to sentence transformers
        if self.use_gemini:
            try:
                self.faiss_index = self._load_faiss_index(gemini=True)
                self.embeddings = self._load_embeddings(gemini=True)
                self.model = None  # Don't need local model with Gemini
                print("✓ Using Gemini embeddings (gemini-embedding-001)")
            except FileNotFoundError:
                print("⚠ Gemini index not found, falling back to sentence transformers")
                self.use_gemini = False
                self.faiss_index = self._load_faiss_index()
                self.embeddings = self._load_embeddings()
                self.model = SentenceTransformer(model_name)
        else:
            # Load FAISS index
            self.faiss_index = self._load_faiss_index()
            # Load embeddings for reference
            self.embeddings = self._load_embeddings()
            # Initialize sentence transformer
            self.model = SentenceTransformer(model_name)

        # Initialize BM25
        self.bm25, self.bm25_corpus = self._load_bm25()

        # Initialize query processor
        self.query_processor = QueryProcessor()

        # Initialize query expander
        if self.use_query_expansion:
            self.query_expander = QueryExpander()
            print("✓ Using query expansion with Gemini 2.5 Flash")
        else:
            self.query_expander = None

    def _load_records(self) -> List[Dict[str, Any]]:
        """Load catalog records."""
        records_path = self.data_dir / "catalog_records.json"
        with open(records_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata (optional - only needed for non-Gemini embeddings)."""
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Return minimal metadata if file doesn't exist
            return {
                "model": "gemini-embedding-001",
                "dimension": 3072,
                "num_records": len(self.records) if hasattr(self, "records") else 353,
            }

    def _load_faiss_index(self, gemini: bool = False):
        """Load FAISS index."""
        if gemini:
            index_path = self.data_dir / "gemini" / "faiss_gemini.index"
        else:
            index_path = self.data_dir / "faiss.index"
        return faiss.read_index(str(index_path))

    def _load_embeddings(self, gemini: bool = False) -> np.ndarray:
        """Load embeddings."""
        if gemini:
            embeddings_path = self.data_dir / "gemini" / "embeddings_gemini.npy"
        else:
            embeddings_path = self.data_dir / "embeddings.npy"
        return np.load(embeddings_path)

    def _load_bm25(self) -> Tuple[BM25Okapi, List[List[str]]]:
        """Load BM25 index and corpus."""
        bm25_path = self.data_dir / "bm25_corpus.json"
        with open(bm25_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        bm25 = BM25Okapi(corpus)
        return bm25, corpus

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        faiss_weight: float = 0.3,
        bm25_weight: float = 0.2,
        metadata_weight: float = 0.5,
        enforce_constraints: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k assessments using hybrid search.

        Args:
            query: User query
            top_k: Number of results to return
            faiss_weight: Weight for FAISS semantic similarity
            bm25_weight: Weight for BM25 keyword matching
            metadata_weight: Weight for metadata matching
            enforce_constraints: If True, hard filter by job level and duration

        Returns:
            List of assessment records with scores
        """
        # Expand query if enabled
        expanded_query = query
        if self.use_query_expansion and self.query_expander:
            expanded_query = self.query_expander.expand_query(query)
            # Use expanded query for semantic and keyword search

        # Process query to extract features (use original for metadata extraction)
        query_features = self.query_processor.process(query)

        # Get candidates from FAISS (retrieve MORE for filtering)
        # Use expanded query for better semantic matching
        faiss_candidates = self._faiss_search(expanded_query, top_k=200)

        # Get candidates from BM25 with query features for boosting
        # Use expanded query for better keyword matching
        bm25_candidates = self._bm25_search(
            expanded_query, top_k=200, query_features=query_features
        )

        # Combine candidates
        all_candidates = {}

        # Add FAISS candidates
        for idx, score in faiss_candidates:
            all_candidates[idx] = {
                "faiss_score": score,
                "bm25_score": 0.0,
                "metadata_score": 0.0,
            }

        # Add BM25 candidates
        for idx, score in bm25_candidates:
            if idx in all_candidates:
                all_candidates[idx]["bm25_score"] = score
            else:
                all_candidates[idx] = {
                    "faiss_score": 0.0,
                    "bm25_score": score,
                    "metadata_score": 0.0,
                }

        # Apply hard constraint filtering if enabled
        if enforce_constraints:
            all_candidates = self._filter_by_constraints(all_candidates, query_features)

        # Calculate metadata scores for all candidates
        for idx in all_candidates:
            all_candidates[idx]["metadata_score"] = self._calculate_metadata_score(
                idx, query_features
            )

        # Normalize scores
        all_candidates = self._normalize_scores(all_candidates)

        # Calculate final weighted scores
        final_scores = []
        for idx, scores in all_candidates.items():
            combined_score = (
                faiss_weight * scores["faiss_score"]
                + bm25_weight * scores["bm25_score"]
                + metadata_weight * scores["metadata_score"]
            )
            final_scores.append((idx, combined_score, scores))

        # Sort by final score
        final_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply diversity and balance
        final_results = self._apply_diversity_balance(
            final_scores, query_features, top_k
        )

        # Prepare results
        results = []
        for idx, score, score_breakdown in final_results:
            record = self.records[idx].copy()
            record["retrieval_score"] = float(score)
            record["score_breakdown"] = {
                k: float(v) for k, v in score_breakdown.items()
            }
            results.append(record)

        return results

    def _faiss_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Perform FAISS semantic search."""
        # Encode query
        if self.use_gemini:
            # Use Gemini embedding API
            result = genai.embed_content(
                model="gemini-embedding-001",
                content=query,
                task_type="retrieval_query",
            )
            query_embedding = np.array([result["embedding"]], dtype="float32")
        else:
            # Use sentence transformers
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype("float32")

        # Search
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        # Convert distances to similarity scores
        if self.use_gemini:
            # Gemini uses cosine similarity (IndexFlatIP), scores are already similarities
            similarities = distances[0]
        else:
            # Sentence transformers use L2 distance, convert to similarity
            similarities = 1 / (1 + distances[0])

        results = [
            (int(idx), float(sim))
            for idx, sim in zip(indices[0], similarities)
            if idx != -1
        ]
        return results

    def _bm25_search(
        self,
        query: str,
        top_k: int = 50,
        query_features: Optional[QueryFeatures] = None,
    ) -> List[Tuple[int, float]]:
        """Perform BM25 keyword search with query expansion."""
        # Tokenize query (simple split)
        query_tokens = query.lower().split()

        # Add extracted features to boost relevant terms
        if query_features:
            # Add job level terms
            if query_features.job_level:
                level_tokens = query_features.job_level.lower().split()
                query_tokens.extend(level_tokens)

            # Add role if extracted
            if query_features.role:
                role_tokens = query_features.role.lower().split()
                query_tokens.extend(role_tokens)

            # Add skill keywords (weight them by repetition)
            for skill in query_features.technical_skills:
                query_tokens.append(skill)
                query_tokens.append(skill)  # Double weight

            for skill in query_features.behavioral_skills:
                query_tokens.append(skill)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0
        ]
        return results

    def _filter_by_constraints(
        self, candidates: Dict[int, Dict[str, float]], query_features: QueryFeatures
    ) -> Dict[int, Dict[str, float]]:
        """
        Hard filter candidates by job level and duration constraints.
        Only keep assessments that match the required constraints.
        """
        filtered = {}

        for idx, scores in candidates.items():
            record = self.records[idx]
            keep = True

            # Hard filter by duration if specified
            if query_features.duration_max is not None:
                assessment_duration = record.get("assessment_length_minutes")
                # If duration is unknown (None), keep it (lenient on missing data)
                # If duration exceeds max by >20%, reject
                if assessment_duration is not None:
                    if assessment_duration > query_features.duration_max * 1.2:
                        keep = False

            # Hard filter by job level if specified
            if query_features.job_level and keep:
                job_levels = record.get("job_levels", [])
                # Keep if matches job level OR "General Population"
                if job_levels:
                    if (
                        query_features.job_level not in job_levels
                        and "General Population" not in job_levels
                    ):
                        keep = False

            if keep:
                filtered[idx] = scores

        return filtered

    def _calculate_metadata_score(
        self, idx: int, query_features: QueryFeatures
    ) -> float:
        """Calculate metadata matching score with stronger penalties for mismatches."""
        record = self.records[idx]
        score = 0.0
        weights_sum = 0.0

        # Duration matching (weight: 0.35 - INCREASED)
        if query_features.duration_max is not None:
            duration_weight = 0.35
            weights_sum += duration_weight

            assessment_duration = record.get("assessment_length_minutes")
            if assessment_duration is not None:
                if assessment_duration <= query_features.duration_max:
                    # Perfect match if within limit
                    score += duration_weight
                    # Bonus for closer matches
                    if query_features.duration_min is not None:
                        if query_features.duration_min <= assessment_duration:
                            score += duration_weight * 0.5
                    else:
                        # Prefer assessments closer to the max
                        ratio = assessment_duration / query_features.duration_max
                        score += duration_weight * 0.4 * ratio
                else:
                    # STRONGER penalty for exceeding duration
                    overage_ratio = query_features.duration_max / assessment_duration
                    score += duration_weight * 0.1 * overage_ratio
            else:
                # Missing duration gets moderate score
                score += duration_weight * 0.5

        # Job level matching (weight: 0.35 - INCREASED)
        if query_features.job_level:
            level_weight = 0.35
            weights_sum += level_weight

            job_levels = record.get("job_levels", [])

            # Treat Graduate and Entry-Level as equivalent
            query_level = query_features.job_level
            if query_level == "Graduate":
                query_level = "Entry-Level"
            elif query_level == "Entry-Level":
                # Also match Graduate assessments
                if job_levels and "Graduate" in job_levels:
                    score += level_weight
                    weights_sum = weights_sum  # Already added above

            if job_levels and query_level in job_levels:
                # Perfect match
                score += level_weight
            elif job_levels and any(
                level in job_levels for level in ["General Population"]
            ):
                # General Population gets 70% credit
                score += level_weight * 0.7
            else:
                # No match gets 0
                score += 0

        # Test type matching (weight: 0.30 - DECREASED)
        if query_features.test_types_needed:
            type_weight = 0.30
            weights_sum += type_weight

            test_types = set(record.get("test_type_keys", []))
            needed_types = query_features.test_types_needed

            if test_types and needed_types:
                # Calculate overlap ratio
                overlap = test_types.intersection(needed_types)
                if overlap:
                    overlap_ratio = len(overlap) / len(needed_types)
                    score += type_weight * overlap_ratio

        # Normalize by sum of applicable weights
        if weights_sum > 0:
            score = score / weights_sum

        return score

    def _normalize_scores(
        self, candidates: Dict[int, Dict[str, float]]
    ) -> Dict[int, Dict[str, float]]:
        """Normalize scores to [0, 1] range."""
        # Get min/max for each score type
        if not candidates:
            return candidates

        for score_type in ["faiss_score", "bm25_score", "metadata_score"]:
            scores = [c[score_type] for c in candidates.values()]
            min_score = min(scores)
            max_score = max(scores)

            if max_score > min_score:
                for idx in candidates:
                    candidates[idx][score_type] = (
                        candidates[idx][score_type] - min_score
                    ) / (max_score - min_score)
            else:
                # All scores are the same
                for idx in candidates:
                    candidates[idx][score_type] = 1.0 if max_score > 0 else 0.0

        return candidates

    def _apply_diversity_balance(
        self,
        scored_results: List[Tuple[int, float, Dict]],
        query_features: QueryFeatures,
        top_k: int,
    ) -> List[Tuple[int, float, Dict]]:
        """
        Apply diversity and balance to results.
        Ensures a balanced mix of test types when multiple types are needed.
        """
        if len(scored_results) <= top_k:
            return scored_results[:top_k]

        # If multiple test types are needed, ensure balance
        if len(query_features.test_types_needed) > 1:
            selected = []
            type_counts = {t: 0 for t in query_features.test_types_needed}
            max_per_type = max(2, top_k // len(query_features.test_types_needed))

            # First pass: select top items ensuring balance
            for idx, score, score_breakdown in scored_results:
                record = self.records[idx]
                test_types = set(record.get("test_type_keys", []))

                # Check if this assessment matches any needed type that hasn't reached limit
                matching_types = test_types.intersection(
                    query_features.test_types_needed
                )

                can_add = False
                for t in matching_types:
                    if type_counts[t] < max_per_type:
                        can_add = True
                        break

                if can_add or len(selected) < top_k // 2:  # Allow some flexibility
                    selected.append((idx, score, score_breakdown))
                    for t in matching_types:
                        type_counts[t] += 1

                    if len(selected) >= top_k:
                        break

            # Second pass: fill remaining slots with top scores
            if len(selected) < top_k:
                for idx, score, score_breakdown in scored_results:
                    if (idx, score, score_breakdown) not in selected:
                        selected.append((idx, score, score_breakdown))
                        if len(selected) >= top_k:
                            break

            return selected[:top_k]
        else:
            # No balance needed, just return top-k
            return scored_results[:top_k]
