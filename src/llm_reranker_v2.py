"""
LLM Reranker V2: Reranks merged candidates to ensure balanced recommendations.
"""

import google.generativeai as genai
from typing import List, Dict, Any
import re
import time
import os


class LLMRerankerV2:
    """
    Reranks candidate assessments using Gemini LLM to ensure:
    1. Balanced mix of technical and behavioral tests
    2. Relevance to all aspects of the job query
    3. Proper consideration of constraints (duration, job level)
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash-lite"):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        max_retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using LLM to ensure balanced, relevant recommendations.

        Args:
            query: Original job query
            candidates: List of candidate assessments (with metadata)
            top_k: Number of top results to return
            max_retries: Maximum retry attempts for rate limits

        Returns:
            Reranked list of top-K candidates
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            return candidates

        for attempt in range(max_retries):
            try:
                # Build reranking prompt
                prompt = self._build_reranking_prompt(query, candidates, top_k)

                # Get LLM reranking
                response = self.model.generate_content(prompt)

                # Parse ranked indices
                ranked_indices = self._parse_reranking_response(
                    response.text, len(candidates)
                )

                # Reorder candidates
                reranked = []
                for idx in ranked_indices[:top_k]:
                    if 0 <= idx < len(candidates):
                        reranked.append(candidates[idx])

                # Fill remaining slots if needed
                if len(reranked) < top_k:
                    for i, candidate in enumerate(candidates):
                        if i not in ranked_indices[:top_k] and len(reranked) < top_k:
                            reranked.append(candidate)

                return reranked[:top_k]

            except Exception as e:
                error_str = str(e)
                if (
                    "429" in error_str
                    or "ResourceExhausted" in error_str
                    or "rate" in error_str.lower()
                ) and attempt < max_retries - 1:
                    # Extract retry delay
                    retry_match = re.search(r"retry in (\d+\.?\d*)", error_str)
                    if retry_match:
                        delay = float(retry_match.group(1)) + 1
                    else:
                        delay = 2**attempt

                    print(
                        f"  Rate limit in reranking, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"Warning: LLM reranking failed ({e}), returning original order"
                    )
                    return candidates[:top_k]

        # If all retries failed
        return candidates[:top_k]

    def _build_reranking_prompt(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int
    ) -> str:
        """Build prompt for LLM reranking."""
        # Build candidate list with metadata
        candidate_descriptions = []
        for i, candidate in enumerate(candidates):
            name = candidate.get("name", "Unknown")
            desc = candidate.get("description", "")[:200]  # Truncate long descriptions
            test_types = ", ".join(candidate.get("test_types", []))
            job_levels = ", ".join(candidate.get("job_levels", []))
            duration = candidate.get("duration_minutes", "N/A")

            candidate_descriptions.append(
                f"{i}. {name}\n"
                f"   Test Types: {test_types}\n"
                f"   Job Levels: {job_levels}\n"
                f"   Duration: {duration} min\n"
                f"   Description: {desc}"
            )

        candidates_text = "\n\n".join(candidate_descriptions)

        return f"""You are an expert HR assessment consultant. Your task is to select the {top_k} MOST RELEVANT assessments for the following job query, ensuring a BALANCED mix of test types.

**Job Query:**
{query}

**CRITICAL INSTRUCTIONS:**
1. If the query mentions BOTH technical/role skills AND soft/behavioral skills, you MUST include BOTH types of tests
2. For example: "Java developer who collaborates" needs BOTH Java technical tests AND collaboration/interpersonal tests
3. Prioritize tests that directly match the query requirements
4. Consider job level and duration constraints if mentioned
5. Ensure diversity - don't select 10 similar tests

**Candidate Assessments:**

{candidates_text}

**Your Task:**
Rank these assessments by relevance. Output ONLY the indices (numbers) of the top {top_k} assessments in order of relevance, separated by commas.

Example output format: 5, 12, 3, 18, 7, 1, 14, 9, 22, 16

Output (comma-separated indices only):"""

    def _parse_reranking_response(
        self, response_text: str, num_candidates: int
    ) -> List[int]:
        """Parse LLM response to extract ranked indices."""
        try:
            # Extract numbers from response
            numbers = re.findall(r"\b(\d+)\b", response_text)
            indices = []

            for num_str in numbers:
                idx = int(num_str)
                if 0 <= idx < num_candidates and idx not in indices:
                    indices.append(idx)

            return indices

        except Exception as e:
            print(f"Warning: Failed to parse reranking response: {e}")
            return list(range(num_candidates))
