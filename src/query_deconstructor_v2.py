"""
Query Deconstructor V2: Breaks queries into distinct search facets for balanced retrieval.
Uses simple, focused prompts to extract technical, behavioral, and constraint-based facets.
"""

import google.generativeai as genai
import json
import re
import time
from typing import List, Dict
import os


class QueryDeconstructorV2:
    """
    Deconstructs recruiter queries into separate search queries for multi-vector retrieval.
    Each facet gets its own semantic search to ensure balanced recommendations.
    """

    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")

    def deconstruct(self, query: str, max_retries: int = 3) -> List[str]:
        """
        Deconstruct query into multiple focused search queries with retry logic.

        Args:
            query: Original recruiter query
            max_retries: Maximum retry attempts for rate limits

        Returns:
            List of search query strings (one per skill domain/facet)
        """
        prompt = self._build_deconstruction_prompt(query)

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                search_queries = self._parse_response(response.text)

                # Ensure at least one search query
                if not search_queries:
                    search_queries = [query]

                return search_queries

            except Exception as e:
                error_str = str(e)
                if (
                    "429" in error_str
                    or "ResourceExhausted" in error_str
                    or "rate" in error_str.lower()
                ) and attempt < max_retries - 1:
                    # Extract retry delay from error message
                    import re

                    retry_match = re.search(r"retry in (\d+\.?\d*)", error_str)
                    if retry_match:
                        delay = float(retry_match.group(1)) + 1
                    else:
                        delay = 2**attempt  # Exponential backoff

                    print(
                        f"  Rate limit in deconstruction, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    print(f"Warning: Query deconstruction failed: {e}")
                    # Fallback: return original query
                    return [query]
            return [query]

    def _build_deconstruction_prompt(self, query: str) -> str:
        """Build prompt for query deconstruction."""
        return f"""You are an expert HR analyst. Analyze the following recruiter query and identify the distinct skill domains. Respond ONLY with a JSON list of search queries, one for each domain.

**Instructions:**
- Break the query into separate search strings for: technical skills, behavioral/soft skills, job level/experience, and any other distinct requirements
- Each search query should be concise and focused on ONE aspect
- For technical roles, separate technical skills from behavioral skills
- Keep each search query under 10 words
- Output ONLY the JSON array, no other text

**Examples:**

Query: 'I am hiring for Java developers who can also collaborate effectively.'
Response: ["Java technical skills programming", "collaboration interpersonal skills"]

Query: 'I am hiring for an analyst and want cognitive and personality tests.'
Response: ["analyst cognitive ability", "analyst personality assessment"]

Query: 'Find me 1 hour long assessment for a QA Engineer'
Response: ["QA Engineer testing quality assurance", "QA attention to detail problem solving"]

Query: 'Need a Senior Data Analyst with SQL and Python skills'
Response: ["Senior Data Analyst SQL Python", "data analysis problem solving"]

Query: 'ICICI Bank Assistant Admin, Experience required 0-2 years, test should be 30-40 mins long'
Response: ["Bank Assistant Admin clerical", "entry-level 0-2 years experience"]

Query: 'Content Writer required, expert in English and SEO'
Response: ["Content Writer English writing SEO", "creativity communication skills"]

**Now analyze this query:**

Query: '{query}'
Response: """

    def _parse_response(self, response_text: str) -> List[str]:
        """Parse LLM response into list of search queries."""
        try:
            # Extract JSON array from response
            json_match = re.search(r"\[.*?\]", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                search_queries = json.loads(json_str)

                # Ensure it's a list of strings
                if isinstance(search_queries, list):
                    return [str(q).strip() for q in search_queries if q]

            return []

        except Exception as e:
            print(f"Warning: Failed to parse deconstruction response: {e}")
            print(f"Response was: {response_text[:200]}")
            return []
