"""
Query Processor for extracting features from job queries.
Minimal version - extracts basic features from queries.
"""

from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class QueryFeatures:
    """Features extracted from a query."""

    technical_skills: List[str]
    behavioral_skills: List[str]
    job_level: Optional[str]
    min_duration_minutes: Optional[int]
    max_duration_minutes: Optional[int]
    test_type_preferences: List[str]
    raw_query: str
    # Additional fields used by hybrid_retriever
    role: Optional[str] = None
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    test_types_needed: List[str] = None

    def __post_init__(self):
        """Set convenience aliases after initialization."""
        # Map old field names to new ones for compatibility
        if self.duration_min is None and self.min_duration_minutes is not None:
            self.duration_min = self.min_duration_minutes
        if self.duration_max is None and self.max_duration_minutes is not None:
            self.duration_max = self.max_duration_minutes
        if self.test_types_needed is None:
            self.test_types_needed = (
                self.test_type_preferences if self.test_type_preferences else []
            )


class QueryProcessor:
    """Extract features from job queries."""

    def __init__(self):
        # Common job levels
        self.job_levels = {
            "graduate": "Graduate",
            "entry": "Entry-Level",
            "entry-level": "Entry-Level",
            "junior": "Entry-Level",
            "mid": "Professional",
            "professional": "Professional",
            "senior": "Professional",
            "manager": "Management",
            "lead": "Management",
            "director": "Management",
            "executive": "Management",
            "coo": "Management",
            "cto": "Management",
            "ceo": "Management",
        }

        # Common test types
        self.test_types = {
            "cognitive": "K",
            "knowledge": "K",
            "technical": "K",
            "personality": "P",
            "behavioral": "P",
            "situational": "S",
        }

    def process(self, query: str) -> QueryFeatures:
        """
        Extract features from query.

        Args:
            query: User query string

        Returns:
            QueryFeatures object
        """
        query_lower = query.lower()

        # Extract role/job title
        role = self._extract_role(query, query_lower)

        # Extract job level
        job_level = self._extract_job_level(query_lower)

        # Extract duration
        min_dur, max_dur = self._extract_duration(query_lower)

        # Extract test type preferences
        test_types = self._extract_test_types(query_lower)

        # Extract skills (basic keyword extraction)
        technical_skills = self._extract_technical_skills(query_lower)
        behavioral_skills = self._extract_behavioral_skills(query_lower)

        return QueryFeatures(
            technical_skills=technical_skills,
            behavioral_skills=behavioral_skills,
            job_level=job_level,
            min_duration_minutes=min_dur,
            max_duration_minutes=max_dur,
            test_type_preferences=test_types,
            raw_query=query,
            role=role,
            duration_min=min_dur,
            duration_max=max_dur,
            test_types_needed=test_types,
        )

    def _extract_role(self, query: str, query_lower: str) -> Optional[str]:
        """Extract job role/title from query."""
        # Common role keywords
        role_patterns = [
            r"(?:hire|hiring|looking for|need)\s+(?:a\s+)?([A-Z][A-Za-z\s]+?)(?:\s+who|\s+with|\s+for|,|\.|$)",
            r"(?:^|\s)([A-Z][A-Za-z\s]+?)\s+(?:role|position|job)",
            r"(?:^|\s)(Java|Python|Senior|Junior|Data|Marketing|Sales|Content|QA|COO|CTO|CEO)\s+([A-Z][A-Za-z\s]+)",
        ]

        for pattern in role_patterns:
            match = re.search(pattern, query)
            if match:
                role = (
                    match.group(1).strip()
                    if len(match.groups()) == 1
                    else f"{match.group(1)} {match.group(2)}".strip()
                )
                if len(role) > 3 and len(role) < 50:  # Reasonable role length
                    return role

        # Fallback: look for capitalized words that might be roles
        words = query.split()
        for i, word in enumerate(words):
            if word in [
                "Manager",
                "Engineer",
                "Developer",
                "Analyst",
                "Writer",
                "Assistant",
                "Admin",
                "Consultant",
            ]:
                # Get surrounding context
                start = max(0, i - 2)
                end = min(len(words), i + 1)
                role = " ".join(words[start:end])
                return role

        return None

    def _extract_job_level(self, query: str) -> Optional[str]:
        """Extract job level from query."""
        for keyword, level in self.job_levels.items():
            if keyword in query:
                return level

        # Check for experience years
        if re.search(r"\b0-2\s*years?\b", query):
            return "Entry-Level"
        elif re.search(r"\b2-5\s*years?\b", query):
            return "Professional"
        elif re.search(r"\b5\+?\s*years?\b", query):
            return "Professional"

        return None

    def _extract_duration(self, query: str) -> tuple[Optional[int], Optional[int]]:
        """Extract duration constraints from query."""
        # Look for duration patterns
        patterns = [
            (r"(\d+)\s*-\s*(\d+)\s*min", lambda m: (int(m.group(1)), int(m.group(2)))),
            (r"(\d+)\s*min", lambda m: (int(m.group(1)), int(m.group(1)))),
            (r"(\d+)\s*hour", lambda m: (int(m.group(1)) * 60, int(m.group(1)) * 60)),
            (
                r"(\d+)\s*-\s*(\d+)\s*hour",
                lambda m: (int(m.group(1)) * 60, int(m.group(2)) * 60),
            ),
        ]

        for pattern, extractor in patterns:
            match = re.search(pattern, query)
            if match:
                return extractor(match)

        return None, None

    def _extract_test_types(self, query: str) -> List[str]:
        """Extract test type preferences."""
        found_types = []
        for keyword, test_type in self.test_types.items():
            if keyword in query:
                found_types.append(test_type)
        return list(set(found_types))

    def _extract_technical_skills(self, query: str) -> List[str]:
        """Extract technical skills (basic keyword matching)."""
        technical_keywords = [
            "java",
            "python",
            "javascript",
            "sql",
            "c++",
            "c#",
            "programming",
            "developer",
            "engineer",
            "analyst",
            "data",
            "software",
            "qa",
            "testing",
            "coding",
        ]

        found = []
        for keyword in technical_keywords:
            if keyword in query:
                found.append(keyword)
        return found

    def _extract_behavioral_skills(self, query: str) -> List[str]:
        """Extract behavioral/soft skills."""
        behavioral_keywords = [
            "communication",
            "leadership",
            "teamwork",
            "collaboration",
            "problem solving",
            "analytical",
            "creative",
            "interpersonal",
            "management",
            "organizational",
        ]

        found = []
        for keyword in behavioral_keywords:
            if keyword in query:
                found.append(keyword)
        return found
