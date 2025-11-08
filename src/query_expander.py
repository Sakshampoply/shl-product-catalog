"""
Query Expansion using Gemini 2.5 Flash.
Expands queries to include implicit competencies and related skills.
"""

import os
from typing import Optional
from dotenv import load_dotenv

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


class QueryExpander:
    """Expands queries using Gemini 2.5 Flash to understand implicit requirements."""

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the query expander.

        Args:
            model_name: Gemini model to use for expansion
        """
        self.model_name = model_name
        self.enabled = GEMINI_AVAILABLE

        if self.enabled:
            self.model = genai.GenerativeModel(model_name)

    def expand_query(self, query: str) -> str:
        """
        Expand query to include implicit competencies and related skills.

        Args:
            query: Original user query

        Returns:
            Expanded query with additional context
        """
        if not self.enabled:
            return query

        try:
            prompt = f"""You are an expert HR assessment specialist. Analyze this job/role query and identify ALL relevant competencies, skills, and assessment areas that should be tested.

Query: "{query}"

Think about:
1. Technical skills explicitly mentioned
2. Soft skills and behavioral competencies implied by the role
3. Communication skills needed (written, verbal, language proficiency)
4. Cognitive abilities required
5. Job-specific assessments (simulations, scenarios)
6. Leadership/management skills if applicable

Provide a comprehensive list of assessment keywords separated by commas. Include synonyms and related terms. Be specific and practical.

Format: Return ONLY comma-separated keywords, no explanations."""

            response = self.model.generate_content(prompt)
            expansion = response.text.strip()

            # Combine original query with expansion
            expanded = f"{query} {expansion}"

            return expanded

        except Exception as e:
            print(f"Warning: Query expansion failed ({e}), using original query")
            return query

    def expand_query_structured(self, query: str) -> dict:
        """
        Expand query with structured output for different competency types.

        Args:
            query: Original user query

        Returns:
            Dictionary with categorized expansions
        """
        if not self.enabled:
            return {
                "original": query,
                "technical_skills": [],
                "behavioral_skills": [],
                "communication_skills": [],
                "cognitive_skills": [],
                "job_specific": [],
            }

        try:
            prompt = f"""You are an expert HR assessment specialist. Analyze this job/role query and categorize relevant competencies.

Query: "{query}"

Provide keywords in these categories:
1. TECHNICAL SKILLS: Programming languages, tools, technologies explicitly or implicitly needed
2. BEHAVIORAL SKILLS: Soft skills like teamwork, leadership, problem-solving, adaptability
3. COMMUNICATION SKILLS: Written, verbal, presentation, language proficiency needs
4. COGNITIVE SKILLS: Analytical thinking, reasoning, decision-making abilities
5. JOB-SPECIFIC: Role-specific simulations, scenarios, or domain knowledge

Format your response EXACTLY as:
TECHNICAL: keyword1, keyword2, keyword3
BEHAVIORAL: keyword1, keyword2, keyword3
COMMUNICATION: keyword1, keyword2, keyword3
COGNITIVE: keyword1, keyword2, keyword3
JOB_SPECIFIC: keyword1, keyword2, keyword3

Keep it concise - 3-5 keywords per category."""

            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Parse structured response
            result = {
                "original": query,
                "technical_skills": [],
                "behavioral_skills": [],
                "communication_skills": [],
                "cognitive_skills": [],
                "job_specific": [],
            }

            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("TECHNICAL:"):
                    result["technical_skills"] = [
                        s.strip()
                        for s in line.replace("TECHNICAL:", "").split(",")
                        if s.strip()
                    ]
                elif line.startswith("BEHAVIORAL:"):
                    result["behavioral_skills"] = [
                        s.strip()
                        for s in line.replace("BEHAVIORAL:", "").split(",")
                        if s.strip()
                    ]
                elif line.startswith("COMMUNICATION:"):
                    result["communication_skills"] = [
                        s.strip()
                        for s in line.replace("COMMUNICATION:", "").split(",")
                        if s.strip()
                    ]
                elif line.startswith("COGNITIVE:"):
                    result["cognitive_skills"] = [
                        s.strip()
                        for s in line.replace("COGNITIVE:", "").split(",")
                        if s.strip()
                    ]
                elif line.startswith("JOB_SPECIFIC:"):
                    result["job_specific"] = [
                        s.strip()
                        for s in line.replace("JOB_SPECIFIC:", "").split(",")
                        if s.strip()
                    ]

            return result

        except Exception as e:
            print(f"Warning: Structured expansion failed ({e}), using original query")
            return {
                "original": query,
                "technical_skills": [],
                "behavioral_skills": [],
                "communication_skills": [],
                "cognitive_skills": [],
                "job_specific": [],
            }
