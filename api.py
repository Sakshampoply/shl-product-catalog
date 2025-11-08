"""
FastAPI Backend for SHL Assessment Recommendation System.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.multi_vector_retriever_v2 import MultiVectorRetrieverV2

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Intelligent assessment recommendation system using Multi-Vector RAG with Query Deconstruction",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever (will be loaded on startup)
retriever: Optional[MultiVectorRetrieverV2] = None


# Request/Response models
class RecommendationRequest(BaseModel):
    """Request model for recommendations."""

    query: str = Field(..., description="Natural language query or job description")
    top_k: Optional[int] = Field(
        10, ge=1, le=10, description="Number of recommendations (1-10)"
    )


class Assessment(BaseModel):
    """Assessment model."""

    name: str = Field(..., description="Name of the assessment")
    url: str = Field(..., description="URL of the assessment in SHL catalog")
    description: Optional[str] = Field(
        None, description="Description of the assessment"
    )
    test_types: List[str] = Field(
        default_factory=list, description="Test type keys (K, P, S, etc.)"
    )
    duration_minutes: Optional[int] = Field(
        None, description="Assessment duration in minutes"
    )
    job_levels: List[str] = Field(
        default_factory=list, description="Applicable job levels"
    )
    score: float = Field(..., description="Relevance score")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""

    query: str = Field(..., description="Original query")
    recommendations: List[Assessment] = Field(
        ..., description="List of recommended assessments"
    )
    total: int = Field(..., description="Total number of recommendations returned")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize retriever on startup."""
    global retriever
    try:
        retriever = MultiVectorRetrieverV2(data_dir="data", gemini_dir="data/gemini")
        print("✓ Multi-Vector Retriever V2 initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize retriever: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API is running.

    Returns:
        HealthResponse with status and message
    """
    if retriever is None:
        return HealthResponse(
            status="error", message="Service is not ready - retriever not initialized"
        )

    return HealthResponse(status="ok", message="Service is running")


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get assessment recommendations based on query.

    Args:
        request: RecommendationRequest with query and optional top_k

    Returns:
        RecommendationResponse with list of recommended assessments

    Raises:
        HTTPException: If retriever not initialized or error occurs
    """
    if retriever is None:
        raise HTTPException(
            status_code=503, detail="Service not ready - retriever not initialized"
        )

    try:
        # Get recommendations
        results = retriever.retrieve(query=request.query, top_k=request.top_k)

        # Convert to response format
        assessments = []
        for result in results:
            assessment = Assessment(
                name=result.get("name", ""),
                url=result.get("url", ""),
                description=result.get("description"),
                test_types=result.get("test_type_keys", []),
                duration_minutes=result.get("assessment_length_minutes"),
                job_levels=result.get("job_levels", []),
                score=result.get("retrieval_score", 0.0),
            )
            assessments.append(assessment)

        return RecommendationResponse(
            query=request.query, recommendations=assessments, total=len(assessments)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating recommendations: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SHL Assessment Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/recommend": "Get assessment recommendations (POST)",
            "/docs": "API documentation (Swagger UI)",
            "/redoc": "API documentation (ReDoc)",
        },
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
