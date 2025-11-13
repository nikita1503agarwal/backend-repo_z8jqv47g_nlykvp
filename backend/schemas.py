from typing import Optional
from pydantic import BaseModel, Field

class Submission(BaseModel):
    text: str = Field(..., description="Input text to analyze")
    url: Optional[str] = Field(None, description="Optional source URL for context")

class AnalysisResult(BaseModel):
    id: str
    text: str
    source_url: Optional[str]
    plagiarism_score: float
    fake_news_score: float
    verdict: str
