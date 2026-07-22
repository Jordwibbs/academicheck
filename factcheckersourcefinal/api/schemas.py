from pydantic import BaseModel
from typing import List, Optional

class ArticleRequest(BaseModel):
    article: str
    model: Optional[str] = "mistral"
    article_title: Optional[str] = None
    article_url: Optional[str] = None

class UrlRequest(BaseModel):
    url: str

class SourceModel(BaseModel):
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    similarity: Optional[float] = None

class ClaimResult(BaseModel):
    claim: str
    verdict: str
    confidence: float
    explanation: str
    supporting_sources: List[int] = []
    contradicting_sources: List[int] = []
    sources: List[SourceModel] = []

class ReportResponse(BaseModel):
    report_id: Optional[str] = None
    total_claims: int
    supported: int
    contradicted: int
    unverifiable: int
    model_used: Optional[str] = None
    results: List[ClaimResult]