from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from api.schemas import ArticleRequest, ReportResponse, ClaimResult, SourceModel, UrlRequest
from pipeline.verify import verify_article
from pipeline.storage import save_report, get_all_reports, get_report_by_id
from newspaper import Article
import uvicorn
import json as json_module

app = FastAPI(
    title="AcademiCheck API",
    description="Fact-checking news articles against academic literature",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/check", response_model=ReportResponse)
def check_article(request: ArticleRequest):
    if not request.article.strip():
        raise HTTPException(status_code=400, detail="Article text cannot be empty")
    if len(request.article) < 100:
        raise HTTPException(status_code=400, detail="Article is too short to fact-check")
    try:
        results = verify_article(request.article)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    claim_results = []
    for r in results:
        sources = [
            SourceModel(
                title=s.get("title"),
                authors=s.get("authors"),
                year=s.get("year"),
                doi=s.get("doi"),
                url=s.get("url"),
                similarity=s.get("similarity")
            )
            for s in r.get("sources", [])
        ]
        claim_results.append(ClaimResult(
            claim=r.get("claim", ""),
            verdict=r.get("verdict", "UNVERIFIABLE"),
            confidence=r.get("confidence", 0.0),
            explanation=r.get("explanation", ""),
            supporting_sources=r.get("supporting_sources", []),
            contradicting_sources=r.get("contradicting_sources", []),
            sources=sources
        ))

    supported = sum(1 for r in claim_results if r.verdict == "SUPPORTED")
    contradicted = sum(1 for r in claim_results if r.verdict == "CONTRADICTED")
    unverifiable = sum(1 for r in claim_results if r.verdict == "UNVERIFIABLE")

    report_id = save_report(
        article_text=request.article,
        results=[r.__dict__ if hasattr(r, '__dict__') else r for r in results],
        article_title=None,
        article_url=None
    )

    return ReportResponse(
        total_claims=len(claim_results),
        supported=supported,
        contradicted=contradicted,
        unverifiable=unverifiable,
        results=claim_results,
        report_id=report_id
    )

@app.post("/check-stream")
def check_article_stream(request: ArticleRequest):
    if not request.article.strip():
        raise HTTPException(status_code=400, detail="Article text cannot be empty")
    if len(request.article) < 100:
        raise HTTPException(status_code=400, detail="Article is too short to fact-check")

    def generate():
        from pipeline.extract import extract_claims
        from pipeline.verify import verify_claim

        selected_model = request.model or "mistral"

        claims = extract_claims(request.article)
        yield f"data: {json_module.dumps({'type': 'claims_found', 'count': len(claims), 'model': selected_model})}\n\n"

        results = []
        for i, claim in enumerate(claims):
            result = verify_claim(claim, model=selected_model)
            results.append(result)

            sources = [
                {
                    "title": s.get("title"),
                    "authors": s.get("authors"),
                    "year": s.get("year"),
                    "doi": s.get("doi"),
                    "url": s.get("url"),
                    "similarity": s.get("similarity")
                }
                for s in result.get("sources", [])
            ]

            yield f"data: {json_module.dumps({'type': 'claim_result', 'index': i, 'total': len(claims), 'claim': result.get('claim'), 'verdict': result.get('verdict'), 'confidence': result.get('confidence'), 'explanation': result.get('explanation'), 'sources': sources})}\n\n"

        report_id = save_report(
            article_text=request.article,
            results=results,
            article_title=None,
            article_url=None
        )

        supported = sum(1 for r in results if r.get("verdict") == "SUPPORTED")
        contradicted = sum(1 for r in results if r.get("verdict") == "CONTRADICTED")
        unverifiable = sum(1 for r in results if r.get("verdict") == "UNVERIFIABLE")

        yield f"data: {json_module.dumps({'type': 'complete', 'report_id': report_id, 'supported': supported, 'contradicted': contradicted, 'unverifiable': unverifiable, 'model': selected_model})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/fetch-article")
def fetch_article(request: UrlRequest):
    try:
        article = Article(request.url)
        article.config.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        article.config.request_timeout = 15
        article.download()
        article.parse()
        if not article.text or len(article.text) < 100:
            raise HTTPException(
                status_code=400,
                detail="Could not extract enough text from that URL. Try copying and pasting the article text directly."
            )
        return {
            "title": article.title,
            "text": article.text
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch article: {str(e)}")

@app.get("/history")
def get_history():
    return get_all_reports()

@app.get("/history/{report_id}")
def get_report(report_id: str):
    report = get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

@app.get("/history-page")
def history_page():
    return FileResponse("frontend/history.html")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)