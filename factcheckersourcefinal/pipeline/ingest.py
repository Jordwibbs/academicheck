import requests
import sqlite3
import time
import os
from pipeline.config import SQLITE_PATH, RAW_DATA_DIR

def get_headers():
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        return {"x-api-key": key}
    return {}

def search_papers(query, limit=50):
    print(f"Searching for: {query}")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,authors,abstract,year,externalIds,openAccessPdf"
    }
    try:
        response = requests.get(url, params=params, headers=get_headers(), timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except requests.RequestException as e:
        print(f"  Error searching '{query}': {e}")
        return []

def save_paper(conn, paper):
    if not paper.get("abstract"):
        return False

    paper_id = paper.get("paperId")
    title = paper.get("title", "Unknown")
    authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])])
    abstract = paper.get("abstract", "")
    year = paper.get("year")

    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI")

    open_access = paper.get("openAccessPdf") or {}
    url = open_access.get("url", "")

    try:
        conn.execute("""
            INSERT OR IGNORE INTO papers (id, title, authors, abstract, doi, url, source, year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (paper_id, title, authors, abstract, doi, url, "semantic_scholar", year))
        return True
    except sqlite3.Error as e:
        print(f"  DB error saving '{title}': {e}")
        return False

def ingest_topics(topics):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    total_saved = 0

    for topic in topics:
        papers = search_papers(topic, limit=50)
        saved = 0
        for paper in papers:
            if save_paper(conn, paper):
                saved += 1
        conn.commit()
        total_saved += saved
        print(f"  Saved {saved} papers for topic: '{topic}'")
        time.sleep(3)  # rate limit: 1 request per second

    conn.close()
    print(f"\nDone. Total papers saved: {total_saved}")

