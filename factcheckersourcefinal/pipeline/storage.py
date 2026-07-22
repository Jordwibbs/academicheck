import sqlite3
import uuid
import json
from datetime import datetime
from pipeline.config import SQLITE_PATH

def save_report(article_text, results, article_title=None, article_url=None):
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()

    report_id = str(uuid.uuid4())
    supported = sum(1 for r in results if r.get("verdict") == "SUPPORTED")
    contradicted = sum(1 for r in results if r.get("verdict") == "CONTRADICTED")
    unverifiable = sum(1 for r in results if r.get("verdict") == "UNVERIFIABLE")

    cur.execute("""
        INSERT INTO reports (id, article_text, article_title, article_url,
                            total_claims, supported, contradicted, unverifiable)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_id,
        article_text[:2000],
        article_title,
        article_url,
        len(results),
        supported,
        contradicted,
        unverifiable
    ))

    for r in results:
        cur.execute("""
            INSERT INTO claim_results
                (id, report_id, claim, verdict, confidence, explanation,
                 supporting_sources, contradicting_sources, sources_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            report_id,
            r.get("claim", ""),
            r.get("verdict", "UNVERIFIABLE"),
            r.get("confidence", 0.0),
            r.get("explanation", ""),
            json.dumps(r.get("supporting_sources", [])),
            json.dumps(r.get("contradicting_sources", [])),
            json.dumps([
                {
                    "title": s.get("title"),
                    "authors": s.get("authors"),
                    "year": s.get("year"),
                    "doi": s.get("doi"),
                    "url": s.get("url"),
                    "similarity": s.get("similarity")
                }
                for s in r.get("sources", [])
            ])
        ))

    conn.commit()
    conn.close()
    return report_id

def get_all_reports():
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, article_title, article_url, total_claims,
               supported, contradicted, unverifiable, created_at
        FROM reports
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "article_title": r[1],
            "article_url": r[2],
            "total_claims": r[3],
            "supported": r[4],
            "contradicted": r[5],
            "unverifiable": r[6],
            "created_at": r[7]
        }
        for r in rows
    ]

def get_report_by_id(report_id):
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, article_text, article_title, article_url,
               total_claims, supported, contradicted, unverifiable, created_at
        FROM reports WHERE id = ?
    """, (report_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    cur.execute("""
        SELECT claim, verdict, confidence, explanation,
               supporting_sources, contradicting_sources, sources_json
        FROM claim_results WHERE report_id = ?
    """, (report_id,))
    claims = cur.fetchall()
    conn.close()

    return {
        "id": row[0],
        "article_text": row[1],
        "article_title": row[2],
        "article_url": row[3],
        "total_claims": row[4],
        "supported": row[5],
        "contradicted": row[6],
        "unverifiable": row[7],
        "created_at": row[8],
        "results": [
            {
                "claim": c[0],
                "verdict": c[1],
                "confidence": c[2],
                "explanation": c[3],
                "supporting_sources": json.loads(c[4] or "[]"),
                "contradicting_sources": json.loads(c[5] or "[]"),
                "sources": json.loads(c[6] or "[]")
            }
            for c in claims
        ]
    }