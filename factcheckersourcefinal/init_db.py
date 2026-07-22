import sqlite3
import os

os.makedirs("db", exist_ok=True)
conn = sqlite3.connect("db/papers.db")
cur = conn.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS papers (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    authors     TEXT,
    abstract    TEXT,
    doi         TEXT,
    url         TEXT,
    source      TEXT,
    year        INTEGER,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    paper_id    TEXT NOT NULL,
    chunk_index INTEGER,
    text        TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE IF NOT EXISTS reports (
    id              TEXT PRIMARY KEY,
    article_text    TEXT NOT NULL,
    article_title   TEXT,
    article_url     TEXT,
    total_claims    INTEGER,
    supported       INTEGER,
    contradicted    INTEGER,
    unverifiable    INTEGER,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claim_results (
    id                  TEXT PRIMARY KEY,
    report_id           TEXT NOT NULL,
    claim               TEXT NOT NULL,
    verdict             TEXT NOT NULL,
    confidence          REAL,
    explanation         TEXT,
    supporting_sources  TEXT,
    contradicting_sources TEXT,
    sources_json        TEXT,
    FOREIGN KEY (report_id) REFERENCES reports(id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_claims_report ON claim_results(report_id);
""")

conn.commit()
conn.close()
print("Database initialised at db/papers.db")