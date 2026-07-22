import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from pipeline.config import SQLITE_PATH, CHROMA_DIR, EMBEDDING_MODEL, TOP_K_CHUNKS

_model = None
_collection = None

def get_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_collection():
    global _collection
    if _collection is None:
        chroma = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = chroma.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def get_paper_metadata(conn, paper_id):
    cur = conn.cursor()
    cur.execute("""
        SELECT title, authors, year, doi, url
        FROM papers WHERE id = ?
    """, (paper_id,))
    row = cur.fetchone()
    if row:
        return {
            "title": row[0],
            "authors": row[1],
            "year": row[2],
            "doi": row[3],
            "url": row[4]
        }
    return {}

def retrieve(claim, top_k=TOP_K_CHUNKS):
    model = get_model()
    collection = get_collection()
    conn = sqlite3.connect(SQLITE_PATH)

    embedding = model.encode(claim).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    seen_papers = set()

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        paper_id = meta.get("paper_id")
        paper_meta = get_paper_metadata(conn, paper_id)

        similarity = round(1 - distance, 4)

        retrieved.append({
            "chunk_text": doc,
            "similarity": similarity,
            "paper_id": paper_id,
            "title": paper_meta.get("title", "Unknown"),
            "authors": paper_meta.get("authors", "Unknown"),
            "year": paper_meta.get("year"),
            "doi": paper_meta.get("doi"),
            "url": paper_meta.get("url"),
            "already_seen": paper_id in seen_papers
        })

        seen_papers.add(paper_id)

    conn.close()
    return retrieved

def format_context(retrieved_chunks):
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        authors = chunk.get("authors", "Unknown")
        year = chunk.get("year", "n.d.")
        title = chunk.get("title", "Unknown")
        doi = chunk.get("doi", "")

        citation = f"{authors} ({year}). {title}."
        if doi:
            citation += f" DOI: {doi}"

        context_parts.append(
            f"[Source {i+1}] {citation}\n"
            f"Relevance: {chunk['similarity']}\n"
            f"Extract: {chunk['chunk_text']}\n"
        )

    return "\n".join(context_parts)

if __name__ == "__main__":
    test_claims = [
        "Vaccines cause autism",
        "Social media spreads misinformation faster than traditional media",
        "AI can reliably detect fake news"
    ]

    for claim in test_claims:
        print(f"\nClaim: {claim}")
        print("-" * 60)
        results = retrieve(claim, top_k=3)
        for r in results:
            print(f"  [{r['similarity']}] {r['title']} ({r['year']})")