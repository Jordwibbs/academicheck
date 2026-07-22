import sqlite3
import chromadb
from pipeline.config import SQLITE_PATH, CHROMA_DIR

def test_ingestion():
    print("=" * 50)
    print("INGESTION TEST REPORT")
    print("=" * 50)

    # --- SQLite Tests ---
    print("\n[1] SQLite Database Tests")
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()

    # total papers
    cur.execute("SELECT COUNT(*) FROM papers")
    total_papers = cur.fetchone()[0]
    print(f"  Total papers ingested: {total_papers}")
    assert total_papers > 0, "FAIL: No papers in database"
    print(f"  Papers count > 0")

    # papers with abstracts
    cur.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND abstract != ''")
    with_abstracts = cur.fetchone()[0]
    print(f"  Papers with abstracts: {with_abstracts}")
    assert with_abstracts == total_papers, "FAIL: Some papers missing abstracts"
    print(f"   All papers have abstracts")

    # papers with DOIs
    cur.execute("SELECT COUNT(*) FROM papers WHERE doi IS NOT NULL")
    with_dois = cur.fetchone()[0]
    print(f"  Papers with DOIs: {with_dois} ({round(with_dois/total_papers*100)}%)")

    # papers with URLs
    cur.execute("SELECT COUNT(*) FROM papers WHERE url IS NOT NULL AND url != ''")
    with_urls = cur.fetchone()[0]
    print(f"  Papers with URLs: {with_urls} ({round(with_urls/total_papers*100)}%)")

    # source distribution
    cur.execute("SELECT source, COUNT(*) FROM papers GROUP BY source")
    sources = cur.fetchall()
    print(f"  Sources:")
    for source, count in sources:
        print(f"    {source}: {count} papers")

    # year distribution
    cur.execute("SELECT MIN(year), MAX(year) FROM papers WHERE year IS NOT NULL")
    min_year, max_year = cur.fetchone()
    print(f"  Year range: {min_year} - {max_year}")

    # sample paper
    cur.execute("SELECT title, authors, year, doi FROM papers LIMIT 1")
    sample = cur.fetchone()
    print(f"\n  Sample paper:")
    print(f"    Title: {sample[0]}")
    print(f"    Authors: {sample[1]}")
    print(f"    Year: {sample[2]}")
    print(f"    DOI: {sample[3]}")

    # --- Chunks Tests ---
    print("\n[2] Chunks Table Tests")
    cur.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cur.fetchone()[0]
    print(f"  Total chunks: {total_chunks}")
    assert total_chunks > 0, "FAIL: No chunks in database"
    print(f"   Chunks count > 0")

    # average chunks per paper
    avg_chunks = round(total_chunks / total_papers, 2)
    print(f"  Average chunks per paper: {avg_chunks}")

    # check foreign key integrity
    cur.execute("""
        SELECT COUNT(*) FROM chunks 
        WHERE paper_id NOT IN (SELECT id FROM papers)
    """)
    orphan_chunks = cur.fetchone()[0]
    assert orphan_chunks == 0, f"FAIL: {orphan_chunks} orphan chunks found"
    print(f"   No orphan chunks (foreign key integrity OK)")

    # check chunk ID format
    cur.execute("SELECT id FROM chunks LIMIT 5")
    sample_ids = cur.fetchall()
    print(f"  Sample chunk IDs:")
    for chunk_id in sample_ids:
        print(f"    {chunk_id[0]}")
        assert "_chunk_" in chunk_id[0], f"FAIL: Unexpected chunk ID format: {chunk_id[0]}"
    print(f"   Chunk ID format correct")

    conn.close()

    # --- ChromaDB Tests ---
    print("\n[3] ChromaDB Vector Store Tests")
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection(
        name="papers",
        metadata={"hnsw:space": "cosine"}
    )

    chroma_count = collection.count()
    print(f"  Total vectors in ChromaDB: {chroma_count}")
    assert chroma_count > 0, "FAIL: No vectors in ChromaDB"
    print(f"   ChromaDB collection not empty")

    # check SQLite and ChromaDB are in sync
    if chroma_count == total_chunks:
        print(f"   SQLite chunks and ChromaDB vectors in sync ({chroma_count})")
    else:
        print(f"    Mismatch: SQLite has {total_chunks} chunks, ChromaDB has {chroma_count} vectors")

    # test a sample retrieval
    print("\n[4] Sample Retrieval Test")
    from sentence_transformers import SentenceTransformer
    from pipeline.config import EMBEDDING_MODEL

    print(f"  Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    test_claim = "vaccines cause autism"
    embedding = model.encode(test_claim).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print(f"  Test claim: '{test_claim}'")
    print(f"  Top 3 results:")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = round(1 - dist, 4)
        print(f"    [{i+1}] {meta.get('title', 'Unknown')[:60]} (similarity: {similarity})")

    assert len(results["documents"][0]) == 3, "FAIL: Expected 3 results"
    print(f"   Retrieval returned 3 results")

    # --- Summary ---
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Papers in SQLite:      {total_papers}")
    print(f"  Chunks in SQLite:      {total_chunks}")
    print(f"  Vectors in ChromaDB:   {chroma_count}")
    print(f"  Avg chunks per paper:  {avg_chunks}")
    print(f"  Year range:            {min_year} - {max_year}")
    print(f"\n  All tests passed ")

if __name__ == "__main__":
    test_ingestion()