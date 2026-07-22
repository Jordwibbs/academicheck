import sqlite3
import chromadb
import uuid
from sentence_transformers import SentenceTransformer
from pipeline.config import SQLITE_PATH, CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

def embed_papers():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to databases...")
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection(
        name="papers",
        metadata={"hnsw:space": "cosine"}
    )

    # get already embedded paper IDs from ChromaDB
    existing = collection.get(include=[])
    existing_ids = set()
    for chunk_id in existing["ids"]:
        # chunk IDs are in format {paper_id}_chunk_{n}
        paper_id = "_".join(chunk_id.split("_")[:-2])
        existing_ids.add(paper_id)

    print(f"Already embedded: {len(existing_ids)} papers")

    # only fetch papers not yet embedded
    cur.execute("SELECT id, title, abstract FROM papers WHERE abstract IS NOT NULL")
    all_papers = cur.fetchall()
    papers = [(pid, title, abstract) for pid, title, abstract in all_papers 
              if pid not in existing_ids]

    print(f"New papers to embed: {len(papers)}")

    if not papers:
        print("Nothing new to embed!")
        conn.close()
        return

    total_chunks = 0
    for i, (paper_id, title, abstract) in enumerate(papers):
        text = f"{title}. {abstract}"
        chunks = chunk_text(text)

        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []

        for j, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk_{j}"
            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk)
            chunk_metadatas.append({
                "paper_id": paper_id,
                "chunk_index": j,
                "title": title[:100]
            })

            cur.execute("""
                INSERT OR IGNORE INTO chunks (id, paper_id, chunk_index, text)
                VALUES (?, ?, ?, ?)
            """, (chunk_id, paper_id, j, chunk))

        embeddings = model.encode(chunk_texts, show_progress_bar=False).tolist()
        collection.upsert(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=chunk_metadatas
        )

        total_chunks += len(chunks)

        if (i + 1) % 50 == 0:
            print(f"  Embedded {i + 1}/{len(papers)} new papers ({total_chunks} chunks so far)")

    conn.commit()
    conn.close()
    print(f"\nDone. New chunks embedded: {total_chunks}")
    print(f"ChromaDB total size: {collection.count()}")