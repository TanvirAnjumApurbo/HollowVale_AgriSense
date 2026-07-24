"""Chunk the collected agronomy docs, embed them locally, and store in Chroma.

Run once (or whenever data/raw/ changes) with:
    python data/ingest.py

Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings so the whole
RAG pipeline is free and works offline -- no external embedding API call
at ingest time or at query time.
"""

import os
import re

import chromadb
from sentence_transformers import SentenceTransformer

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "agrisense_kb"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE_CHARS = 900
CHUNK_OVERLAP_CHARS = 150


def load_raw_documents():
    docs = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(RAW_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source_match = re.search(r"^Source:\s*(.+)$", text, re.MULTILINE)
        source = source_match.group(1).strip() if source_match else fname
        title_match = re.search(r"^#\s*(.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else fname
        docs.append({"filename": fname, "title": title, "source": source, "text": text})
    return docs


def chunk_text(text, size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    """Simple sliding-window chunker on paragraphs, falling back to raw
    character windows for long single paragraphs (e.g. tables)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > size:
                for i in range(0, len(para), size - overlap):
                    chunks.append(para[i:i + size])
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


CROP_ALIASES = {
    "rice": ["rice", "boro", "paddy", "aman", "aus"],
    "wheat": ["wheat"],
    "maize": ["maize"],
    "potato": ["potato"],
    "lentil": ["lentil", "masur"],
    "jute": ["jute"],
}


def guess_crop_tags(text):
    """Return canonical crop keys mentioned in the text.

    Aliases are collapsed to the canonical key used by the financial
    engine (e.g. 'boro'/'paddy' -> 'rice', 'masur' -> 'lentil') so the
    crop_filter passed by the agent lines up with these tags.
    """
    lower = text.lower()
    tags = []
    for canonical, aliases in CROP_ALIASES.items():
        if any(a in lower for a in aliases):
            tags.append(canonical)
    return tags


def build_index():
    os.makedirs(DB_DIR, exist_ok=True)
    docs = load_raw_documents()
    if not docs:
        raise SystemExit(f"No .md files found in {RAW_DIR} -- add knowledge base sources first.")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, texts, metadatas = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        # Tag every chunk with the crops of the WHOLE document, not just
        # the crops named in that chunk -- otherwise a short fertilizer or
        # dose paragraph that doesn't repeat the crop name (e.g. lentil's
        # one-line dose block) becomes unfindable by crop filter.
        doc_crops = guess_crop_tags(doc["filename"] + " " + doc["text"])
        for i, chunk in enumerate(chunks):
            chunk_crops = sorted(set(doc_crops) | set(guess_crop_tags(chunk)))
            ids.append(f"{doc['filename']}::{i}")
            texts.append(chunk)
            metadatas.append({
                "source_file": doc["filename"],
                "title": doc["title"],
                "source_url": doc["source"],
                "crops": ",".join(chunk_crops) or "general",
            })

    print(f"Embedding {len(texts)} chunks from {len(docs)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(texts)} chunks into '{COLLECTION_NAME}' at {DB_DIR}")


if __name__ == "__main__":
    build_index()
