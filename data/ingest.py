"""Chunk the collected agronomy docs, embed them locally, and store in Chroma.

Run once (or whenever data/raw/ changes) with:
    python data/ingest.py

Uses Chroma's bundled ONNX all-MiniLM-L6-v2 runtime for embeddings so the
whole RAG pipeline is free and works offline -- no external embedding API
call and no PyTorch dependency at ingest time or at query time.
"""

import os
import re

import chromadb
from chromadb.utils import embedding_functions

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "agrisense_kb"

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
    "mustard": ["mustard", "sarisha", "rapeseed", "canola"],
    "onion": ["onion", "piyaj", "peyaj"],
    "chili": ["chili", "morich"],
    "tomato": ["tomato"],
    "chickpea": ["chickpea", "chick pea", "chola", "chhola", "chana"],
}


def _alias_present(alias, lower_text):
    """Whole-word alias match. Substring matching would tag 'boron' as `boro`
    (rice), 'aus' inside longer words, etc. -- and the short rice aliases in
    particular are the ones that collide. The query-side `_detect_crop` in
    tools/knowledge_base.py must use the same rule so tags and detection agree.
    """
    return re.search(rf"\b{re.escape(alias)}\b", lower_text) is not None


def guess_crop_tags(text):
    """Return canonical crop keys mentioned in the text.

    Aliases are collapsed to the canonical key used by the financial
    engine (e.g. 'boro'/'paddy' -> 'rice', 'masur' -> 'lentil') so the
    crop_filter passed by the agent lines up with these tags.
    """
    lower = text.lower()
    tags = []
    for canonical, aliases in CROP_ALIASES.items():
        if any(_alias_present(a, lower) for a in aliases):
            tags.append(canonical)
    return tags


def build_index():
    os.makedirs(DB_DIR, exist_ok=True)
    docs = load_raw_documents()
    if not docs:
        raise SystemExit(f"No .md files found in {RAW_DIR} -- add knowledge base sources first.")

    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # Attach the ONNX embedding function to the collection so Chroma embeds
    # documents at add() time and queries at query() time with the same
    # model -- query-side retrieval must use this identical EF or vectors
    # won't match.
    collection = client.create_collection(
        COLLECTION_NAME,
        embedding_function=embedding_functions.ONNXMiniLM_L6_V2(),
    )

    ids, texts, metadatas = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        # Crop tagging drives the crop_filter promotion at query time, so it
        # must reflect what a doc is ABOUT -- not every crop it mentions in
        # passing. A single-crop doc names its crop in the filename (e.g.
        # `mustard.md`, `rice_boro_bamis.md`); trust that and ignore rotation
        # asides like "fits between Aman and Boro rice" (which otherwise tag
        # mustard/lentil/jute as `rice` too, making the `rice` filter match
        # nearly everything and silently do nothing). A topic doc with no crop
        # in its filename (the crop calendar, the FRG guide, soil/season notes)
        # is legitimately multi-crop, so fall back to tagging by its content.
        # Every chunk carries the whole doc's crops so a short dose paragraph
        # that doesn't repeat the crop name stays findable by crop filter.
        # Normalise separators first: whole-word matching treats "_" as a word
        # character, so `\bjute\b` would miss "jute_cultivation.md".
        filename_words = doc["filename"].replace("_", " ").replace("-", " ")
        filename_crops = guess_crop_tags(filename_words)
        doc_crops = sorted(filename_crops or guess_crop_tags(doc["text"]))
        for i, chunk in enumerate(chunks):
            chunk_crops = doc_crops
            ids.append(f"{doc['filename']}::{i}")
            texts.append(chunk)
            metadatas.append({
                "source_file": doc["filename"],
                "title": doc["title"],
                "source_url": doc["source"],
                "crops": ",".join(chunk_crops) or "general",
            })

    print(f"Embedding {len(texts)} chunks from {len(docs)} documents (ONNX MiniLM)...")
    # No explicit embeddings= : Chroma embeds the documents with the
    # collection's ONNX embedding function.
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"Indexed {len(texts)} chunks into '{COLLECTION_NAME}' at {DB_DIR}")


if __name__ == "__main__":
    build_index()
