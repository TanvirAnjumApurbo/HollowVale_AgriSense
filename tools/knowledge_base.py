"""RAG retrieval tool over the local Chroma knowledge base.

Embeddings are computed locally with sentence-transformers -- same model
used at ingest time -- so retrieval works fully offline once the index
has been built via data/ingest.py.
"""

import os

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "agrisense_kb"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_collection = None

CROP_ALIASES = {
    "rice": ["rice", "boro", "paddy", "aman", "aus"],
    "wheat": ["wheat"],
    "maize": ["maize"],
    "potato": ["potato"],
    "lentil": ["lentil", "masur"],
    "jute": ["jute"],
}


def _detect_crop(text):
    """Return the canonical crop key referenced in text, or None."""
    if not text:
        return None
    lower = text.lower()
    for canonical, aliases in CROP_ALIASES.items():
        if any(a in lower for a in aliases):
            return canonical
    return None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def search_knowledge_base(query, n_results=4, crop_filter=None):
    """Retrieve the most relevant agronomy passages for a query.

    If crop_filter is given (or a crop is detected in the query itself),
    chunks tagged with that crop are strongly preferred: we pull a wide
    candidate pool, keep the crop-matching chunks first, and only backfill
    with other chunks if there aren't enough. This is done in Python
    because Chroma's metadata `where` has no substring operator (its
    `$contains` applies to document text, not metadata), and because a
    generic query like "fertilizer timing" otherwise pulls whichever crop
    has the longest fertilizer section rather than the one being asked
    about.
    """
    model = _get_model()
    collection = _get_collection()

    # Auto-detect the crop from the query if the caller didn't pass one.
    effective_crop = (crop_filter or _detect_crop(query))
    if effective_crop:
        effective_crop = effective_crop.lower()

    query_embedding = model.encode([query]).tolist()

    # Pull a wide pool so crop-matching chunks are available to promote.
    pool_size = max(n_results * 4, 12)
    results = collection.query(query_embeddings=query_embedding, n_results=pool_size)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)

    ranked = list(zip(docs, metas, distances))

    if effective_crop:
        def matches(meta):
            crops = (meta.get("crops") or "").lower()
            return effective_crop in [c.strip() for c in crops.split(",")]

        matching = [r for r in ranked if matches(r[1])]
        others = [r for r in ranked if not matches(r[1])]
        # Crop-matching chunks first (still in similarity order), then
        # backfill with the best non-matching chunks up to n_results.
        ranked = matching + others

    ranked = ranked[:n_results]

    return {
        "query": query,
        "crop_filter_applied": effective_crop,
        "results": [
            {
                "text": doc,
                "source_title": meta.get("title"),
                "source_url": meta.get("source_url"),
                "relevance_distance": dist,
            }
            for doc, meta, dist in ranked
        ],
    }


if __name__ == "__main__":
    import json

    for q in [
        "fertilizer dose for boro rice",
        "what soil is best for lentil",
        "wheat sowing time",
        "pests affecting rice during flowering",
    ]:
        print(f"\n=== {q} ===")
        out = search_knowledge_base(q, n_results=2)
        for r in out["results"]:
            print(f"- ({r['source_title']}) {r['text'][:150]!r}...")
