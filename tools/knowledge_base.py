"""RAG retrieval tool over the local Chroma knowledge base.

Embeddings are computed with Chroma's bundled ONNX all-MiniLM-L6-v2
runtime -- the same model used at ingest time, so retrieval works fully
offline once the index is built, with no PyTorch dependency. If the index
is missing (e.g. on a fresh clone), it is rebuilt from data/raw on first use.
"""

import os

import chromadb
from chromadb.utils import embedding_functions

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION_NAME = "agrisense_kb"

_ef_instance = None
_collection = None

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


def _detect_crop(text):
    """Return the canonical crop key referenced in text, or None."""
    if not text:
        return None
    lower = text.lower()
    for canonical, aliases in CROP_ALIASES.items():
        if any(a in lower for a in aliases):
            return canonical
    return None


def _ef():
    """Singleton ONNX MiniLM embedding function (shared by ingest + query)."""
    global _ef_instance
    if _ef_instance is None:
        _ef_instance = embedding_functions.ONNXMiniLM_L6_V2()
    return _ef_instance


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_DIR)
        try:
            _collection = client.get_collection(COLLECTION_NAME, embedding_function=_ef())
        except Exception:
            # Fresh clone / missing index: build it from data/raw, then open.
            from data.ingest import build_index
            build_index()
            _collection = client.get_collection(COLLECTION_NAME, embedding_function=_ef())
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
    collection = _get_collection()

    # Auto-detect the crop from the query if the caller didn't pass one.
    effective_crop = (crop_filter or _detect_crop(query))
    if effective_crop:
        effective_crop = effective_crop.lower()

    # Pull a wide pool so crop-matching chunks are available to promote.
    # Chroma embeds `query` with the collection's own embedding function,
    # so it must be the same one used at ingest time (ONNX MiniLM).
    pool_size = max(n_results * 4, 12)
    results = collection.query(query_texts=[query], n_results=pool_size)

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
                "source_file": meta.get("source_file"),
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
