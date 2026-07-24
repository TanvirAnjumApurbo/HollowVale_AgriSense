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

    crop_filter, if given (e.g. "rice"), restricts results to chunks
    tagged with that crop where possible; falls back to unfiltered
    search if nothing matches.
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()

    where = None
    if crop_filter:
        where = {"crops": {"$contains": crop_filter.lower()}}

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
        )
    except Exception:
        results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)

    if not docs and crop_filter:
        return search_knowledge_base(query, n_results=n_results, crop_filter=None)

    return {
        "query": query,
        "results": [
            {
                "text": doc,
                "source_title": meta.get("title"),
                "source_url": meta.get("source_url"),
                "relevance_distance": dist,
            }
            for doc, meta, dist in zip(docs, metas, distances)
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
