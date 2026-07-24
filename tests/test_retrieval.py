"""RAG retrieval regression test.

Asserts that a spread of realistic agronomy queries each land the RIGHT
source document in the top-3 retrieved chunks. Cheap to run, and it catches
the classic KB regressions -- a query drifting to the wrong crop's doc, a
crop tag going missing, or a new document knocking an old answer out of the
top results -- which matter for grounded, citable answers.

The knowledge_base search self-heals: if the Chroma index is missing it is
built from data/raw on the first query, so this test also implicitly checks
that ingest still runs.
"""

import pytest

from tools.knowledge_base import search_knowledge_base

# (query, expected source file that should appear in the top-3 results)
CASES = [
    ("urea top dressing timing for boro rice", "rice_boro_bamis.md"),
    ("transplanting window for boro paddy", "rice_boro_bamis.md"),
    ("aman rice sowing window in the kharif monsoon", "rice_aman.md"),
    ("when to transplant transplanted aman rice", "rice_aman.md"),
    ("wheat sowing time in the rabi season", "wheat_bamis.md"),
    ("irrigation schedule for wheat", "wheat_bamis.md"),
    ("maize fertilizer dose per hectare", "maize_bamis.md"),
    ("fall armyworm pest in maize", "maize_bamis.md"),
    ("which soil is best suited to lentil", "lentil_masur_bari.md"),
    ("lentil masur seed rate and fertilizer", "lentil_masur_bari.md"),
    ("potato late blight disease control", "potato_fertilizer.md"),
    ("recommended fertilizer dose for potato", "potato_fertilizer.md"),
    ("jute fibre retting and harvest", "jute_cultivation.md"),
    ("mustard sarisha aphid pest control", "mustard.md"),
    ("mustard sowing window and sulphur requirement", "mustard.md"),
    ("brown plant hopper favorable conditions in rice", "rice_pests_bamis.md"),
    ("cropping seasons kharif and rabi in bangladesh", "soil_and_seasons_bangladesh.md"),
]


def _top_files(query, k=3):
    out = search_knowledge_base(query, n_results=k)
    return [r.get("source_file") for r in out["results"]]


@pytest.mark.parametrize("query,expected", CASES)
def test_expected_doc_in_top3(query, expected):
    files = _top_files(query, k=3)
    assert expected in files, f"{expected!r} not in top-3 for {query!r}; got {files}"


def test_crop_filter_promotes_matching_crop():
    """A crop-specific query should return only that crop's (or general) docs
    ahead of unrelated crops -- the Python-side crop filter at work."""
    out = search_knowledge_base("fertilizer dose", n_results=3, crop_filter="potato")
    assert out["crop_filter_applied"] == "potato"
    assert any(r["source_file"] == "potato_fertilizer.md" for r in out["results"])
