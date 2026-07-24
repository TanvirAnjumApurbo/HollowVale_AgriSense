# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AgriSense AI is an agentic advisor (IUT ICT Fest hackathon) that takes a Bangladeshi farmer from a vague opening message to a grounded, explained, costed season plan. It is a **single-process Streamlit app** with a **hand-rolled tool-calling loop** (deliberately no LangChain/CrewAI) so every tool call is transparent for the "visible agent trace" requirement.

## Commands

Environment is Windows (PowerShell primary; Git Bash also available). No build step — it's a Python app.

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate                 # PowerShell/cmd; use `source .venv/Scripts/activate` in Git Bash
pip install -r requirements.txt
cp .env.example .env                    # then set OPENAI_API_KEY (Copy-Item in PowerShell)

# Run the app
streamlit run app.py

# Build the RAG index (optional — knowledge_base self-heals and builds it on first query)
python data/ingest.py                   # re-run after editing any data/raw/*.md

# Tests (pytest is dev-only, kept out of requirements.txt)
pip install -r requirements-dev.txt
pytest tests/ -q                                                   # full suite
pytest tests/test_crops.py -q                                      # crop schema + cost-consistency only
pytest "tests/test_crops.py::test_projection_cost_is_schedule_derived[potato]" -q   # a single test/param

# Deterministic self-checks — runnable modules, no pytest needed
python -m tools.agronomy                # scoring engine: soil/season flip the ranking, budget flag flips, reasons present
python -m tools.financials              # cost scales with area, aliases resolve, fertilizer cost is schedule-derived
python -m data.crop_loader              # dump all crops + derived per-acre cost breakdowns
python tools/weather.py                 # live Open-Meteo geocode+forecast (hits network)
```

Note: the retrieval tests query Chroma, which triggers a one-time ONNX model download/load (~30–60s cold) the first time in a fresh environment.

## Core principle: the LLM narrates, tools decide

The single most important constraint: **the model never does arithmetic or suitability judgement.** All numbers (weather, cost, yield, ROI, break-even) and all crop rankings come from tools; the LLM's job is to gather intake, chain the tool calls, and explain the results. `tools/agronomy.py` even emits its reasoning as a `reasons` list of strings (each naming the exact input value used) specifically so the model quotes them rather than inventing its own rationale. When adding features, keep computation in tools and preserve this separation.

## The agent loop (`agent/orchestrator.py::run_turn`)

- State is passed **by reference** from `st.session_state`: `conversation_history`, `farmer_profile`, and `trace_log` are mutated in place, which is how the UI reflects updates. `run_turn(conversation_history, user_message, farmer_profile, trace_log)`.
- Each iteration: `messages = [system(farmer_profile)] + history + [user]`, call `chat()`, execute any tool calls, append results, then **rebuild the system prompt** from the (possibly updated) profile so intake tracking stays current mid-turn.
- A per-turn `context = {farmer_profile, last_weather}` is threaded into `_execute_tool`. `get_weather` caches its real result into `context`; `rank_crops` reads the live profile + cached forecast from `context` — **not** from LLM-supplied args — so scoring uses injected real data, not numbers the model might fudge.
- On the final iteration (`MAX_TOOL_ITERATIONS`) tools are withheld (`tools=None`) to force a text answer instead of dead-ending on the cap.
- On success it appends user+assistant to `conversation_history`. On exception it raises *before* appending; `app.py` catches that and records the failed turn (message + error) into history and trace, so failures are visible rather than silently lost on `st.rerun()`.

## Two-layer knowledge design (the key architectural split)

Knowledge is deliberately split so the recommendation is deterministic:

- **Layer A — machine-readable, drives the tools.** `data/crops.yaml` (13 crops: `rice_boro`, `rice_aman`, `rice_aus`, `wheat`, `maize`, `potato`, `lentil`, `jute`, `mustard`, `onion`, `chili`, `tomato`, `chickpea`) + `data/input_prices.yaml`, loaded via `data/crop_loader.py`. This is the **single source of truth** shared by both `tools/financials.py` and `tools/agronomy.py`. Fertilizer cost is **derived** (`sum(kg/acre × input_price)` over each crop's dose schedule), never a flat guess.
- **Layer B — prose corpus, drives RAG + citations.** `data/raw/*.md`, one sourced document per crop/topic, each with a `Source:` line that `data/ingest.py` parses into metadata. `tools/knowledge_base.py` retrieves from the local Chroma store and returns `source_file`/`source_title`/`source_url` so the agent can cite.

**Unit conversions are load-bearing** (get them wrong and every fertilizer number is off): BAMIS source tables are **kg/bigha → ×3 for kg/acre** (rice); BARC/BARI tables are **kg/ha → ÷2.471 for kg/acre** (everything else). Doses in `crops.yaml` are already per-acre material amounts.

**Crop-key duality:** Layer A uses fine keys (`rice_boro`, `rice_aman`) with aliases in `crop_loader.py` (`rice`→`rice_boro`, `aman`→`rice_aman`, `masur`→`lentil`, …). Layer B tags chunks with coarse crop *families* (`rice` covers both boro and aman) because RAG filtering is family-level. The KB alias maps live in **both** `data/ingest.py` and `tools/knowledge_base.py` and must stay in sync.

## Scoring engine specifics (`tools/agronomy.py`)

- `score_crop` returns component scores (soil .25, season .25, water .20, temp .10, profit .20; **budget is a flag, not weighted**), an overall weighted score, and the `reasons` list. `rank_crops` scores all 8 and sorts.
- **Date/season awareness is the demo's differentiator.** `infer_season` (Bangladesh Kharif-1 / Kharif-2 / Rabi calendar) and `window_timing` mean that asked on a Kharif-2 date it ranks Aman rice first (window open) and scores Boro low (Rabi window months away). Don't flatten this into profit-sorting.
- **Profit is scored against a fixed strong-ROI benchmark (`PROFIT_REF_PCT`), not min-max across candidates** — this is intentional. Min-max lets one low-cost pulse (lentil, ~450% ROI) peg at 1.0 and swamp the soil weight, so soil/season could never change the ranking. If you touch profit scoring, preserve the property that flipping soil or season flips the top crop (the `python -m tools.agronomy` self-check asserts exactly this).

## Adding or editing a crop

1. Add/edit the entry in `data/crops.yaml` with **all** required keys (the schema test enumerates them; `soil_suitability` needs all five texture classes; every `fertilizer_schedule.input` must exist in `input_prices.yaml`).
2. Add a `data/raw/<crop>.md` prose doc (with a `Source:` line) covering sowing/fertilizer/irrigation/pest/harvest/soil so RAG can ground and cite it.
3. If it's a new crop family, register aliases in `data/ingest.py` and `tools/knowledge_base.py`.
4. `python data/ingest.py` to rebuild the index, then `pytest tests/ -q`.

## Other conventions

- `agent/llm.py::chat()` is the only LLM entry point (OpenAI `gpt-4o-mini` default; swap via `LLM_PROVIDER`/`LLM_MODEL`). Secrets resolve **Streamlit secrets → env var** via `_get_secret`, so the app runs both on Streamlit Cloud and from a local `.env`.
- `memory/session_store.py` is the state seam for future persistence — keep its function signatures stable so swapping `st.session_state` for SQLite/Postgres needs no changes in `app.py`/`orchestrator.py`. (Note: repo `memory/` is a Python package, unrelated to any Claude memory.)
- Embeddings are Chroma's bundled ONNX `all-MiniLM-L6-v2` (no PyTorch). The same embedding function must be used at ingest and query time.
- **Never commit `data/chroma_db/`** — it's gitignored and rebuilds from `data/raw`; a partial index is worse than none.
- Commits in this repo **omit the `Co-Authored-By` trailer**.
