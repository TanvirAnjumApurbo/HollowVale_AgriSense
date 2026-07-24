# AgriSense AI

An agentic advisor that takes a farmer from a short conversation to a grounded, explained, costed season plan -- built for the IUT ICT Fest Agentic AI Hackathon (sponsored by bdapps, powered by Codex).

> Submission note: rename this repository to `TeamName-AgriSense` (with your actual team name) before submitting, per the hackathon's naming requirement.

## What this is

The agent gathers a farmer's situation (location, farm size, soil type, water availability, budget, target season) through conversation, pulls a real live weather forecast, ranks candidate crops with a deterministic financial model grounded in a retrieved agronomic knowledge base, and produces a dated, costed season plan with every recommendation tied back to the specific data behind it.

## Tier reached

**Tier 0 (core), complete end to end:**

1. Conversational intake with targeted follow-ups for missing fields only.
2. Live weather grounding via a real API (Open-Meteo), used verbatim in recommendations.
3. Crop recommendation: a deterministic multi-factor scoring engine (`tools/agronomy.py`) ranks 13 candidate crops on soil fit, season/plantability, water fit vs the live forecast, temperature, and profit/ROI -- returning per-crop component scores, a budget flag (with max affordable area), and quotable reasons that each name the exact input used. It is **date-aware**: asked on 24 July it surfaces Aman rice (whose Kharif-2 window is open now), not Boro (a November-December Rabi crop), because season logic that actually bites is more convincing than profit-sorting.
4. Deterministic dated season plan (`tools/season_plan.py`): an explicit sowing date plus stored day offsets produces one sourced chronological calendar from land preparation through harvest, with event costs reconciled to the financial projection.
5. Financial projection: itemized costs, yield, revenue, net profit, ROI, break-even -- computed deterministically, not by the LLM.
6. Explained reasoning: every agent-facing tool returns a structured `reasons` array. The prompt requires each recommendation to quote a producing-tool reason verbatim and forbids unsupported claims or reworded numbers.
7. Knowledge base with RAG: real public agronomic sources, chunked and embedded locally, retrieved and grounding every crop/fertilizer/season-plan answer.
8. Visible agent trace: a sidebar panel logs every tool call, its arguments, and its raw returned value.

**Tier 1 now implemented:** proactive weather-triggered nitrogen scheduling plus session-coherent facts. If Open-Meteo shows more than 10 mm total rain within 48 hours of a rain-sensitive, non-basal application, the calendar moves it to the first supplied safe forecast day under 5 mm and records the original date, forecast amount, and reason. Successful weather, ranking, chosen crop, projection, and calendar outputs persist for the Streamlit session and are injected into later prompts as a compact digest with verbatim quotable reasons. Acreage-only follow-ups recompute the dependent projection/calendar while reusing the established forecast, so the same farm cannot silently receive different weather numbers.

## What's real vs mock

| Data | Real or mock |
|---|---|
| Weather (rainfall, temperature forecast) | **Real** -- live call to Open-Meteo (forecast + geocoding APIs). The open-access, non-commercial prototype endpoint needs no account/key; commercial customer capacity uses a paid API key. |
| Agronomic knowledge base (fertilizer doses, sowing windows, irrigation schedules, pest risk, soil types) | **Real** -- sourced from public institutional documents (see `data/raw/*.md`, each with a `Source:` line): Bangladesh Agro-Meteorological Information Service (BAMIS)/Department of Agricultural Extension (DAE) package-and-practices pages, Bangladesh Rice Research Institute (BRRI), Bangladesh Agricultural Research Institute (BARI), and Bangladesh Jute Research Institute (BJRI) guidance covering all 13 crops (Boro/Aman/Aus rice, wheat, maize, potato, lentil, jute, mustard, onion, chili, tomato, chickpea), plus Banglapedia's soil-type and agro-ecological-zone reference. The canonical **BARC Fertilizer Recommendation Guide 2024 (FRG-2024)** is now included as five curated, sourced documents distilled from the official chapters -- per-crop fertilizer doses, fertilizer types and use, agro-ecological-zone soil fertility, climate-smart soil management, and the national crop calendar. |
| Fertilizer cost in the financial calculator | **Derived (inspectable)** -- summed from each crop's real dose schedule in `data/crops.yaml` priced by `data/input_prices.yaml` (`fertilizer cost = sum(kg/acre x input price)`), not a flat guess. Source doses are unit-converted correctly (BAMIS kg/bigha x3 -> kg/acre; BARC/BARI kg/ha / 2.471 -> kg/acre), which the doc-table numbers can be audited against. |
| Crop yield and market price figures | **Estimated** -- ballpark per-acre yield and price figures for a hackathon demo, not pulled from a live market feed. Clearly labeled as such in every financial projection returned by the tool (`data_source_note` field). `data/crops.yaml` + `data/input_prices.yaml` are the single source of truth shared by the financial and agronomy engines. |
| LLM | **Real** -- OpenAI `gpt-4o-mini` via the OpenAI API, used for conversation, tool-call planning, and explanation only (never for arithmetic). |

## Architecture

- **`app.py`** -- Streamlit chat UI + a sidebar trace panel. Single process, no separate backend server needed for Tier 0 (or for Tier 2's bdapps CaaS call, which is a synchronous outbound REST call -- see below).
- **`agent/`** -- the agent itself:
  - `llm.py` -- provider-agnostic `chat()` wrapper (OpenAI `gpt-4o-mini` by default; swappable to Groq/other via `LLM_PROVIDER`).
  - `prompts.py` -- system prompt construction, required-field tracking, compact established-fact digest, exact-reason quoting rules, and the intake-to-calendar workflow.
  - `orchestrator.py` -- a hand-rolled tool-calling loop (no agent framework). Every tool call, its arguments, and its raw result are captured into a trace log. Successful decision facts are also persisted, and same-location weather requests are served from the established session fact instead of refetching unless the farmer explicitly requests a refresh.
- **`tools/`** -- the agent's real capabilities:
  - `weather.py` -- Open-Meteo geocoding + forecast.
  - `knowledge_base.py` -- RAG retrieval over a local Chroma vector store.
  - `agronomy.py` -- deterministic multi-factor crop scoring (`score_crop` / `rank_crops`): soil / season / water / temp / profit components, an overall weighted score, a budget flag, and a `reasons` list emitted **as data** so the LLM narrates the reasoning rather than inventing it.
  - `financials.py` -- deterministic cost/yield/ROI/break-even calculator (plain Python, not LLM arithmetic).
  - `season_plan.py` -- merges crop stages, fertilizer applications, irrigation checkpoints and pest windows into real ISO dates, adds seed/labour/harvest events, allocates every cost category without cent drift, and applies forecast-backed nitrogen shifts without inventing replacement dates.
- **`data/`** -- a deliberately **two-layer** knowledge design:
  - `crops.yaml` + `input_prices.yaml` -- the machine-readable layer (Layer A) that drives the deterministic tools: per-crop seasons, sowing windows, soil suitability, water/temperature needs, dose schedules, pest windows and economics. `crop_loader.py` loads it and derives per-acre costs. This is the single source of truth for both `agronomy.py` and `financials.py`.
  - `raw/*.md` -- the prose corpus (Layer B) that drives RAG + citations, one sourced document per crop/topic with a `Source:` line.
  - `ingest.py` chunks and embeds the corpus locally (Chroma's bundled ONNX `all-MiniLM-L6-v2` runtime -- no embedding API call and no PyTorch) into `chroma_db/`. The index is gitignored and rebuilt automatically on first use if missing, so a fresh clone is self-sufficient.
- **`tests/`** -- schema/cost, retrieval, season-calendar/weather-adjustment, and orchestrator-memory regression suites. The scripted conversation tests run fully offline and verify fact persistence, structured reasons, area-scenario consistency, and no weather network refetch.
- **`memory/session_store.py`** -- conversation, profile, trace, and established facts (`weather`, `ranking`, `chosen_crop`, `projection`, `calendar`) for the current session. It remains the seam for later durable SQLite/Postgres storage without changing the orchestrator or UI.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv\Scripts\activate on cmd/PowerShell
pip install -r requirements.txt

cp .env.example .env               # then fill in your real OPENAI_API_KEY
python data/ingest.py              # builds the local knowledge base index (data/chroma_db/)

streamlit run app.py
```

For Streamlit Community Cloud deployment, set `OPENAI_API_KEY` (and optionally `LLM_PROVIDER`/`LLM_MODEL`) in the app's Secrets manager using `.streamlit/secrets.toml.example` as a template.

## Tools and APIs used

- **[Open-Meteo](https://open-meteo.com/en/docs)** (geocoding + forecast APIs) -- the open-access endpoint used here needs no account/key for non-commercial evaluation/prototyping. [Paid commercial plans](https://open-meteo.com/en/pricing) use an API key and the dedicated `customer-api.open-meteo.com` endpoint; [attribution is required](https://open-meteo.com/en/license) and is displayed in the app sidebar.
- **OpenAI API** (`gpt-4o-mini`) -- requires `OPENAI_API_KEY`.
- **ChromaDB** -- local, in-process vector store, no external service.
- **ChromaDB ONNX embeddings** (`all-MiniLM-L6-v2`) -- local embeddings via Chroma's bundled ONNX runtime (no PyTorch), no API call, works offline once the model is downloaded.

## Verification

- `pip install -r requirements-dev.txt` then `pytest tests/ -q` -- the dataset + retrieval gate (crop schema, schedule-derived cost consistency, 37 RAG queries each landing the right source doc in the top-3, and whole-word crop-detection checks).
- `python -m tools.agronomy` -- self-checks the scoring engine: flipping soil (clay vs sandy) changes the top crop, flipping season (Kharif-2 vs Rabi) changes the top crop, halving the budget flips the affordability flag and the max affordable area, and every ranked crop carries >=4 reasons.
- `python -m tools.financials` -- confirms costs scale linearly with area, yield adjustments move profit correctly, aliases resolve (`rice`->`rice_boro`), and fertilizer cost is the schedule-derived figure.
- `pytest tests/test_season_plan.py -q` -- confirms exact harvest offsets, chronological/source-complete events, synthetic 40 mm rain rescheduling, strict wet/dry thresholds, no fabricated dry date, and event-to-projection cost reconciliation.
- `pytest tests/test_orchestrator_memory.py -q` -- scripts a full multi-turn tool chain and verifies established-fact injection, top-level reasons, scenario recomputation, trace-backed reply numbers, and weather-cache reuse.
- `python tools/weather.py` -- confirms real geocoding + forecast values return for sample Bangladesh locations.
- `python data/ingest.py` then `python tools/knowledge_base.py` -- confirms retrieval returns topically relevant, correctly-sourced chunks.
- `streamlit run app.py` -- full conversational walkthrough from a vague opener to a costed season plan, with every tool call visible in the sidebar trace.

## Where later upgrades hook in

- **Durable cross-session memory** -- swap the implemented `memory/session_store.py` session-fact backend to SQLite (or Supabase Postgres), keeping the same interface.
- **Next agronomy upgrades** -- irrigation rescheduling, pest/disease risk triggers, and broader scenario simulation can extend the same deterministic calendar/tool pattern.
- **bdapps CaaS payment integration** -- per the official `bdapps-API-DGD` guide, the Direct Debit charge call is a synchronous, merchant-initiated REST call (`POST https://developer.bdapps.com/caas/direct/debit`) with the transaction result returned in the same HTTP response -- no inbound webhook needed, so it drops straight into this same Streamlit app as another outbound tool call once `applicationId`/`password` are provisioned via the bdapps developer portal.
