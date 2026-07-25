# AgriSense AI

An agentic advisor that takes a Bangladeshi smallholder farmer from a short, vague conversation to a grounded, explained, costed season plan — and keeps advising through the season. Built for the **IUT ICT Fest Agentic AI Hackathon** (sponsored by bdapps, powered by Codex).

**Team:** HollowVale

## What it does

From a vague opener ("I want to grow something this year"), the agent:

1. holds a conversation to collect the farm's specifics — location, farm size, soil type, water availability, budget — asking targeted follow-ups **only for the fields still missing**;
2. pulls a **live** weather forecast for the farm's location and uses the returned rainfall/temperature verbatim;
3. ranks candidate crops with a deterministic, date-aware scoring engine, grounded in a retrieved agronomic knowledge base;
4. builds a dated, costed season plan for the chosen crop, from land preparation to harvest; and
5. explains every recommendation with the exact farm inputs and retrieved evidence behind it — with every tool call, its arguments, and its raw return value visible in a trace panel.

The governing principle is **"the LLM narrates; tools decide."** Every number (weather, cost, yield, ROI, break-even, risk) and every crop ranking comes from deterministic Python tools that emit their reasoning **as data**; the model gathers intake, chains the tool calls, and explains the results. It never does arithmetic or invents a figure — and the system prompt forbids stating any claim that no tool `reasons` entry supports.

## Which tier each feature reaches

### Tier 0 — Core (complete, runs end to end)

| # | Capability | How it is met |
|---|---|---|
| 1 | **Conversational intake** | Tracks 5 required fields (location, farm size, soil, water, budget) and asks only for the ones still missing; the target season is derived deterministically from today's date (`infer_season`) and stated to the farmer, who can override it. `agent/prompts.py`, `agent/orchestrator.py`. |
| 2 | **Live weather grounding** | Real call to Open-Meteo (geocoding + forecast). The ranking and calendar consume the **injected** forecast from application state — the tool schema exposes no field for the model to pass its own numbers. `tools/weather.py`. |
| 3 | **Crop recommendation** | A deterministic multi-factor engine ranks 13 candidate crops, each returned with **suitability** (weighted soil/season/water/temp/profit score), **water need** (low/med/high), a **risk level** (Low/Med/High), and a **rough profit/ROI** estimate — every value backed by a quotable reason. `tools/agronomy.py`. |
| 4 | **Season plan** | A dated calendar from land preparation through harvest: sowing/transplanting window, fertilizer timing, irrigation, **weed-control and pest-scouting checkpoints**, and harvest — event costs reconciled to the financial projection to the cent. `tools/season_plan.py`. |
| 5 | **Financial projection** | Itemized cost breakdown + expected yield, revenue, net profit, ROI, and break-even, computed in plain Python. Internally consistent: change area/price/yield and every downstream number moves correctly. `tools/financials.py`. |
| 6 | **Explained reasoning** | Every tool returns a structured `reasons` array naming the exact input used; the prompt requires each claim to quote a producing-tool reason verbatim and forbids reworded numbers or unsupported claims. |
| 7 | **Knowledge base + RAG** | Real public agronomic sources chunked and embedded locally into Chroma; retrieval grounds crop/fertilizer/season advice. The orchestrator **deterministically** retrieves KB evidence for each top-3 finalist and folds the citations into the ranking, so grounding does not depend on the model choosing to search. `tools/knowledge_base.py`, `data/ingest.py`. |
| 8 | **Visible agent trace** | A per-turn expandable panel logs every tool call — name, arguments sent, and **raw returned JSON** — persisted alongside each assistant message so a judge can confirm any number came from a real call. `app.py`. |

### Tier 1 — Advanced (implemented)

- **Fertilizer and irrigation scheduler, with an organic alternative.** Every fertilizer split (material, kg/acre, cost) and every irrigation checkpoint is dated from the crop's own offsets in `data/crops.yaml`, and each event's cost is allocated from the financial projection so the calendar reconciles to it. Alongside the chemical plan the calendar returns an `organic_alternative` block: the crop's scheduled nitrogen derived from its own dose schedule, the safe 25% share of it re-supplied as cow dung/compost at 0.5% N, the urea it displaces, the P2O5/K2O the manure also supplies, and **both** cash cases stated honestly — farm-supplied (the usual case, where the only cash movement is buying less urea) and bought at the list price (dearer than the urea it replaces; worth it for soil organic matter, not as a way to cut cash cost). It is an option shown next to the dates it would replace, not a change to the costed plan.
- **Proactive, weather-triggered scheduling.** If Open-Meteo shows more than 10 mm total rain within 48 hours of a rain-sensitive, non-basal nitrogen application, the calendar moves it to the first supplied forecast day under 5 mm and records the original date, forecast amount, and reason — it never invents a replacement date.
- **Pest and disease risk, scaled by the live forecast.** Each ranked crop carries a Low/Medium/High tier blended from five named 0–1 drivers (water stress, pest/disease pressure, off-window timing, temperature stress, affordability). The pest driver starts from the count of the crop's tracked pest windows and is then scaled by a bounded multiplier read off the same Open-Meteo forecast — 0.85 on a dry forecast, up to 1.30 when rainfall runs ≥8 mm/day and the average max temperature sits inside the 25–35°C band. With no forecast the multiplier stays 1.00 and the reason says pressure is "unmodified": it never invents pressure. The tier ships the actionable detail too — `risk.pests` lists each tracked pest with its growth-stage day window, the sign to scout for, the control, and its per-acre cost, all from `crops.yaml`. Risk is computed **after and separately from** `overall_score`, so this weather scaling never re-ranks the crops; tests pin both halves of that.
- **Scenario simulation.** "What if rainfall cuts yield 30%?" or "what if the sale price changes?" recompute the projection through `yield_adjustment_pct` / `price_override`, returning changed numbers rather than a generic answer.
- **Persistent cross-session memory + accounts.** With a database configured, user accounts (PBKDF2-hashed passwords, hashed session tokens) and **Neon PostgreSQL**-backed conversation persistence let a farmer reopen a prior conversation with the full transcript, trace, profile, and established facts restored. DB-backed per-user/global daily rate limiting protects the API budget. `memory/`.

### Tier 2 — Bonus (implemented, simulator mode)

- **bdapps CaaS payment gateway.** A separate FastAPI sidecar (`server.py`, `bdapps/`) runs the full Charging-as-a-Service checkout → response → callback → receipt flow with a SQLite ledger and simulated operator-balance deduction. It runs against a **deterministic local simulator by default** (`BDAPPS_SIMULATE=true`) that mirrors the documented `S1000` response envelope — no credentials, no network, no real money. The checkout panel shows the **exact CaaS request/response pair** behind each charge (app password masked), so the exchange is verifiable rather than claimed — the same principle as the agent trace. The charge amount is quoted from `compute_financial_projection` via `/bdapps/quote`, so money is tool-derived on the payment side too. See `BDAPPS.md` for the run-through and the human-only portal steps.

## What's real vs. what's mock

| Component | Status |
|---|---|
| Weather (rainfall, temperature forecast) | **Real** — live Open-Meteo forecast + geocoding APIs. The open-access, non-commercial endpoint needs no account/key (commercial capacity uses a paid key); attribution is shown in the app. |
| Agronomic knowledge base (fertilizer doses, sowing windows, irrigation, pests, soils) | **Real** — sourced from public institutional documents in `data/raw/*.md`, each carrying a `Source:` line: BAMIS/DAE package-&-practice pages, BRRI, BARI, BJRI, the **BARC Fertilizer Recommendation Guide 2024**, the national crop calendar, and Banglapedia soil/AEZ references, covering all 13 crops. |
| Crop suitability ranking | **Real logic** — deterministic scoring over the real crop data and the live forecast; date-aware (asked in late July it surfaces Aman rice, whose Kharif-2 window is open, not Boro). |
| Fertilizer cost | **Derived (inspectable)** — `sum(kg/acre × input price)` from each crop's real dose schedule in `data/crops.yaml` priced by `data/input_prices.yaml`, with correct unit conversions — not a flat guess. |
| Crop yield & market price figures | **Estimated** — ballpark per-acre yield/price for a demo, not a live market feed. Labelled as such in every projection (`data_source_note`). |
| Crop **risk level** | **Derived (heuristic)** — a Low/Med/High tier computed from real signals (the crop's tracked pest windows, **scaled by the live forecast's rainfall-per-day and temperature band**, plus forecast-driven water and temperature shortfall and budget affordability). Not a live pest/disease-forecast feed: the forecast modulates the crop's own tracked pest load and never invents pressure — no forecast means no scaling. |
| Season-plan **weed checkpoint** | **Derived (heuristic)** — a generic critical-weed-competition window (~15–40% of the crop cycle); `crops.yaml` has no per-crop weeding offsets. Fertilizer/irrigation/pest/stage dates are from the sourced crop data; the nitrogen shift uses the real forecast. |
| Season-plan **organic alternative** | **Derived (inspectable)** — the manure quantity is back-calculated from the crop's own dose schedule in `data/crops.yaml` and priced by `data/input_prices.yaml`, and both cash cases (farm-supplied vs. purchased) are reported. The nutrient percentages behind that conversion (urea 46% N, cow dung 0.5% N) are conventional planning figures, not lab values for a particular farmer's heap. |
| LLM | **Real** — OpenAI `gpt-5` via the OpenAI API, for conversation, tool-call planning, and explanation only (never arithmetic). |
| Accounts + conversation persistence | **Real when `DATABASE_URL` (Neon) is set**; degrades gracefully to guest / single-session mode when no database is configured. |
| bdapps CaaS **operator charge** | **Simulated by default** — the request/response/callback/receipt flow, MSISDN validation, ledger, and balance deduction are real; the charge itself deducts from a seeded local balance. The live API is a single documented seam (`bdapps/client.py`), off unless explicitly enabled. |

## Architecture

- **`app.py`** — Streamlit chat UI, login gate, and the per-turn trace panel.
- **`agent/`** — the agent itself:
  - `llm.py` — provider-agnostic `chat()` wrapper (OpenAI `gpt-5` default; swappable via `LLM_PROVIDER`/`LLM_MODEL`). Secrets resolve Streamlit-secrets → env var.
  - `prompts.py` — system-prompt construction, required-field tracking, the compact quotable established-fact digest, and the intake-to-calendar workflow with exact-reason-quoting rules.
  - `orchestrator.py` — a **hand-rolled** tool-calling loop (no LangChain/CrewAI, deliberately, so every step is transparent). Captures every tool call into the trace, persists decision facts, serves same-location weather from cache, and deterministically grounds each finalist in the KB.
- **`tools/`** — `weather.py` (Open-Meteo), `knowledge_base.py` (Chroma RAG), `agronomy.py` (crop scoring + risk), `financials.py` (cost/yield/ROI/break-even), `season_plan.py` (dated, cost-reconciled calendar + weather-driven nitrogen shift).
- **`data/`** — a two-layer knowledge design: **Layer A** (`crops.yaml` + `input_prices.yaml`) is the machine-readable single source of truth for the tools; **Layer B** (`raw/*.md`) is the sourced prose corpus for RAG + citations. `ingest.py` embeds it locally into `chroma_db/` (gitignored; rebuilt on first use).
- **`memory/`** — `db.py` (Neon Postgres pool + schema), `conversations.py` (per-user transcripts/traces/state as JSONB), `auth.py` (accounts + sessions), `rate_limit.py` (usage limits), `session_store.py` (in-session facts). All degrade gracefully without a database.
- **`server.py` + `bdapps/`** — the independently deployable Tier-2 CaaS sidecar (Render blueprint in `render.yaml`, lean `requirements-sidecar.txt`).
- **`tests/`** — schema/cost (including the organic substitution), RAG retrieval, season-calendar/weather-adjustment, forecast-scaled pest risk (`test_agronomy_risk.py`), orchestrator-memory, persistence/auth, and bdapps suites. The scripted conversation tests run fully offline.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # PowerShell/cmd; source .venv/Scripts/activate in Git Bash
pip install -r requirements.txt

cp .env.example .env                    # then set OPENAI_API_KEY (Copy-Item in PowerShell)
python data/ingest.py                   # builds the local RAG index (optional — self-heals on first query)

streamlit run app.py
```

Optional:

```bash
# Tier-2 bdapps CaaS payment sidecar (simulator by default — no creds/network)
pip install -r requirements-sidecar.txt
uvicorn server:app --host 0.0.0.0 --port 8000

# Cross-session persistence + accounts: set DATABASE_URL to a Neon Postgres URL
# (Streamlit secrets or .env). Without it, the app runs in guest/single-session mode.
```

For Streamlit Community Cloud, set `OPENAI_API_KEY` (and optionally `DATABASE_URL`, `LLM_PROVIDER`/`LLM_MODEL`) in the app's Secrets manager.

## Tools & APIs used

- **[Open-Meteo](https://open-meteo.com/en/docs)** (geocoding + forecast) — open-access endpoint, no key for non-commercial evaluation; attribution shown in-app.
- **OpenAI API** (`gpt-5`) — requires `OPENAI_API_KEY`.
- **ChromaDB** + its bundled **ONNX `all-MiniLM-L6-v2`** embeddings — local, in-process, no external service and no PyTorch.
- **Neon PostgreSQL** (optional) — accounts, conversation persistence, and rate limiting via `DATABASE_URL`.
- **bdapps CaaS (TAP)** — Tier-2 payment integration, simulator by default; live path documented in `BDAPPS.md`.
- **FastAPI + Uvicorn** — the bdapps sidecar.

## Verification

```bash
pip install -r requirements-dev.txt
pytest tests/ -q                        # full suite — 220 tests
```

Deterministic self-checks (no pytest needed):

- `python -m tools.agronomy` — soil/season each flip the top crop, the budget flag flips with budget, and every ranked crop carries a risk tier, a water-need label, and ≥4 reasons.
- `python -m tools.financials` — cost scales linearly with area, yield/price scenarios move profit correctly, and fertilizer cost equals the schedule-derived figure.
- `python -m tools.season_plan` — a dated Boro calendar (with a synthetic rain-driven urea shift) whose events reconcile to the financial projection.
- `python tools/weather.py` — live geocoding + forecast for sample Bangladesh locations.
- `python data/ingest.py && python tools/knowledge_base.py` — retrieval returns topically relevant, correctly-sourced chunks.

## Notes & limitations

- Yield and market-price figures are demo estimates (labelled in every projection), not a live market feed — the natural next upgrade is a real price source.
- Risk tiers and the weed-control window are transparent heuristics over the real data. Pest/disease pressure is scaled by the live rainfall and temperature forecast, but that is a bounded weather-response rule over the crop's tracked pest windows — not a dedicated pest/disease-forecast model or a field-scouting feed.
- Manure nutrient content varies a lot in practice (animal, bedding, moisture, how long the heap was left to decompose), so the organic alternative's 0.5% N planning figure makes its cow-dung quantity a planning guide rather than a precise dose.
- Durable memory and accounts require a Neon `DATABASE_URL`; without one the app still runs fully, single-session, as a guest.
