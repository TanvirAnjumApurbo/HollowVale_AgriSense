# bdapps CaaS integration (Charging-as-a-Service)

This is the Tier‑2 **bdapps Payment Gateway** feature (`hackathon_context.md` line 78; 10 pts).
It runs as a small **FastAPI sidecar** (`server.py`) next to the Streamlit app,
because Streamlit cannot serve the inbound HTTP callback that bdapps POSTs to a
registered **host address**. The sidecar owns the charge call, the callback
host, and the receipt; Streamlit stays the demo/trace UI and triggers checkout
through it.

By default everything runs against a **deterministic local simulator**
(`BDAPPS_SIMULATE=true`) — no credentials, no network, no real money — so the
sandbox demo works offline. Flipping to the live bdapps API is a one‑function
change (`bdapps/client.py::_charge_via_bdapps`).

## Architecture

```
Streamlit UI (app.py) ──"Charge via bdapps"──► POST /bdapps/checkout ─┐
                                                                      │
  bdapps portal ──async charge notification──► POST /bdapps/notify    │  server.py
                                                                      │  (FastAPI
  farmer / judge ──────────────────────────► GET  /bdapps/receipt/…  ┘   sidecar)
                                                        │
                    bdapps/client.py  ◄──── charge() ───┤  simulate ► SQLite balance
                    (real call OR simulator)            │  live ► real bdapps API
                    bdapps/store.py  (SQLite: transactions + sim balances)
```

- `bdapps/config.py` — settings from env (read live, so tests can override).
- `bdapps/store.py` — SQLite transaction ledger + simulated operator balances, keyed by transaction reference. Persistence seam (swap for Postgres here).
- `bdapps/client.py` — `charge()`; the **only** place the real bdapps wire format lives is `_charge_via_bdapps`.
- `server.py` — the FastAPI app and its routes.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/bdapps/checkout` | App → bdapps charge. Body: `{msisdn, amount, currency?, description?, reference?}`. |
| POST | `/bdapps/notify` | **The host address you register.** bdapps → you (async result). Returns the `S1000` ack. |
| GET | `/bdapps/receipt/{ref}` | HTML receipt (or `?format=json`). |
| GET | `/bdapps/quote?crop=&area_acres=&item=advisory\|inputs` | Suggested amount from the financial‑projection tool. |
| GET | `/bdapps/balance/{msisdn}` | Simulated operator balance. |
| GET | `/bdapps/transactions` | Recent transactions (debug/demo). |
| GET | `/healthz` | Health + mode. |

## Run it locally (simulator — the demo path)

```bash
# 1. Sidecar (terminal 1)
uvicorn server:app --host 0.0.0.0 --port 8000
#   -> http://localhost:8000/healthz  ->  {"status":"ok","simulate":true,...}

# 2. Streamlit UI (terminal 2)
streamlit run app.py
#   Sidebar -> "💳 Pay via bdapps (CaaS sandbox)" -> Charge -> receipt link.
```

Smoke test without the UI:

```bash
curl -X POST http://localhost:8000/bdapps/checkout \
  -H "Content-Type: application/json" \
  -d '{"msisdn":"01712345678","amount":20,"description":"AgriSense advisory"}'
```

Tests: `pytest tests/test_bdapps.py -q` (22 tests, simulator only).

## What YOU need to do (the human‑only steps)

The code is done. These four steps need your account/portal access:

1. **bdapps account + app.** Sign up at <https://developer.bdapps.com>, create an
   application, and note its **App ID** and **App Password/Key**. Subscribe the
   app to the **CaaS / Charging (TAP)** API in sandbox mode.

2. **Deploy the sidecar to Render** (the priority path — gives a stable public
   URL that bdapps can reach):
   - Push this repo to GitHub.
   - Render Dashboard → **New → Blueprint** → pick the repo. It reads
     `render.yaml`, builds with the lean `requirements-sidecar.txt` (no
     Streamlit/ChromaDB — fast build), and starts uvicorn.
   - You get **`https://<name>.onrender.com`** (you choose `<name>` — so the
     URL is predictable). Verify `https://<name>.onrender.com/healthz`.
   - Alt for a laptop-only demo: `cloudflared tunnel --url http://localhost:8000`
     (see `deploy/cloudflared-config.example.yml`).

3. **Register the host address** in the bdapps portal, then it issues your key:
   - **Application / host URL** field → `https://<name>.onrender.com`
   - **Callback / notification URL** field (if separate) →
     `https://<name>.onrender.com/bdapps/notify`
   - The sidecar also accepts POST on `/` as a fallback, so a callback is caught
     even if only the bare host can be registered.
   - (`GET /` on the sidecar prints the exact notify URL once
     `BDAPPS_PUBLIC_BASE_URL` is set.)

4. **Set the credentials and go live** — in the **Render dashboard → Environment**
   (or `.env` for local runs), only when you want the real API instead of the
   simulator:
   ```
   BDAPPS_SIMULATE=false
   BDAPPS_APP_ID=<the key/ID bdapps issued in step 3>
   BDAPPS_APP_PASSWORD=<the password/secret bdapps issued>
   BDAPPS_ACCOUNT_ID=<optional payment-instrument accountId, if the portal issues one>
   BDAPPS_CHARGE_URL=<override only if not the documented Direct Debit endpoint>
   BDAPPS_PUBLIC_BASE_URL=https://<name>.onrender.com
   ```
   Redeploy. The outbound charge request is built and sent in
   `bdapps/client.py::_charge_via_bdapps`; `BDAPPS_CHARGE_URL` now defaults to
   the documented `https://developer.bdapps.com/caas/direct/debit` endpoint.

> **Contract status.** The charge **request/response** now follow the documented
> bdapps CaaS `caas/direct/debit` (Direct Debit) schema — flat
> `{applicationId, password, externalTrxId, subscriberId,
> paymentInstrumentName:"MobileAccount", accountId?, amount, currency}` in, and
> the `{externalTrxId, internalTrxId, referenceId, timeStamp, statusCode:"S1000",
> statusDetail}` envelope out. The simulator mirrors that same envelope. Two
> things the doc does **not** pin down and that stay isolated to
> `bdapps/client.py::_charge_via_bdapps` / the `notify` handler: how `password`
> is **encoded** (the sample is a 32-char value — we send the app password
> as-is) and the exact **ack body** `/bdapps/notify` must return.

## Real vs. mock (for the README / judges)

- **Real:** the full request → response → callback → receipt flow, the SQLite
  ledger, MSISDN validation, balance deduction, and idempotent references.
- **Simulated (by default):** the operator charge itself. `BDAPPS_SIMULATE=true`
  deducts from a seeded local balance instead of a real subscriber account —
  which is exactly what "sandbox/simulator mode" in the brief asks for.
