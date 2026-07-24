"""AgriSense bdapps CaaS sidecar (FastAPI).

The Streamlit app cannot serve inbound HTTP routes, so bdapps -- which POSTs
asynchronous charge notifications to a registered *host address* -- needs a
real web server. This is it. It owns three things:

    POST /bdapps/checkout        app -> bdapps charge request (outbound)
    POST /bdapps/notify          THE host address you register; bdapps -> you
    GET  /bdapps/receipt/{ref}   human/JSON receipt

plus small helpers (/bdapps/quote wires to the financial-projection tool,
/bdapps/balance and /bdapps/transactions make the sandbox inspectable).

Run:  uvicorn server:app --host 0.0.0.0 --port 8000
Defaults to the local simulator -- no credentials, no network required.
"""

from __future__ import annotations

import json
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from bdapps import store
from bdapps.client import BdappsCaasClient
from bdapps.config import get_settings
from tools.financials import compute_financial_projection

# Local dev convenience: load .env if present. On Render/cloud there is no .env
# -- env vars come from the dashboard -- so this is a harmless no-op there.
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    yield


app = FastAPI(
    title="AgriSense bdapps CaaS sidecar",
    version="1.0.0",
    summary="Charging-as-a-Service checkout, callback host, and receipts for AgriSense AI.",
    lifespan=lifespan,
)


class CheckoutRequest(BaseModel):
    msisdn: str = Field(..., description="Subscriber mobile number (Bangladesh).")
    amount: float = Field(..., gt=0, description="Amount to charge.")
    currency: str | None = Field(None, description="Defaults to the configured currency (BDT).")
    description: str | None = Field(None, description="What the farmer is paying for.")
    reference: str | None = Field(None, description="Idempotency key; auto-generated if omitted.")


def _new_reference():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"AGRI-{stamp}-{secrets.token_hex(3).upper()}"


def _receipt_url(settings, reference):
    base = settings.public_base_url.rstrip("/")
    return f"{base}/bdapps/receipt/{reference}" if base else f"/bdapps/receipt/{reference}"


@app.get("/")
def index():
    settings = get_settings()
    return {
        "service": "AgriSense bdapps CaaS sidecar",
        "mode": "SIMULATOR" if settings.simulate else "LIVE",
        "host_address_to_register": settings.notify_url,
        "endpoints": {
            "checkout": "POST /bdapps/checkout",
            "notify (register this)": "POST /bdapps/notify",
            "receipt": "GET /bdapps/receipt/{reference}",
            "quote": "GET /bdapps/quote?crop=rice_boro&area_acres=1&item=advisory",
            "balance": "GET /bdapps/balance/{msisdn}",
            "transactions": "GET /bdapps/transactions",
        },
    }


@app.get("/healthz")
def healthz():
    settings = get_settings()
    return {"status": "ok", "simulate": settings.simulate, "currency": settings.currency}


@app.get("/bdapps/quote")
def quote(crop: str, area_acres: float = 1.0, item: str = "advisory"):
    """Derive a charge amount + description from the deterministic financial tool.

    item=advisory -> a flat, DCB-friendly advisory fee.
    item=inputs   -> the plan's fertilizer input-package cost.
    """
    settings = get_settings()
    projection = compute_financial_projection(crop, area_acres)
    if isinstance(projection, dict) and projection.get("error"):
        return JSONResponse(status_code=400, content={"error": projection["error"]})

    label = projection["label"]
    if item == "inputs":
        amount = projection["cost_breakdown_bdt"]["fertilizer"]
        description = f"Fertilizer input package for {label} ({area_acres} acre)"
    else:
        item = "advisory"
        amount = settings.advisory_fee_bdt
        description = f"AgriSense season advisory for {label} ({area_acres} acre)"

    return {
        "crop": projection["crop"],
        "label": label,
        "area_acres": area_acres,
        "item": item,
        "amount": round(float(amount), 2),
        "currency": settings.currency,
        "description": description,
        "projection_total_cost_bdt": projection["total_cost_bdt"],
    }


@app.post("/bdapps/checkout")
def checkout(req: CheckoutRequest):
    """Initiate a charge: record it, call bdapps (or the simulator), record the
    result, and return a status the UI can render into a receipt."""
    settings = get_settings()
    currency = req.currency or settings.currency
    description = req.description or "AgriSense advisory service"
    reference = req.reference or _new_reference()

    client = BdappsCaasClient(settings)
    request_payload = client._build_charge_payload(
        req.msisdn, req.amount, currency, description, reference
    )
    # Never persist the app password, even in the sandbox.
    safe_payload = {**request_payload, "password": "***"}
    store.create_transaction(
        reference=reference,
        msisdn=req.msisdn,
        amount=req.amount,
        currency=currency,
        description=description,
        request_json=json.dumps(safe_payload),
    )

    result = client.charge(req.msisdn, req.amount, currency, description, reference)
    store.update_transaction(
        reference,
        status=result.status,
        status_code=result.status_code,
        status_detail=result.status_detail,
        provider_ref=result.provider_ref,
        balance_after=result.balance_after,
        response_json=json.dumps(result.raw_response),
    )

    return {
        "reference": reference,
        "status": result.status,
        "status_code": result.status_code,
        "status_detail": result.status_detail,
        "amount": round(float(req.amount), 2),
        "currency": currency,
        "description": description,
        "provider_ref": result.provider_ref,
        "balance_after": result.balance_after,
        "receipt_url": _receipt_url(settings, reference),
        "mode": "SIMULATOR" if settings.simulate else "LIVE",
    }


@app.post("/bdapps/notify")
async def notify(request: Request):
    """The host address registered on the bdapps portal.

    bdapps POSTs asynchronous charge/subscription notifications here. We match
    them to a transaction by any of the reference fields bdapps might use,
    record the payload, and return the acknowledgement it expects.
    """
    return await _handle_notification(request)


@app.post("/")
async def notify_root(request: Request):
    """Safety net: some bdapps app configs let you register only the bare host
    address and POST notifications to it, rather than to a /bdapps/notify path.
    Route those to the same handler so a callback is never missed. (GET / is
    still the info page -- only POST is treated as a notification.)"""
    return await _handle_notification(request)


async def _handle_notification(request: Request):
    settings = get_settings()

    # Optional shared-secret gate so the public endpoint is not wide open.
    if settings.notify_token:
        supplied = request.query_params.get("token") or request.headers.get("X-Notify-Token")
        if supplied != settings.notify_token:
            return JSONResponse(status_code=401, content={"statusCode": "E1901", "statusDetail": "Unauthorized"})

    try:
        payload = await request.json()
    except Exception:
        form = await request.form()
        payload = dict(form)

    reference = _find_reference(payload)
    matched = False
    if reference:
        existing = store.get_transaction(reference)
        if existing:
            matched = True
            new_status = _status_from_notification(payload) or existing["status"]
            store.update_transaction(
                reference,
                status=new_status,
                notify_json=json.dumps(payload),
            )

    # TODO(confirm): the exact acknowledgement body/status bdapps expects here.
    return {
        "statusCode": "S1000",
        "statusDetail": "Notification received",
        "reference": reference,
        "matched": matched,
    }


@app.get("/bdapps/receipt/{reference}")
def receipt(reference: str, request: Request, format: str | None = None):
    txn = store.get_transaction(reference)
    if txn is None:
        return JSONResponse(status_code=404, content={"error": f"No transaction '{reference}'."})

    wants_json = format == "json" or "application/json" in request.headers.get("accept", "")
    if wants_json:
        return txn
    return HTMLResponse(_receipt_html(txn))


@app.get("/bdapps/balance/{msisdn}")
def balance(msisdn: str):
    from bdapps.client import normalize_msisdn

    canonical = normalize_msisdn(msisdn) or msisdn
    return {"msisdn": canonical, "balance": store.get_balance(canonical)}


@app.get("/bdapps/transactions")
def transactions(limit: int = 25):
    return {"transactions": store.list_transactions(limit)}


# --- helpers --------------------------------------------------------------

def _find_reference(payload):
    """bdapps may nest the reference under several names; try them all."""
    if not isinstance(payload, dict):
        return None
    for key in ("externalTrxId", "referenceCode", "reference", "transactionId"):
        if payload.get(key):
            return str(payload[key])
    txn = payload.get("amountTransaction")
    if isinstance(txn, dict):
        for key in ("referenceCode", "externalTrxId"):
            if txn.get(key):
                return str(txn[key])
    return None


def _status_from_notification(payload):
    if not isinstance(payload, dict):
        return None
    code = str(payload.get("statusCode") or "")
    if code == "S1000":
        return "NOTIFIED_SUCCESS"
    if code:
        return "NOTIFIED_FAILED"
    op = payload.get("transactionOperationStatus")
    if op:
        return f"NOTIFIED_{str(op).upper()}"
    return "NOTIFIED"


def _receipt_html(txn):
    charged = txn["status"] in {"CHARGED", "NOTIFIED_SUCCESS"}
    colour = "#137333" if charged else "#b3261e"
    badge = "PAID" if charged else txn["status"]
    balance = "" if txn.get("balance_after") is None else (
        f"<tr><td>Balance after</td><td>{txn['balance_after']:.2f} {txn['currency']}</td></tr>"
    )
    provider = txn.get("provider_ref") or "-"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Receipt {txn['reference']}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#f6f7f9; margin:0; padding:2rem; }}
  .card {{ max-width:460px; margin:0 auto; background:#fff; border-radius:14px;
           box-shadow:0 2px 14px rgba(0,0,0,.08); overflow:hidden; }}
  .head {{ padding:1.4rem 1.6rem; border-bottom:1px solid #eee; }}
  .brand {{ font-weight:700; font-size:1.1rem; }} .brand span {{ color:#137333; }}
  .badge {{ float:right; color:#fff; background:{colour}; padding:.2rem .6rem;
            border-radius:999px; font-size:.8rem; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; }}
  td {{ padding:.55rem 1.6rem; border-bottom:1px solid #f0f0f0; font-size:.95rem; }}
  td:first-child {{ color:#666; }} td:last-child {{ text-align:right; font-weight:600; }}
  .amt {{ font-size:1.6rem; font-weight:800; padding:1rem 1.6rem; }}
  .foot {{ padding:1rem 1.6rem; color:#888; font-size:.8rem; }}
</style></head><body>
  <div class="card">
    <div class="head"><span class="badge">{badge}</span>
      <div class="brand">🌾 AgriSense <span>AI</span></div>
      <div style="color:#888;font-size:.85rem">bdapps CaaS receipt</div></div>
    <div class="amt">{txn['amount']:.2f} {txn['currency']}</div>
    <table>
      <tr><td>Description</td><td>{txn.get('description') or '-'}</td></tr>
      <tr><td>Reference</td><td>{txn['reference']}</td></tr>
      <tr><td>Subscriber</td><td>{txn['msisdn']}</td></tr>
      <tr><td>Status</td><td>{txn['status']} ({txn.get('status_code') or '-'})</td></tr>
      <tr><td>bdapps ref</td><td>{provider}</td></tr>
      {balance}
      <tr><td>Time</td><td>{txn['updated_at']}</td></tr>
    </table>
    <div class="foot">Charged via bdapps Charging-as-a-Service. Amounts computed by
      AgriSense deterministic tools.</div>
  </div>
</body></html>"""
