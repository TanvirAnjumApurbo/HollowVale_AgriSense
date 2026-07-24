"""bdapps CaaS (Charging-as-a-Service) client.

``charge()`` is the one call the rest of the app makes. In simulate mode it
runs a deterministic local model of a Direct Carrier Billing charge (validates
the MSISDN, deducts from a seeded test balance, returns a bdapps-style status
envelope). In live mode it delegates to ``_charge_via_bdapps`` -- the SINGLE
place the real wire format lives, so wiring the sandbox to the real API is a
one-function change once the exact endpoint/fields are confirmed from
https://dev.bdapps.com/API_Documentation/bdapps_tap_api.html
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

import requests

from bdapps import store
from bdapps.config import Settings

# bdapps/hSenid-style status codes, so the demo response looks like the real
# envelope. Confirm the exact codes/labels against the portal docs.
STATUS_SUCCESS = "S1000"
STATUS_INSUFFICIENT = "P1300"
STATUS_INVALID_MSISDN = "E1330"
STATUS_UPSTREAM_ERROR = "E1900"

# A BD mobile number: optional +, optional 88 country code, then 01[3-9] + 8
# digits. Accepts "8801712345678", "+8801712345678", "01712345678".
_MSISDN_RE = re.compile(r"^(?:\+?88)?01[3-9]\d{8}$")


def normalize_msisdn(msisdn):
    """Return a canonical 8801XXXXXXXXX form, or None if it is not a BD mobile."""
    if not msisdn:
        return None
    digits = re.sub(r"[^\d]", "", str(msisdn))
    if digits.startswith("88"):
        digits = digits[2:]
    if not re.fullmatch(r"01[3-9]\d{8}", digits):
        return None
    return "88" + digits


@dataclass
class ChargeResult:
    ok: bool
    status: str            # normalized txn status: CHARGED | FAILED
    status_code: str       # bdapps-style code, e.g. S1000
    status_detail: str
    provider_ref: str | None = None
    balance_after: float | None = None
    raw_response: dict = field(default_factory=dict)


class BdappsCaasClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def charge(self, msisdn, amount, currency, description, reference):
        """Charge a subscriber. Dispatches to the simulator or the real API."""
        if self.settings.simulate:
            return self._simulate_charge(msisdn, amount, currency, description, reference)
        return self._charge_via_bdapps(msisdn, amount, currency, description, reference)

    # --- payload shared by the real call (documents the expected wire shape) ---

    def _build_charge_payload(self, msisdn, amount, currency, description, reference):
        """The request body sent to bdapps. Field names follow the documented
        CaaS/TAP direct-debit shape; adjust to match the portal exactly."""
        canonical = normalize_msisdn(msisdn) or str(msisdn)
        return {
            "applicationId": self.settings.app_id,
            "password": self.settings.app_password,
            "externalTrxId": reference,
            "subscriberId": f"tel:{canonical}",
            "paymentInstrumentName": "Mobile Account",
            "amountTransaction": {
                "referenceCode": reference,
                "transactionOperationStatus": "Charged",
                "paymentAmount": {
                    "chargingInformation": {
                        "amount": f"{float(amount):.2f}",
                        "currency": currency,
                        "description": description,
                    },
                    "chargingMetaData": {
                        "onBehalfOf": "AgriSense AI",
                        "purchaseCategoryCode": "Agri-Advisory",
                        "channel": "WEB",
                    },
                },
            },
        }

    # --- live call: the ONLY place the real bdapps HTTP contract lives --------

    def _charge_via_bdapps(self, msisdn, amount, currency, description, reference):
        payload = self._build_charge_payload(msisdn, amount, currency, description, reference)
        if not self.settings.charge_url:
            return ChargeResult(
                ok=False,
                status="FAILED",
                status_code=STATUS_UPSTREAM_ERROR,
                status_detail="BDAPPS_CHARGE_URL is not configured; cannot reach bdapps.",
                raw_response={"error": "missing charge_url"},
            )
        try:
            # TODO(confirm): exact endpoint path, auth scheme (app password vs
            # bearer token / HMAC header) and JSON field names from the portal
            # docs. Everything else in this package is provider-agnostic.
            resp = requests.post(
                self.settings.charge_url,
                json=payload,
                timeout=self.settings.http_timeout,
                headers={"Content-Type": "application/json"},
            )
            body = _safe_json(resp)
            code = str(body.get("statusCode") or "")
            detail = body.get("statusDetail") or resp.reason or ""
            ok = code == STATUS_SUCCESS
            provider_ref = _extract_provider_ref(body)
            return ChargeResult(
                ok=ok,
                status="CHARGED" if ok else "FAILED",
                status_code=code or STATUS_UPSTREAM_ERROR,
                status_detail=detail or ("Charged" if ok else "Charge declined"),
                provider_ref=provider_ref,
                raw_response=body,
            )
        except requests.RequestException as exc:
            return ChargeResult(
                ok=False,
                status="FAILED",
                status_code=STATUS_UPSTREAM_ERROR,
                status_detail=f"Network error contacting bdapps: {exc}",
                raw_response={"error": str(exc)},
            )

    # --- simulator: deterministic, no network, no credentials ----------------

    def _simulate_charge(self, msisdn, amount, currency, description, reference):
        canonical = normalize_msisdn(msisdn)
        if canonical is None:
            return ChargeResult(
                ok=False,
                status="FAILED",
                status_code=STATUS_INVALID_MSISDN,
                status_detail=f"'{msisdn}' is not a valid Bangladesh mobile number.",
                raw_response=self._sim_envelope(STATUS_INVALID_MSISDN, reference, None),
            )
        store.ensure_subscriber(canonical, self.settings.sim_initial_balance)
        ok, balance_after = store.debit(canonical, amount)
        if not ok:
            return ChargeResult(
                ok=False,
                status="FAILED",
                status_code=STATUS_INSUFFICIENT,
                status_detail=(
                    f"Insufficient balance: {balance_after:.2f} {currency} available, "
                    f"{float(amount):.2f} {currency} required."
                ),
                balance_after=balance_after,
                raw_response=self._sim_envelope(STATUS_INSUFFICIENT, reference, None),
            )
        provider_ref = "SIM" + secrets.token_hex(6).upper()
        return ChargeResult(
            ok=True,
            status="CHARGED",
            status_code=STATUS_SUCCESS,
            status_detail="Charged (simulated)",
            provider_ref=provider_ref,
            balance_after=balance_after,
            raw_response=self._sim_envelope(STATUS_SUCCESS, reference, provider_ref),
        )

    @staticmethod
    def _sim_envelope(status_code, reference, provider_ref):
        return {
            "statusCode": status_code,
            "statusDetail": {
                STATUS_SUCCESS: "Success",
                STATUS_INSUFFICIENT: "Insufficient balance",
                STATUS_INVALID_MSISDN: "Invalid subscriber",
            }.get(status_code, "Error"),
            "simulated": True,
            "amountTransaction": {
                "referenceCode": reference,
                "serverReferenceCode": provider_ref,
            },
        }


def _safe_json(resp):
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {"raw": data}
    except ValueError:
        return {"raw": resp.text}


def _extract_provider_ref(body):
    """Pull bdapps' server-side reference out of a (possibly nested) response."""
    if not isinstance(body, dict):
        return None
    for key in ("serverReferenceCode", "internalTrxId", "transactionId"):
        if body.get(key):
            return str(body[key])
    txn = body.get("amountTransaction")
    if isinstance(txn, dict) and txn.get("serverReferenceCode"):
        return str(txn["serverReferenceCode"])
    return None
