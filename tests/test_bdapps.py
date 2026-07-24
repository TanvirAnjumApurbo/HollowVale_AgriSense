"""Tests for the bdapps CaaS sidecar: store, simulator client, and endpoints.

All run against the local simulator (BDAPPS_SIMULATE=true) with a per-test temp
SQLite database, so no credentials or network are touched.
"""

import pytest
from fastapi.testclient import TestClient

from bdapps import store
from bdapps.client import BdappsCaasClient, normalize_msisdn
from bdapps.config import get_settings
from server import app

GOOD_MSISDN = "8801712345678"


@pytest.fixture(autouse=True)
def _sandbox_env(tmp_path, monkeypatch):
    """Point the sidecar at a fresh temp DB in simulate mode for every test."""
    monkeypatch.setenv("BDAPPS_SIMULATE", "true")
    monkeypatch.setenv("BDAPPS_DB_PATH", str(tmp_path / "bdapps.db"))
    monkeypatch.setenv("BDAPPS_SIM_INITIAL_BALANCE", "500")
    monkeypatch.setenv("BDAPPS_ADVISORY_FEE_BDT", "20")
    monkeypatch.setenv("BDAPPS_CURRENCY", "BDT")
    monkeypatch.delenv("BDAPPS_NOTIFY_TOKEN", raising=False)
    store.init_db()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# --- MSISDN normalization -------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("8801712345678", "8801712345678"),
        ("+8801712345678", "8801712345678"),
        ("01712345678", "8801712345678"),
        ("017-1234 5678", "8801712345678"),
        ("0171234567", None),      # too short
        ("8801212345678", None),   # 012 is not a valid BD mobile prefix
        ("", None),
        (None, None),
    ],
)
def test_normalize_msisdn(raw, expected):
    assert normalize_msisdn(raw) == expected


# --- store ----------------------------------------------------------------

def test_transaction_lifecycle():
    store.create_transaction("REF1", GOOD_MSISDN, 20.0, "BDT", "advisory", request_json="{}")
    txn = store.get_transaction("REF1")
    assert txn["status"] == "INITIATED"
    assert txn["amount"] == 20.0

    store.update_transaction("REF1", status="CHARGED", provider_ref="X1", balance_after=480.0)
    txn = store.get_transaction("REF1")
    assert txn["status"] == "CHARGED"
    assert txn["provider_ref"] == "X1"
    assert txn["balance_after"] == 480.0
    assert txn["updated_at"] >= txn["created_at"]

    assert len(store.list_transactions()) == 1


def test_debit_success_and_insufficient():
    store.ensure_subscriber(GOOD_MSISDN, 100.0)
    assert store.get_balance(GOOD_MSISDN) == 100.0

    ok, after = store.debit(GOOD_MSISDN, 30.0)
    assert ok is True and after == 70.0

    ok, after = store.debit(GOOD_MSISDN, 999.0)
    assert ok is False and after == 70.0     # unchanged on failure


# --- simulator client -----------------------------------------------------

def test_simulate_charge_success_deducts_balance():
    client = BdappsCaasClient(get_settings())
    result = client.charge(GOOD_MSISDN, 20.0, "BDT", "advisory", "REFOK")
    assert result.ok is True
    assert result.status == "CHARGED"
    assert result.status_code == "S1000"
    assert result.provider_ref and result.provider_ref.startswith("SIM")
    assert result.balance_after == 480.0     # 500 seeded - 20


def test_simulate_charge_insufficient_balance():
    client = BdappsCaasClient(get_settings())
    result = client.charge(GOOD_MSISDN, 9999.0, "BDT", "advisory", "REFNO")
    assert result.ok is False
    assert result.status == "FAILED"
    assert result.status_code == "P1300"


def test_simulate_charge_invalid_msisdn():
    client = BdappsCaasClient(get_settings())
    result = client.charge("0123", 20.0, "BDT", "advisory", "REFBAD")
    assert result.ok is False
    assert result.status_code == "E1330"


# --- endpoints ------------------------------------------------------------

def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["simulate"] is True


def test_quote_advisory_and_inputs(client):
    advisory = client.get("/bdapps/quote", params={"crop": "rice_boro", "area_acres": 2}).json()
    assert advisory["item"] == "advisory"
    assert advisory["amount"] == 20.0
    assert advisory["currency"] == "BDT"

    inputs = client.get(
        "/bdapps/quote", params={"crop": "rice_boro", "area_acres": 2, "item": "inputs"}
    ).json()
    assert inputs["item"] == "inputs"
    assert inputs["amount"] > 0


def test_quote_bad_crop_is_400(client):
    resp = client.get("/bdapps/quote", params={"crop": "banana", "area_acres": 1})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_checkout_success_then_receipt(client):
    resp = client.post(
        "/bdapps/checkout",
        json={"msisdn": GOOD_MSISDN, "amount": 20, "description": "AgriSense advisory"},
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "CHARGED"
    assert body["status_code"] == "S1000"
    assert body["balance_after"] == 480.0
    assert body["provider_ref"].startswith("SIM")
    reference = body["reference"]

    # JSON receipt
    receipt = client.get(f"/bdapps/receipt/{reference}", params={"format": "json"}).json()
    assert receipt["reference"] == reference
    assert receipt["status"] == "CHARGED"

    # HTML receipt
    html = client.get(f"/bdapps/receipt/{reference}")
    assert html.status_code == 200
    assert "text/html" in html.headers["content-type"]
    assert reference in html.text
    assert "PAID" in html.text


def test_checkout_insufficient_balance(client):
    body = client.post(
        "/bdapps/checkout", json={"msisdn": GOOD_MSISDN, "amount": 100000}
    ).json()
    assert body["status"] == "FAILED"
    assert body["status_code"] == "P1300"


def test_checkout_rejects_nonpositive_amount(client):
    resp = client.post("/bdapps/checkout", json={"msisdn": GOOD_MSISDN, "amount": 0})
    assert resp.status_code == 422     # pydantic gt=0 validation


def test_notify_matches_transaction_and_acks(client):
    reference = client.post(
        "/bdapps/checkout", json={"msisdn": GOOD_MSISDN, "amount": 20}
    ).json()["reference"]

    ack = client.post(
        "/bdapps/notify",
        json={"externalTrxId": reference, "statusCode": "S1000", "statusDetail": "Charged"},
    )
    assert ack.status_code == 200
    assert ack.json()["statusCode"] == "S1000"
    assert ack.json()["matched"] is True

    txn = client.get(f"/bdapps/receipt/{reference}", params={"format": "json"}).json()
    assert txn["status"] == "NOTIFIED_SUCCESS"
    assert txn["notify_json"] is not None


def test_notify_unknown_reference_still_acks(client):
    ack = client.post("/bdapps/notify", json={"externalTrxId": "NOPE", "statusCode": "S1000"})
    assert ack.status_code == 200
    assert ack.json()["matched"] is False


def test_notify_root_alias_matches_transaction(client):
    # bdapps posting to the bare host (POST /) must reach the same handler.
    reference = client.post(
        "/bdapps/checkout", json={"msisdn": GOOD_MSISDN, "amount": 20}
    ).json()["reference"]
    ack = client.post("/", json={"externalTrxId": reference, "statusCode": "S1000"})
    assert ack.status_code == 200
    assert ack.json()["matched"] is True


def test_notify_token_enforced(client, monkeypatch):
    monkeypatch.setenv("BDAPPS_NOTIFY_TOKEN", "s3cr3t")
    denied = client.post("/bdapps/notify", json={"externalTrxId": "X"})
    assert denied.status_code == 401

    allowed = client.post(
        "/bdapps/notify", params={"token": "s3cr3t"}, json={"externalTrxId": "X"}
    )
    assert allowed.status_code == 200
