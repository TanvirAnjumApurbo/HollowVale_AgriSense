"""Runtime settings for the bdapps CaaS sidecar, resolved from the environment.

Settings are read live from ``os.environ`` on every ``get_settings()`` call
(no import-time caching) so tests can point the sidecar at a temp database and
flip simulate mode with ``monkeypatch.setenv`` without reimporting the app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # bdapps app credentials (from the developer portal). Only needed when
    # simulate is False -- the simulator ignores them.
    app_id: str
    app_password: str
    # Optional payment-instrument account id (documented ``accountId`` field).
    # Sent only when set; unused while simulate is True.
    account_id: str
    # The real bdapps charging endpoint. Defaults to the documented Direct Debit
    # path; override for a sandbox host. Unused while simulate is True.
    charge_url: str
    # Master switch. True => deterministic local simulator (default, demo-safe).
    simulate: bool
    currency: str
    # Simulator-only: starting mobile balance seeded per new test MSISDN, so
    # "operator balance deduction" is visible in the demo.
    sim_initial_balance: float
    # Flat advisory fee used by /bdapps/quote?item=advisory. DCB caps are small,
    # so the default is a token amount.
    advisory_fee_bdt: float
    # Optional shared secret required on /bdapps/notify (?token= or
    # X-Notify-Token header). Empty => open endpoint (fine for sandbox).
    notify_token: str
    # Public base URL of this sidecar (the tunnel/Render address). Used only to
    # build absolute receipt links in responses/notifications.
    public_base_url: str
    db_path: str
    http_timeout: float

    @property
    def notify_url(self):
        """The exact host address to register on the bdapps portal."""
        base = self.public_base_url.rstrip("/")
        return f"{base}/bdapps/notify" if base else "/bdapps/notify"


def get_settings():
    return Settings(
        app_id=os.environ.get("BDAPPS_APP_ID", ""),
        app_password=os.environ.get("BDAPPS_APP_PASSWORD", ""),
        account_id=os.environ.get("BDAPPS_ACCOUNT_ID", ""),
        charge_url=os.environ.get(
            "BDAPPS_CHARGE_URL", "https://developer.bdapps.com/caas/direct/debit"
        ),
        simulate=_env_bool("BDAPPS_SIMULATE", True),
        currency=os.environ.get("BDAPPS_CURRENCY", "BDT"),
        sim_initial_balance=_env_float("BDAPPS_SIM_INITIAL_BALANCE", 1000.0),
        advisory_fee_bdt=_env_float("BDAPPS_ADVISORY_FEE_BDT", 20.0),
        notify_token=os.environ.get("BDAPPS_NOTIFY_TOKEN", ""),
        public_base_url=os.environ.get("BDAPPS_PUBLIC_BASE_URL", ""),
        db_path=os.environ.get("BDAPPS_DB_PATH", os.path.join("data", "bdapps.db")),
        http_timeout=_env_float("BDAPPS_HTTP_TIMEOUT", 20.0),
    )
