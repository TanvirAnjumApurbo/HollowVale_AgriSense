"""bdapps CaaS (Charging-as-a-Service) integration for AgriSense.

This package is the payment seam. It is deliberately independent of Streamlit
so it can run under a FastAPI/uvicorn sidecar that exposes a public host
address to the bdapps portal (the Streamlit UI cannot serve inbound callback
routes). The single place the real bdapps wire format lives is
``bdapps.client.BdappsCaasClient._charge_via_bdapps`` -- everything else runs
against a deterministic simulator by default so the sandbox demo works with no
network and no credentials.
"""
