import base64
import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "cvee-20260208")
BILLING_API = "https://cloudbilling.googleapis.com/v1"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_token():
    """Get OAuth2 token from GCP metadata server (always available)."""
    req = Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/"
        "service-accounts/default/token"
        "?scopes=https://www.googleapis.com/auth/cloud-platform"
    )
    req.add_header("Metadata-Flavor", "Google")
    with urlopen(req, timeout=5) as resp:  # nosec B310 -- GCP metadata endpoint only
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def _api_request(method, path, body=None):
    """Authenticated request to Cloud Billing REST API. Pure stdlib."""
    url = f"{BILLING_API}/{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {_get_token()}")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=15) as resp:  # nosec B310 -- GCP billing API, authenticated
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8")[:300] if e.fp else str(e)
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"URL error: {e.reason}") from e


def stop_billing(event, context=None):
    """
    Cloud Function triggered by Pub/Sub budget alert.
    If cost >= 100% of budget, calls Cloud Billing API to unlink project.

    Zero external dependencies. Uses only Python stdlib + GCP metadata server.
    """
    # Parse Pub/Sub message (compatible 1st gen and 2nd gen)
    if hasattr(event, "get_json"):
        envelope = event.get_json()
        pubsub_message = envelope.get("message", {})
        data = base64.b64decode(pubsub_message.get("data", "")).decode("utf-8")
    else:
        data = base64.b64decode(event.get("data", "")).decode("utf-8")

    try:
        alert = json.loads(data)
    except Exception:
        logger.exception("Failed to parse Pub/Sub message")
        return ("OK (parse error)", 200)

    budget_name = alert.get("budgetDisplayName", "unknown")
    cost_amount = alert.get("costAmount", 0)
    budget_amount = alert.get("budgetAmount", 0)
    threshold = alert.get("alertThresholdExceeded", 0)
    currency = alert.get("currencyCode", "EUR")

    logger.info(
        "Budget: %s | %.2f/%.2f %s (threshold %.2f)",
        budget_name,
        cost_amount,
        budget_amount,
        currency,
        threshold,
    )

    if threshold < 1.0:
        logger.info("Below cutoff (%.2f < 1.0).", threshold)
        return ("OK (below cutoff)", 200)

    project_name = f"projects/{PROJECT_ID}"

    # Read current billing state
    try:
        info = _api_request("GET", f"{project_name}/billingInfo")
    except Exception:
        logger.exception("getBillingInfo failed")
        return ("ERROR: getBillingInfo failed", 500)

    billing_enabled = info.get("billingEnabled", False)
    logger.info("billingEnabled = %s", billing_enabled)

    if not billing_enabled:
        logger.info("Already disabled.")
        return ("OK (already disabled)", 200)

    # DISABLE billing
    logger.warning(
        "DISABLING BILLING! %s spent %.2f %s / budget %.2f %s",
        PROJECT_ID,
        cost_amount,
        currency,
        budget_amount,
        currency,
    )

    try:
        _api_request("PUT", f"{project_name}/billingInfo", {"billingAccountName": ""})
    except Exception:
        logger.exception("updateBillingInfo failed")
        return ("ERROR: updateBillingInfo failed", 500)

    logger.info("BILLING DISABLED for %s.", PROJECT_ID)
    return ("OK (billing disabled)", 200)
