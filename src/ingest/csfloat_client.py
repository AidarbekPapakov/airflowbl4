import time
from typing import List, Dict, Any
from urllib.parse import quote

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

BASE_URL = "https://csfloat.com/api/v1"
_MIN_SLEEP = 1.0


def _sleep_from_headers(headers: httpx.Headers) -> None:
    remaining = headers.get("x-ratelimit-remaining")
    reset = headers.get("x-ratelimit-reset")
    if remaining is not None and int(remaining) <= 5 and reset is not None:
        wait = max(0.0, int(reset) - time.time()) + 0.5
        time.sleep(wait)
    else:
        time.sleep(_MIN_SLEEP)


class RateLimitError(Exception):
    pass


def _raise_on_rate_limit(response: httpx.Response) -> None:
    if response.status_code == 429:
        raise RateLimitError(f"429 rate-limited: {response.text[:200]}")


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
)
def fetch_sales_history(
    market_hash_name: str,
    paint_index: int | None,
    api_key: str,
) -> List[Dict[str, Any]]:
    """Return all sales history entries for one skin variant.

    paint_index=0 is treated by the API as "no filter"; pass None to omit the
    param entirely (same behaviour, avoids the confusing 0-case).
    """
    url = f"{BASE_URL}/history/{quote(market_hash_name)}/sales"
    params: Dict[str, int] = {}
    if paint_index:
        params["paint_index"] = paint_index

    with httpx.Client(timeout=20) as client:
        r = client.get(url, headers={"Authorization": api_key}, params=params)

    _raise_on_rate_limit(r)

    if r.status_code != 200:
        r.raise_for_status()

    data = r.json()
    if not isinstance(data, list):
        # e.g. {"code": 200, "message": "ErrorCodeSalesHistoryNotAvailable"}
        return []

    _sleep_from_headers(r.headers)
    return data
