from __future__ import annotations

import re
from collections.abc import Mapping


SECRET_KEY_RE = re.compile(
    r"(TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL)",
    re.IGNORECASE,
)
REDACTED = "***REDACTED***"


def is_secret_key(key: str) -> bool:
    return bool(SECRET_KEY_RE.search(key))


def redact_mapping(values: Mapping[str, object]) -> dict[str, object]:
    return {
        key: REDACTED if is_secret_key(key) else value
        for key, value in values.items()
    }
