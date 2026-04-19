import re

# MiDaS normalized depth: 0.0 = far, 1.0 = very close
# Tune these based on your fine-tuned model's output
DEPTH_THRESHOLDS = {
    "very close": 0.80,
    "close":      0.60,
    "medium":     0.35,
}

ALERT_PROXIMITIES = {"very close", "close", "medium"}

PROXIMITY_PRIORITY = {
    "very close": 4,
    "close":      3,
    "medium":     2,
    "far":        0,
}


def parse_label(text: str):
    """
    Parse MiDaS bounding box label.

    Expected format: "object | proximity | direction | d = 0.92"
    Examples:
        "table | medium | left | d = 0.92"
        "column | far | ahead | d= 0.42"
        "person | close | center | d=0.78"

    Returns dict or None if alert not needed.
    """
    text  = text.strip()
    parts = [p.strip() for p in text.split("|")]

    if len(parts) < 3:
        return None

    label     = parts[0].strip().lower().replace("-", " ")
    proximity = parts[1].strip().lower()
    direction = parts[2].strip().lower()

    # Extract d value from last part if present
    d_value = None
    if len(parts) >= 4:
        match = re.search(r"d\s*=\s*(\d+\.?\d*)", parts[3], re.IGNORECASE)
        if match:
            d_value = float(match.group(1))

    # Primary filter: use proximity label
    if proximity not in ALERT_PROXIMITIES:
        # Secondary check: if d_value disagrees, trust d_value
        if d_value is not None and d_value >= DEPTH_THRESHOLDS["medium"]:
            proximity = _proximity_from_d(d_value)
            if proximity not in ALERT_PROXIMITIES:
                return None
        else:
            return None

    # Cross-check: d_value can only upgrade proximity, never downgrade
    if d_value is not None:
        d_proximity = _proximity_from_d(d_value)
        if PROXIMITY_PRIORITY.get(d_proximity, 0) > PROXIMITY_PRIORITY.get(proximity, 0):
            proximity = d_proximity

    return {
        "label":     label,
        "proximity": proximity,
        "direction": direction,
        "d_value":   d_value,
    }


def _proximity_from_d(d: float) -> str:
    """Convert raw MiDaS d value to proximity label."""
    if d >= DEPTH_THRESHOLDS["very close"]:
        return "very close"
    elif d >= DEPTH_THRESHOLDS["close"]:
        return "close"
    elif d >= DEPTH_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "far"


def build_alert_phrase(parsed: dict) -> str:
    """
    Build spoken alert from parsed label.

    Examples:
        "table medium on the left"
        "person very close ahead, stop"
        "column close ahead"
    """
    label     = parsed["label"]
    proximity = parsed["proximity"]
    direction = parsed["direction"]

    # Normalize direction
    if direction in ("center", "ahead"):
        dir_phrase = "ahead"
    elif direction in ("left", "right"):
        dir_phrase = f"on the {direction}"
    else:
        dir_phrase = ""

    # Critical stop warning
    if proximity == "very close" and dir_phrase == "ahead":
        return f"{label} very close ahead, stop"

    if proximity == "very close":
        return f"{label} very close {dir_phrase}"

    parts = [label, proximity]
    if dir_phrase:
        parts.append(dir_phrase)

    return " ".join(parts)