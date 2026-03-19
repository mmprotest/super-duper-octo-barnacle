from __future__ import annotations

from pathlib import Path


def _parse_scalar(raw: str):
    value = raw.strip()
    if value.lower() in {"null", "none"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_parse(text: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, raw_value = line.partition(":")
        if not _:
            continue
        parsed[key.strip()] = _parse_scalar(raw_value)
    return parsed


def load_yaml_file(path: str) -> dict[str, object]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return _simple_yaml_parse(text)
