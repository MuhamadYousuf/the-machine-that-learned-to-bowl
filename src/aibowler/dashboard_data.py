"""Tolerant read-only readers used by the live training dashboard."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def read_csv_tolerant(path: str | Path) -> list[dict[str, str]]:
    """Read complete CSV records and ignore a partial final write."""
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", newline="", encoding="utf-8") as file:
        header_line = file.readline()
        if not header_line.endswith(("\n", "\r")):
            return []
        try:
            headers = next(csv.reader([header_line]))
        except (csv.Error, StopIteration):
            return []
        rows = []
        for line in file:
            if not line.endswith(("\n", "\r")):
                continue
            try:
                values = next(csv.reader([line]))
            except (csv.Error, StopIteration):
                continue
            if len(values) != len(headers):
                continue
            rows.append(dict(zip(headers, values)))
        return rows


def read_jsonl_tolerant(path: str | Path) -> list[dict]:
    """Read complete JSONL records, skipping malformed or partial lines."""
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    with source.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.endswith(("\n", "\r")):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def read_latest_jsonl_tolerant(path: str | Path, *, tail_bytes: int = 8_000_000) -> dict | None:
    """Read the latest complete JSONL object without parsing the whole log."""
    source = Path(path)
    if not source.exists():
        return None
    with source.open("rb") as file:
        file.seek(0, 2)
        size = file.tell()
        file.seek(max(0, size - tail_bytes))
        content = file.read().decode("utf-8", errors="ignore")
    lines = content.splitlines()
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None
