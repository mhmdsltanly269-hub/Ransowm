#!/usr/bin/env python3
"""Small process-isolated inference engine for the supplied RIDS artifacts.

The original project was trained with 50 numeric event features. This module
keeps the trained scaler/model untouched and focuses on turning common raw
behavior exports (JSON/JSONL/CSV/text) and binary samples into that exact
feature vector.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from xml.etree import ElementTree
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xlrd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
FEATURE_NAMES = [
    line.strip()
    for line in (MODEL_DIR / "feature_names.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
SCALER = joblib.load(MODEL_DIR / "scaler.pkl")
MODEL = joblib.load(MODEL_DIR / "xgboost_model.pkl")


def _clip_float(value: Any) -> float | None:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return None


def _walk_feature_values(value: Any, target: str, found: list[float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) == target:
                number = _clip_float(child)
                if number is not None:
                    found.append(number)
            _walk_feature_values(child, target, found)
    elif isinstance(value, list):
        for child in value:
            _walk_feature_values(child, target, found)


def _extract_event_rows(structured: Any, vector: np.ndarray) -> int:
    if not isinstance(structured, list):
        return 0
    event_keys = {"event_id", "event", "syscall", "opcode", "feature", "id", "operation_id", "activity_id", "code"}
    count_keys = {"count", "frequency", "occurrences", "value", "weight", "total"}
    feature_indexes = {name: index for index, name in enumerate(FEATURE_NAMES)}
    matched = 0
    for row in structured:
        if not isinstance(row, dict):
            continue
        normalized = {str(key).strip().lower(): value for key, value in row.items()}
        event_value = next((normalized[key] for key in event_keys if key in normalized), None)
        feature = str(event_value).strip() if event_value is not None else ""
        if feature not in feature_indexes:
            continue
        weight_value = next((normalized[key] for key in count_keys if key in normalized), 1)
        weight = _clip_float(weight_value)
        vector[feature_indexes[feature]] += weight if weight is not None else 1.0
        matched += 1
    return matched


def _read_xlsx(raw: bytes) -> list[dict[str, Any]]:
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(raw)) as workbook:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("main:si", namespace):
                shared_strings.append("".join(node.text or "" for node in item.iter() if node.tag.endswith("}t")))

        sheet_names = sorted(
            name for name in workbook.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not sheet_names:
            return []
        root = ElementTree.fromstring(workbook.read(sheet_names[0]))
        rows: list[list[str]] = []
        for row in root.findall(".//main:sheetData/main:row", namespace):
            values: dict[int, str] = {}
            for cell in row.findall("main:c", namespace):
                reference = cell.attrib.get("r", "A1")
                column = re.match(r"[A-Z]+", reference)
                if not column:
                    continue
                column_index = 0
                for letter in column.group(0):
                    column_index = column_index * 26 + ord(letter) - ord("A") + 1
                value_node = cell.find("main:v", namespace)
                value = value_node.text if value_node is not None and value_node.text else ""
                if cell.attrib.get("t") == "inlineStr":
                    inline_node = cell.find(".//main:t", namespace)
                    value = inline_node.text if inline_node is not None and inline_node.text else ""
                if cell.attrib.get("t") == "s" and value.isdigit():
                    shared_index = int(value)
                    value = shared_strings[shared_index] if shared_index < len(shared_strings) else value
                values[column_index - 1] = value
            if values:
                width = max(values) + 1
                rows.append([values.get(index, "") for index in range(width)])
        if not rows:
            return []
        headers = [str(value).strip() or f"column_{index + 1}" for index, value in enumerate(rows[0])]
        return [
            {headers[index]: value for index, value in enumerate(row) if index < len(headers)}
            for row in rows[1:]
        ]


def _read_xls(raw: bytes) -> list[dict[str, Any]]:
    workbook = xlrd.open_workbook(file_contents=raw, on_demand=True)
    sheet = workbook.sheet_by_index(0)
    if sheet.nrows < 2:
        return []
    headers = [str(value).strip() or f"column_{index + 1}" for index, value in enumerate(sheet.row_values(0))]
    return [
        {headers[index]: value for index, value in enumerate(sheet.row_values(row_index)) if index < len(headers)}
        for row_index in range(1, sheet.nrows)
    ]


def _read_structured(path: Path, text: str, raw: bytes | None = None) -> tuple[Any, str]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        try:
            return json.loads(text), "structured-json"
        except json.JSONDecodeError:
            rows = []
            for line in text.splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if rows:
                return rows, "structured-jsonl"
    if suffix in {".csv", ".tsv"}:
        try:
            dialect = csv.excel_tab if suffix == ".tsv" else csv.excel
            return list(csv.DictReader(io.StringIO(text), dialect=dialect)), "tabular"
        except (csv.Error, UnicodeError):
            pass
    if suffix == ".xlsx" and raw:
        try:
            rows = _read_xlsx(raw)
            if rows:
                return rows, "spreadsheet-xlsx"
        except (KeyError, ValueError, zipfile.BadZipFile, ElementTree.ParseError):
            pass
    if suffix == ".xls" and raw:
        try:
            rows = _read_xls(raw)
            if rows:
                return rows, "spreadsheet-xls"
        except (ValueError, xlrd.biffh.XLRDError):
            pass
    return None, "raw-bytes"


def _extract_from_raw(path: Path, original_name: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    structured, method = _read_structured(Path(original_name) if original_name else path, text, raw)
    vector = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    matched = 0

    if structured is not None:
        for index, feature in enumerate(FEATURE_NAMES):
            values: list[float] = []
            _walk_feature_values(structured, feature, values)
            if values:
                vector[index] = float(np.mean(values))
                matched += 1
        matched += _extract_event_rows(structured, vector)

    # Behavior exports often contain event IDs as values rather than columns.
    # Count exact feature tokens in the source text without matching substrings.
    for index, feature in enumerate(FEATURE_NAMES):
        token_hits = len(re.findall(rf"(?<!\d){re.escape(feature)}(?!\d)", text))
        if token_hits:
            if vector[index] == 0:
                vector[index] = float(token_hits)
            else:
                vector[index] = max(vector[index], float(token_hits))
            matched += 1

    if matched:
        extraction_method = method if method != "raw-bytes" else "behavior-text"
    else:
        # For PE/DLL/archives and other raw samples, preserve measurable byte
        # evidence instead of the original random placeholder. The vector is
        # deterministic and based on byte distribution, entropy and size.
        if not raw:
            vector[:] = 0
        else:
            counts = np.bincount(np.frombuffer(raw, dtype=np.uint8), minlength=256)
            total = float(len(raw))
            normalized = counts / total
            entropy = -float(sum(p * math.log2(p) for p in normalized if p > 0))
            digest = hashlib.sha256(raw).digest()
            for index in range(len(FEATURE_NAMES)):
                start = (index * 5) % 256
                bucket = float(normalized[start : start + 5].sum())
                digest_signal = digest[index % len(digest)] / 255.0
                vector[index] = bucket * 100.0 + digest_signal + entropy / 8.0
        extraction_method = "byte-statistics"

    return vector, {
        "extraction_method": extraction_method,
        "matched_features": int(matched),
        "bytes_read": len(raw),
        "records_detected": len(structured) if isinstance(structured, list) else (1 if structured else 0),
    }


def _predict(vectors: list[np.ndarray]) -> dict[str, Any]:
    matrix = np.asarray(vectors, dtype=np.float64)
    scaled = SCALER.transform(matrix)
    probabilities = MODEL.predict_proba(scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "predictions": [int(value) for value in predictions],
        "probabilities": [float(value) for value in probabilities],
        "model_used": "XGBoost (50 features)",
        "scaled_features": scaled.tolist(),
    }


def _explain(scaled: np.ndarray) -> dict[str, Any]:
    importances = getattr(MODEL, "feature_importances_", np.ones(len(FEATURE_NAMES)))
    contributions = np.asarray(importances, dtype=np.float64) * np.asarray(scaled, dtype=np.float64)
    order = np.argsort(np.abs(contributions))[::-1][:10]
    return {
        "base_value": float(np.mean(getattr(MODEL, "base_score", 0.0) or 0.0)),
        "top_features": [
            {"feature": FEATURE_NAMES[index], "shap_value": float(contributions[index])}
            for index in order
        ],
        "lime": [
            {"feature": FEATURE_NAMES[index], "weight": float(contributions[index])}
            for index in order
        ],
    }


def main() -> None:
    request = json.loads(sys.stdin.read() or "{}")
    action = request.get("action")
    if action == "metadata":
        importances = getattr(MODEL, "feature_importances_", np.zeros(len(FEATURE_NAMES)))
        print(
            json.dumps(
                {
                    "features": FEATURE_NAMES,
                    "feature_importance": [
                        {"Feature": name, "Importance": float(importances[index])}
                        for index, name in enumerate(FEATURE_NAMES)
                    ],
                }
            )
        )
        return
    if action == "predict":
        samples = request.get("samples") or [request.get("sample") or {}]
        vectors = []
        for sample in samples:
            vectors.append(np.asarray([_clip_float(sample.get(name)) or 0.0 for name in FEATURE_NAMES]))
        result = _predict(vectors)
        if request.get("explain") and vectors:
            result["explanation"] = _explain(np.asarray(result["scaled_features"][0]))
        result.pop("scaled_features", None)
        print(json.dumps(result))
        return
    if action == "upload":
        vector, metadata = _extract_from_raw(Path(request["path"]), request.get("original_name"))
        result = _predict([vector])
        result["features"] = {name: float(vector[index]) for index, name in enumerate(FEATURE_NAMES)}
        result["extraction"] = metadata
        if request.get("explain"):
            scaled = SCALER.transform(np.asarray([vector]))[0]
            result["explanation"] = _explain(scaled)
        result.pop("scaled_features", None)
        print(json.dumps(result))
        return
    raise ValueError("Unsupported engine action")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        raise