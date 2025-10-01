"""Formatting utilities for structured outputs."""

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union

from .constants import CLASS_NAMES


def ordered_probs(prob_map: Dict[str, float]) -> OrderedDict[str, float]:
    """Ensure all classes exist in canonical order; missing classes get 0.0."""
    return OrderedDict((cls, float(prob_map.get(cls, 0.0))) for cls in CLASS_NAMES)


def topk_from_probs(prob_map: Dict[str, float], k: int) -> List[Dict[str, float]]:
    """Extract top-k predictions sorted by probability (descending)."""
    items = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
    return [{"label": lbl, "prob": float(p)} for lbl, p in items[:k]]


def build_record(image_path: str, prob_map: Dict[str, float], k: int = 3) -> Dict:
    """Build a structured record for JSON/CSV output."""
    probs = ordered_probs(prob_map)
    top = topk_from_probs(prob_map, k)
    
    # Handle empty probability map (error case)
    if not top:
        pred_lbl, pred_p = "", 0.0
    else:
        pred_lbl, pred_p = top[0]["label"], top[0]["prob"]
    
    return OrderedDict([
        ("image", image_path),
        ("predicted_label", pred_lbl),
        ("confidence", float(pred_p)),
        ("topk", top),
        ("probs", probs),
    ])


def write_jsonl(file_handle, records: List[Dict]) -> None:
    """Write records as JSONL (one JSON object per line)."""
    for record in records:
        json.dump(record, file_handle, separators=(',', ':'))
        file_handle.write('\n')


def write_json_array(file_handle, records: List[Dict], pretty: bool = False) -> None:
    """Write records as a JSON array."""
    indent = 2 if pretty else None
    json.dump(records, file_handle, indent=indent, separators=(',', ':') if not pretty else None)


def write_csv_row(writer, record: Dict) -> None:
    """Write a single CSV row from a record."""
    row = [record["image"], record["predicted_label"], record["confidence"]]
    row += [record["probs"][cls] for cls in CLASS_NAMES]
    writer.writerow(row)


def get_output_format(output_path: Union[str, Path, None]) -> str:
    """Determine output format based on file extension."""
    if not output_path:
        return "jsonl"  # default to JSONL for stdout
    
    ext = Path(output_path).suffix.lower()
    if ext == ".json":
        return "json"
    elif ext == ".jsonl":
        return "jsonl"
    elif ext == ".csv":
        return "csv"
    else:
        return "jsonl"  # default to JSONL for unknown extensions
