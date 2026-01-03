"""
Hardware-in-the-loop adapters for ingesting and validating time-tag streams.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Tuple
import csv
import json
from pathlib import Path
import socket

import numpy as np

from .timetags import TimeTags
from .coincidence import match_coincidences
from .timing import TimingModel


def ingest_timetag_file(path: str) -> Tuple[TimeTags, TimeTags]:
    """
    Ingest time tags from a JSON or CSV file into TimeTags for A/B detectors.
    """
    tag_path = Path(path)
    if not tag_path.exists():
        raise FileNotFoundError(f"Time-tag file not found: {path}")

    if tag_path.suffix.lower() == ".csv":
        with open(tag_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        with open(tag_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            rows = data.get("tags") or data.get("timetags")
        else:
            rows = data
        if not isinstance(rows, list):
            raise ValueError("Time-tag JSON must be a list or contain 'tags'.")

    tags_a: List[Dict[str, Any]] = []
    tags_b: List[Dict[str, Any]] = []
    for row in rows:
        detector = str(row.get("detector") or row.get("channel") or "A").upper()
        time_val = row.get("time_s")
        if time_val is None:
            time_val = row.get("time")
        entry = {
            "time_s": float(time_val),
            "is_signal": bool(int(row.get("is_signal", 1))) if isinstance(row.get("is_signal"), str) else bool(row.get("is_signal", True)),
            "basis": int(row.get("basis", 0)),
            "bit": int(row.get("bit", 0)),
        }
        if detector in ("A", "ALICE"):
            tags_a.append(entry)
        elif detector in ("B", "BOB"):
            tags_b.append(entry)
        else:
            raise ValueError(f"Unknown detector label: {detector}")

    return _to_timetags(tags_a), _to_timetags(tags_b)


def playback_pass(
    tags_a: TimeTags,
    tags_b: TimeTags,
    tau_seconds: float,
    timing_model: TimingModel | None = None,
    estimate_offset: bool = False,
) -> Dict[str, Any]:
    """
    Playback time tags through coincidence matching and derive metrics.
    """
    result = match_coincidences(
        tags_a=tags_a,
        tags_b=tags_b,
        tau_seconds=tau_seconds,
        timing_model=timing_model,
        estimate_offset=estimate_offset,
    )
    qber = _qber_from_matrices(result.matrices)
    return {
        "coincidences": int(result.coincidences),
        "accidentals": int(result.accidentals),
        "car": float(result.car),
        "qber_mean": qber,
        "matrices": result.matrices,
        "estimated_clock_offset_s": result.estimated_offset_s,
    }


def compute_validation_diffs(
    observed: Dict[str, Any],
    expected: Dict[str, Any],
    qber_abort_threshold: float,
) -> Dict[str, Any]:
    """Compute diffs between observed and expected metrics."""
    headroom_obs = qber_abort_threshold - observed.get("qber_mean", float("nan"))
    headroom_exp = qber_abort_threshold - expected.get("qber_mean", float("nan"))
    return {
        "observed": {
            "qber_mean": observed.get("qber_mean"),
            "car": observed.get("car"),
            "headroom": headroom_obs,
        },
        "expected": {
            "qber_mean": expected.get("qber_mean"),
            "car": expected.get("car"),
            "headroom": headroom_exp,
        },
        "diffs": {
            "qber_mean": _safe_delta(observed.get("qber_mean"), expected.get("qber_mean")),
            "car": _safe_delta(observed.get("car"), expected.get("car")),
            "headroom": _safe_delta(headroom_obs, headroom_exp),
        },
    }


def socket_read_lines(host: str, port: int, limit: int = 1000, timeout_s: float = 1.0) -> List[str]:
    """Minimal socket adapter for reading newline-delimited tag lines."""
    lines: List[str] = []
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        buf = b""
        while len(lines) < limit:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                lines.append(line.decode("utf-8", errors="ignore"))
                if len(lines) >= limit:
                    break
    return lines


def serial_read_lines(port: str, baudrate: int = 115200, limit: int = 1000, timeout_s: float = 1.0) -> List[str]:
    """Minimal serial adapter for reading newline-delimited tag lines."""
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise ImportError("pyserial is required for serial adapters.") from exc

    lines: List[str] = []
    with serial.Serial(port=port, baudrate=baudrate, timeout=timeout_s) as ser:
        for _ in range(limit):
            raw = ser.readline()
            if not raw:
                break
            lines.append(raw.decode("utf-8", errors="ignore").strip())
    return lines


def _to_timetags(rows: Iterable[Dict[str, Any]]) -> TimeTags:
    if not rows:
        return TimeTags(
            times=np.array([], dtype=float),
            is_signal=np.array([], dtype=bool),
            basis=np.array([], dtype=np.int8),
            bit=np.array([], dtype=np.int8),
        )
    times = np.array([float(r["time_s"]) for r in rows], dtype=float)
    order = np.argsort(times)
    return TimeTags(
        times=times[order],
        is_signal=np.array([bool(r["is_signal"]) for r in rows], dtype=bool)[order],
        basis=np.array([int(r["basis"]) for r in rows], dtype=np.int8)[order],
        bit=np.array([int(r["bit"]) for r in rows], dtype=np.int8)[order],
    )


def _qber_from_matrices(matrices: Dict[str, List[List[int]]]) -> float:
    total = 0
    errors = 0
    for counts in matrices.values():
        total += sum(sum(row) for row in counts)
        errors += counts[0][1] + counts[1][0]
    if total == 0:
        return float("nan")
    return float(errors / total)


def _safe_delta(value: Any, reference: Any) -> Any:
    if value is None or reference is None:
        return None
    return float(value) - float(reference)
