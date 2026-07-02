from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
ANALYTICS_FILE = BASE_DIR / "data" / "analysis_history.csv"


ANALYTICS_COLUMNS = [
    "analysis_id",
    "timestamp",
    "selected_modules",
    "persons",
    "cars",
    "buses",
    "trucks",
    "motorcycles",
    "bicycles",
    "traffic_lights",
    "stop_signs",
    "total_objects",
    "total_vehicles",
    "total_people",
    "average_detection_confidence",
    "processing_time",
    "traffic_density",
    "traffic_confidence",
    "weather_label",
    "weather_confidence",
    "scene_label",
    "scene_confidence",
]


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "None"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int:
    return int(round(_safe_number(value, 0.0)))


def append_analysis_record(
    summary: dict[str, Any],
    selected_modules: list[str],
) -> None:
    """
    Save one completed VisionAIHub analysis.

    Call this after the model predictions are added to `summary`
    and before rendering results.html.
    """
    ANALYTICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "analysis_id": summary.get("analysis_id", ""),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selected_modules": ",".join(selected_modules),
        "persons": _safe_int(summary.get("persons")),
        "cars": _safe_int(summary.get("cars")),
        "buses": _safe_int(summary.get("buses")),
        "trucks": _safe_int(summary.get("trucks")),
        "motorcycles": _safe_int(summary.get("motorcycles")),
        "bicycles": _safe_int(summary.get("bicycles")),
        "traffic_lights": _safe_int(summary.get("traffic_lights")),
        "stop_signs": _safe_int(summary.get("stop_signs")),
        "total_objects": _safe_int(summary.get("total_objects")),
        "total_vehicles": _safe_int(summary.get("total_vehicles")),
        "total_people": _safe_int(summary.get("total_people")),
        "average_detection_confidence": round(
            _safe_number(summary.get("avg_confidence")) * 100,
            2,
        ),
        "processing_time": round(
            _safe_number(summary.get("processing_time")),
            3,
        ),
        "traffic_density": summary.get("ml_traffic_density", ""),
        "traffic_confidence": _safe_number(
            summary.get("ml_traffic_confidence")
        ),
        "weather_label": summary.get("weather_label", ""),
        "weather_confidence": _safe_number(
            summary.get("weather_confidence")
        ),
        "scene_label": summary.get("scene_label", ""),
        "scene_confidence": _safe_number(
            summary.get("scene_confidence")
        ),
    }

    write_header = not ANALYTICS_FILE.exists()

    with ANALYTICS_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=ANALYTICS_COLUMNS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(record)


def _read_records() -> list[dict[str, str]]:
    if not ANALYTICS_FILE.exists():
        return []

    with ANALYTICS_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def _most_common(
    records: list[dict[str, str]],
    field: str,
) -> str:
    values = [
        row.get(field, "").strip()
        for row in records
        if row.get(field, "").strip()
    ]

    if not values:
        return "N/A"

    return Counter(values).most_common(1)[0][0]


def _distribution(
    records: list[dict[str, str]],
    field: str,
) -> dict[str, int]:
    values = [
        row.get(field, "").strip()
        for row in records
        if row.get(field, "").strip()
    ]

    return dict(Counter(values))


def _average(
    records: list[dict[str, str]],
    field: str,
) -> float:
    values = [
        _safe_number(row.get(field))
        for row in records
        if row.get(field, "") not in ("", None)
    ]

    if not values:
        return 0.0

    return sum(values) / len(values)


def build_analytics_summary() -> dict[str, Any]:
    records = _read_records()

    total_analyses = len(records)

    object_fields = [
        "persons",
        "cars",
        "buses",
        "trucks",
        "motorcycles",
        "bicycles",
        "traffic_lights",
        "stop_signs",
    ]

    object_distribution = {
        field.replace("_", " ").title(): sum(
            _safe_int(row.get(field))
            for row in records
        )
        for field in object_fields
    }

    analyses_over_time: Counter[str] = Counter()

    for row in records:
        timestamp = row.get("timestamp", "")

        try:
            date_label = datetime.fromisoformat(
                timestamp
            ).strftime("%d %b")
        except ValueError:
            date_label = "Unknown"

        analyses_over_time[date_label] += 1

    total_objects = sum(
        _safe_int(row.get("total_objects"))
        for row in records
    )

    total_vehicles = sum(
        _safe_int(row.get("total_vehicles"))
        for row in records
    )

    total_people = sum(
        _safe_int(row.get("total_people"))
        for row in records
    )

    chart_data = {
        "analyses_over_time": dict(analyses_over_time),
        "object_distribution": object_distribution,
        "traffic_distribution": _distribution(
            records,
            "traffic_density",
        ),
        "weather_distribution": _distribution(
            records,
            "weather_label",
        ),
        "scene_distribution": _distribution(
            records,
            "scene_label",
        ),
        "average_confidence": {
            "YOLO": round(
                _average(
                    records,
                    "average_detection_confidence",
                ),
                2,
            ),
            "Traffic": round(
                _average(
                    records,
                    "traffic_confidence",
                ),
                2,
            ),
            "Weather": round(
                _average(
                    records,
                    "weather_confidence",
                ),
                2,
            ),
            "Scene": round(
                _average(
                    records,
                    "scene_confidence",
                ),
                2,
            ),
        },
        "people_vehicles": {
            "People": total_people,
            "Vehicles": total_vehicles,
        },
    }

    return {
        "total_analyses": total_analyses,
        "total_objects": total_objects,
        "total_vehicles": total_vehicles,
        "average_objects": round(
            total_objects / total_analyses,
            2,
        ) if total_analyses else 0,
        "most_common_traffic": _most_common(
            records,
            "traffic_density",
        ),
        "most_common_weather": _most_common(
            records,
            "weather_label",
        ),
        "most_common_scene": _most_common(
            records,
            "scene_label",
        ),
        "average_processing_time": round(
            _average(records, "processing_time"),
            2,
        ),
        "chart_data": chart_data,
    }
