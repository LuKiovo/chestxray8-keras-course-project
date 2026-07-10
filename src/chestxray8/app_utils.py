"""Helpers shared by the Streamlit demo app and tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chestxray8.constants import DISEASE_LABELS


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    rows = []
    per_label = metrics.get("per_label", {})
    for label in DISEASE_LABELS:
        item = per_label.get(label, {})
        rows.append(
            {
                "label": label,
                "auc": item.get("auc"),
                "f1": item.get("f1"),
                "accuracy": item.get("accuracy"),
                "positive_count": item.get("positive_count", 0),
                "threshold": item.get("threshold"),
                "validation_f1": item.get("validation_f1"),
            }
        )
    return pd.DataFrame(rows)


def prediction_to_dataframe(prediction: dict) -> pd.DataFrame:
    labels = prediction.get("labels", [])
    df = pd.DataFrame(labels)
    if df.empty:
        return pd.DataFrame(columns=["label", "probability", "predicted"])
    columns = ["label", "probability", "predicted"]
    if "threshold" in df:
        columns.append("threshold")
    return df[columns]


def top_predictions(prediction: dict, n: int = 5) -> list[dict]:
    labels = prediction.get("labels", [])
    return sorted(labels, key=lambda item: item.get("probability", 0), reverse=True)[:n]


def load_training_log(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch" in df:
        df["_epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.dropna(subset=["_epoch"]).drop_duplicates(subset="_epoch", keep="last").sort_values("_epoch")
        df["epoch"] = df["_epoch"].astype(int) + 1
        df = df.drop(columns="_epoch")
    else:
        df.insert(0, "epoch", range(1, len(df) + 1))
    return df
