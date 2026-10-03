import numpy as np
import pandas as pd
import json
import time
import os
from typing import Optional

from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.metric_preset import DataDriftPreset

class ModelPerformanceMonitor:
    def __init__(
        self,
        n_classes: int,
        labels: list[str],
        log_path: str = "predictions_log.jsonl",
        window: int = 500,
        report_dir: str = "reports",
    ):
        self.n_classes  = n_classes
        self.labels     = labels
        self.log_path   = log_path
        self.window     = window
        self.report_dir = report_dir
        self.records: list[dict] = []

    @staticmethod
    def _entropy(probs: list[float]) -> float:
        p = np.array(probs)
        p = np.clip(p, 1e-9, 1.0)
        return float(-np.sum(p * np.log2(p)))

    def log_prediction(self, prediction: dict, true_label_idx: Optional[int] = None):
        record = {
            "timestamp":  time.time(),
            "prediction": prediction["label"],
            "label_idx":  prediction["label_idx"],
            "confidence": prediction["confidence"],
            "entropy":    self._entropy(prediction["probabilities"]),
            "target":     self.labels[true_label_idx] if true_label_idx is not None else None,
        }
        self.records.append(record)
        if len(self.records) > self.window:
            self.records.pop(0)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def summary(self) -> dict:
        if not self.records:
            return {}
        confidences = [r["confidence"] for r in self.records]
        entropies   = [r["entropy"]    for r in self.records]
        return {
            "n_predictions":       len(self.records),
            "mean_confidence":     float(np.mean(confidences)),
            "std_confidence":      float(np.std(confidences)),
            "mean_entropy":        float(np.mean(entropies)),
            "low_confidence_rate": float(np.mean([c < 0.5 for c in confidences])),
        }

    def alert_low_confidence(self, threshold: float = 0.5) -> bool:
        if not self.records:
            return False
        recent = self.records[-50:]
        return float(np.mean([r["confidence"] < threshold for r in recent])) > 0.3

    def evidently_classification_report(self) -> Optional[dict]:
        labeled = [r for r in self.records if r["target"] is not None]
        if len(labeled) < 20:
            return None
        df  = pd.DataFrame(labeled)[["target", "prediction", "confidence"]]
        mid = len(df) // 2
        report = Report(metrics=[ClassificationPreset()])
        report.run(reference_data=df.iloc[:mid].copy(), current_data=df.iloc[mid:].copy())
        os.makedirs(self.report_dir, exist_ok=True)
        report_path = os.path.join(self.report_dir, "model_report.html")
        report.save_html(report_path)
        result  = report.as_dict()
        quality = result["metrics"][0]["result"]["current"]
        return {
            "accuracy":    quality.get("accuracy"),
            "f1":          quality.get("f1"),
            "report_path": report_path,
        }

    def evidently_confidence_drift_report(self) -> Optional[dict]:
        if len(self.records) < 100:
            return None
        df  = pd.DataFrame(self.records)[["confidence", "entropy"]]
        mid = len(df) // 2
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df.iloc[:mid].copy(), current_data=df.iloc[mid:].copy())
        os.makedirs(self.report_dir, exist_ok=True)
        report_path = os.path.join(self.report_dir, "confidence_drift_report.html")
        report.save_html(report_path)
        result  = report.as_dict()
        dataset = result["metrics"][0]["result"]
        return {
            "confidence_drift_detected": dataset["dataset_drift"],
            "report_path":               report_path,
        }
