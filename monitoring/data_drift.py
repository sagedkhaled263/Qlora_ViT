import numpy as np
import pandas as pd
import json
import time
import os
from PIL import Image
from typing import Optional
from sklearn.decomposition import PCA

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric


class DataDriftMonitor:
    def __init__(
        self,
        reference_size: int = 200,
        n_pca_components: int = 50,
        report_dir: str = "reports",
        log_path: str = "drift_log.jsonl",
    ):
        self.reference_size   = reference_size
        self.n_pca_components = n_pca_components
        self.report_dir       = report_dir
        self.log_path         = log_path

        self.ref_embeddings:  list[np.ndarray] = []
        self.ref_pixel_means: list[float]      = []
        self.ref_labels:      list[int]        = []

        self.prod_embeddings:  list[np.ndarray] = []
        self.prod_pixel_means: list[float]      = []
        self.prod_labels:      list[int]        = []

        self.pca: Optional[PCA] = None
        self.is_reference_set   = False

    def _pixel_mean(self, image: Image.Image) -> float:
        arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        return float(arr.mean())

    def add_reference_sample(self, image: Image.Image, embedding: np.ndarray, label: int):
        self.ref_embeddings.append(embedding)
        self.ref_pixel_means.append(self._pixel_mean(image))
        self.ref_labels.append(label)
        if len(self.ref_embeddings) >= self.reference_size:
            self._fit_pca()
            self.is_reference_set = True

    def add_production_sample(self, image: Image.Image, embedding: np.ndarray, pred_label: int):
        self.prod_embeddings.append(embedding)
        self.prod_pixel_means.append(self._pixel_mean(image))
        self.prod_labels.append(pred_label)
        if len(self.prod_embeddings) > self.reference_size:
            self.prod_embeddings.pop(0)
            self.prod_pixel_means.pop(0)
            self.prod_labels.pop(0)

    def _fit_pca(self):
        ref_matrix = np.array(self.ref_embeddings)
        n = min(self.n_pca_components, ref_matrix.shape[0], ref_matrix.shape[1])
        self.pca = PCA(n_components=n)
        self.pca.fit(ref_matrix)

    def _build_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        ref_pca  = self.pca.transform(np.array(self.ref_embeddings))
        prod_pca = self.pca.transform(np.array(self.prod_embeddings))
        cols     = [f"emb_{i}" for i in range(ref_pca.shape[1])]

        ref_df              = pd.DataFrame(ref_pca, columns=cols)
        ref_df["pixel_mean"] = self.ref_pixel_means
        ref_df["label"]      = self.ref_labels

        prod_df               = pd.DataFrame(prod_pca, columns=cols)
        prod_df["pixel_mean"] = self.prod_pixel_means
        prod_df["label"]      = self.prod_labels

        return ref_df, prod_df

    def check_drift(self) -> Optional[dict]:
        if not self.is_reference_set or len(self.prod_embeddings) < self.reference_size:
            return None

        ref_df, prod_df = self._build_dataframes()

        report = Report(metrics=[DatasetDriftMetric(), DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=prod_df)

        os.makedirs(self.report_dir, exist_ok=True)
        report_path = os.path.join(self.report_dir, "drift_report.html")
        report.save_html(report_path)

        result  = report.as_dict()
        dataset = result["metrics"][0]["result"]

        summary = {
            "timestamp":              time.time(),
            "dataset_drift_detected": dataset["dataset_drift"],
            "n_drifted_features":     dataset["number_of_drifted_columns"],
            "drift_share":            dataset["share_of_drifted_columns"],
            "report_path":            report_path,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

        return summary
