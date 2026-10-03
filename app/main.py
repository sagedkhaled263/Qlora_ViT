import io, os, time
from contextlib import asynccontextmanager

import numpy as np
import torch, torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

from monitoring.data_drift    import DataDriftMonitor
from monitoring.model_monitor import ModelPerformanceMonitor
from monitoring.infra_monitor import InfraMonitor

MODEL_DIR            = os.getenv("MODEL_DIR", "model")
DRIFT_CHECK_INTERVAL = 100

state            = {}
prediction_count = 0

def load_model():
    from peft import PeftConfig, PeftModel

    ADAPTER_DIR = "artifacts/vit-qlora-cifar100"

    config   = ViTConfig.from_pretrained(MODEL_DIR)
    id2label = {int(k): v for k, v in config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    base_id = PeftConfig.from_pretrained(ADAPTER_DIR).base_model_name_or_path

    print("Merging adapter...")
    base   = ViTForImageClassification.from_pretrained(
        base_id,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    merged = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload().eval()

    print("Quantizing to INT8...")
    int8 = torch.ao.quantization.quantize_dynamic(
        merged, {nn.Linear}, dtype=torch.qint8).eval()

    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    return int8, processor

def extract_features(x: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        outputs = state["model"](pixel_values=x, output_hidden_states=True)
    return outputs.hidden_states[-1][:, 0, :].cpu().numpy()[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, processor = load_model()
    state["model"]     = model
    state["processor"] = processor

    id2label = model.config.id2label
    labels   = [id2label[i] for i in range(len(id2label))]

    state["drift_monitor"] = DataDriftMonitor(reference_size=200)
    state["perf_monitor"]  = ModelPerformanceMonitor(n_classes=len(labels), labels=labels)
    state["infra_monitor"] = InfraMonitor(prometheus_port=8001)
    state["infra_monitor"].start()

    yield
    state.clear()


app = FastAPI(title="ViT QLoRA CIFAR-100 Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "num_classes": len(state["model"].config.id2label)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), true_label: int | None = None):
    global prediction_count
    state["infra_monitor"].record_request(success=True)

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        state["infra_monitor"].record_request(success=False)
        raise HTTPException(status_code=400, detail="Not a valid image")

    try:
        with state["infra_monitor"].latency_timer():
            t0    = time.perf_counter()
            x     = state["processor"](image, return_tensors="pt")["pixel_values"]
            with torch.inference_mode():
                probs = state["model"](pixel_values=x).logits.softmax(-1)[0]
            embedding = extract_features(x)
            ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        state["infra_monitor"].record_request(success=False)
        raise HTTPException(status_code=500, detail=str(e))

    id2label = state["model"].config.id2label
    top      = int(probs.argmax())

    prediction = {
        "label":         id2label[top],
        "label_idx":     top,
        "confidence":    round(float(probs[top]), 4),
        "probabilities": probs.tolist(),
        "top5": [
            {"label": id2label[i], "confidence": round(float(probs[i]), 4)}
            for i in probs.topk(5).indices.tolist()
        ],
        "inference_ms": round(ms, 1),
    }

    state["perf_monitor"].log_prediction(prediction, true_label_idx=true_label)
    state["drift_monitor"].add_production_sample(image, embedding, top)
    prediction_count += 1

    drift_result = None
    if prediction_count % DRIFT_CHECK_INTERVAL == 0:
        drift_result = state["drift_monitor"].check_drift()

    response           = dict(prediction)
    response["alerts"] = {
        "low_confidence": state["perf_monitor"].alert_low_confidence(),
        "data_drift":     drift_result["dataset_drift_detected"] if drift_result else False,
    }
    return response


@app.post("/reference")
async def add_reference(file: UploadFile = File(...), label: int = 0):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Not a valid image")
    x         = state["processor"](image, return_tensors="pt")["pixel_values"]
    embedding = extract_features(x)
    state["drift_monitor"].add_reference_sample(image, embedding, label)
    return {"status": "ok", "reference_count": len(state["drift_monitor"].ref_embeddings)}


@app.post("/monitoring/check")
def trigger_check():
    drift      = state["drift_monitor"].check_drift()
    cls_report = state["perf_monitor"].evidently_classification_report()
    conf_drift = state["perf_monitor"].evidently_confidence_drift_report()
    infra      = state["infra_monitor"].latest()
    return {
        "drift":            drift,
        "classification":   cls_report,
        "confidence_drift": conf_drift,
        "infra":            infra.__dict__ if infra else None,
    }


@app.get("/metrics/model")
def model_metrics():
    return state["perf_monitor"].summary()


@app.get("/metrics/infra")
def infra_metrics():
    snap = state["infra_monitor"].latest()
    return snap.__dict__ if snap else {}