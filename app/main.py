import io, os, time
from contextlib import asynccontextmanager

import torch, torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

MODEL_DIR = os.getenv("MODEL_DIR", "model")
state = {}


def load_model():
    config = ViTConfig.from_pretrained(MODEL_DIR)
    model  = ViTForImageClassification(config)
    model  = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8)
    sd = torch.load(f"{MODEL_DIR}/model_int8.pt",
                    map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    return model.eval(), ViTImageProcessor.from_pretrained(MODEL_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"], state["processor"] = load_model()
    yield
    state.clear()


app = FastAPI(title="ViT QLoRA CIFAR-100 Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "num_classes": len(state["model"].config.id2label),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Not a valid image")

    t0 = time.perf_counter()
    x  = state["processor"](image, return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        probs = state["model"](pixel_values=x).logits.softmax(-1)[0]
    ms = (time.perf_counter() - t0) * 1000

    id2label = state["model"].config.id2label
    top      = int(probs.argmax())

    return {
        "label":        id2label[top],
        "confidence":   round(float(probs[top]), 4),
        "top5": [
            {"label": id2label[i], "confidence": round(float(probs[i]), 4)}
            for i in probs.topk(5).indices.tolist()
        ],
        "inference_ms": round(ms, 1),
    }
