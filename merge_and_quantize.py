import json, os, time, tempfile
import torch, torch.nn as nn
from PIL import Image
from peft import PeftConfig, PeftModel
from transformers import ViTForImageClassification, ViTImageProcessor

ADAPTER_DIR = "artifacts/vit-qlora-cifar100"
TEST_DIR    = f"{ADAPTER_DIR}/test"
OUT_DIR     = "model"


def size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / 1e6


def load_test(label2id):
    items = []
    for cls in sorted(os.listdir(TEST_DIR)):
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in sorted(os.listdir(cls_dir)):
            items.append((os.path.join(cls_dir, f), label2id[cls]))
    return items


@torch.inference_mode()
def evaluate(model, processor, items, bs=8):
    correct, t0 = 0, time.perf_counter()
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        x = processor(
            [Image.open(p).convert("RGB") for p, _ in chunk],
            return_tensors="pt")["pixel_values"]
        preds = model(pixel_values=x).logits.argmax(-1)
        correct += (preds == torch.tensor([y for _, y in chunk])).sum().item()
    return correct / len(items), (time.perf_counter() - t0) * 1000 / len(items)


def main():
    id2label = {int(k): v for k, v in
                json.load(open(f"{ADAPTER_DIR}/labels.json")).items()}
    label2id = {v: k for k, v in id2label.items()}

    base_id   = PeftConfig.from_pretrained(ADAPTER_DIR).base_model_name_or_path
    processor = ViTImageProcessor.from_pretrained(ADAPTER_DIR)

    # 1) merge LoRA adapter into FP32 base
    print("Loading base model and merging adapter...")
    base = ViTForImageClassification.from_pretrained(
        base_id, num_labels=len(id2label),
        id2label=id2label, label2id=label2id,
        torch_dtype=torch.float32)
    fp32 = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload().eval()

    # 2) dynamic INT8 quantization for CPU
    print("Quantizing to INT8...")
    int8 = torch.ao.quantization.quantize_dynamic(
        fp32, {nn.Linear}, dtype=torch.qint8).eval()

    # 3) compare
    items = load_test(label2id)
    print(f"\nEvaluating on {len(items)} test images...")
    fp32_acc, fp32_ms = evaluate(fp32, processor, items)
    int8_acc, int8_ms = evaluate(int8, processor, items)

    print(f"\n{'':6}{'size (MB)':>12}{'accuracy':>10}{'ms/img':>10}")
    print(f"{'FP32':6}{size_mb(fp32):12.1f}{fp32_acc:10.4f}{fp32_ms:10.1f}")
    print(f"{'INT8':6}{size_mb(int8):12.1f}{int8_acc:10.4f}{int8_ms:10.1f}")

    # 4) save
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(int8.state_dict(), f"{OUT_DIR}/model_int8.pt")
    fp32.config.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
