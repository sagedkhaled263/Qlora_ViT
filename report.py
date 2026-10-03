# experiment.py
import requests, io, json
from pathlib import Path
from torchvision.datasets import CIFAR100

BASE_URL    = "http://localhost:8000"
TEST_DIR    = "artifacts/vit-qlora-cifar100/test"
LABELS_JSON = "artifacts/vit-qlora-cifar100/labels.json"

# load id2label and invert it
id2label = {int(k): v for k, v in json.load(open(LABELS_JSON)).items()}
label2id = {v: k for k, v in id2label.items()}

# ── 1. Load reference from CIFAR-100 train set ────────────────────────────
print("Downloading CIFAR-100 training set...")
dataset = CIFAR100(root="./data", train=True, download=True)

seen  = {}
total = 0
for image, label in dataset:
    if seen.get(label, 0) >= 2:
        continue
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    requests.post(
        f"{BASE_URL}/reference",
        files={"file": ("img.jpg", buf, "image/jpeg")},
        params={"label": label},
    )
    seen[label] = seen.get(label, 0) + 1
    total += 1
    if total >= 200:
        break
print(f"  {total} reference samples loaded")

# ── 2. Send test set as production ────────────────────────────────────────
print("\nSending production images (test set)...")
total = 0
for class_dir in sorted(Path(TEST_DIR).iterdir()):
    if not class_dir.is_dir():
        continue
    class_name = class_dir.name
    if class_name not in label2id:
        print(f"  Skipping unknown class: {class_name}")
        continue
    true_label = label2id[class_name]
    for img_path in class_dir.glob("*.jpg"):
        with open(img_path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/predict",
                files={"file": (img_path.name, f, "image/jpg")},
                params={"true_label": true_label},
            )
        total += 1
        result = r.json()
        print(f"  [{total}] {class_name} → {result['label']} ({result['confidence']:.2f})")

print(f"\n  {total} production predictions done")

# ── 3. Trigger full Evidently report ─────────────────────────────────────
print("\nGenerating reports...")
r      = requests.post(f"{BASE_URL}/monitoring/check")
result = r.json()

print("\n=== DRIFT ===")
if result["drift"]:
    d = result["drift"]
    print(f"  Detected      : {d['dataset_drift_detected']}")
    print(f"  Drifted cols  : {d['n_drifted_features']}")
    print(f"  Drift share   : {d['drift_share']:.2%}")
    print(f"  HTML report   : {d['report_path']}")
else:
    print("  Not enough data yet")

print("\n=== MODEL PERFORMANCE ===")
if result["classification"]:
    c = result["classification"]
    print(f"  Accuracy : {c['accuracy']}")
    print(f"  F1       : {c['f1']}")
    print(f"  HTML     : {c['report_path']}")
else:
    print("  Not enough labeled data yet")

print("\n=== CONFIDENCE DRIFT ===")
if result["confidence_drift"]:
    print(f"  Detected : {result['confidence_drift']['confidence_drift_detected']}")
    print(f"  HTML     : {result['confidence_drift']['report_path']}")