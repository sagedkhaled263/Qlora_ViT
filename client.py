import sys, requests

img_path = sys.argv[1]
url      = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/predict"

with open(img_path, "rb") as f:
    r = requests.post(url, files={"file": (img_path, f, "image/jpeg")})

print(r.status_code)
import json; print(json.dumps(r.json(), indent=2))
