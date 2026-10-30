import json
from pathlib import Path

import requests

file_path = Path("/home/thangnm1/project/ocr_gpkd/data/GP.DKKD -KVMT/CAI LẬY/2857929 - HKD DIEM MY.pdf")

with file_path.open("rb") as file:
    response = requests.post(
        "http://localhost:8010/ocr",
        files={"file": (file_path.name, file)},
        data={"template": "auto"},
        timeout=600,
    )

if not response.ok:
    print(f"Lỗi {response.status_code}: {response.text}")
    response.raise_for_status()

result = response.json()
Path("ket_qua.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, indent=2))