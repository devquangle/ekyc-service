import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import base64
import time
from fastapi.testclient import TestClient
from main import app

print("Starting TestClient context...")
with TestClient(app) as client:
    print("TestClient started!")
    front_path = os.path.join("tests", "image", "cccd_c_mt.jpg")
    with open(front_path, "rb") as f:
        f_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

    t0 = time.time()
    print("Testing JSON payload...")
    r_json = client.post("/api/v1/ekyc/card", json={"front_image": f_b64})
    print(f"JSON completed in {time.time()-t0:.2f}s, status={r_json.status_code}, identityNumber={r_json.json().get('extractedData', {}).get('identityNumber')}")

    t0 = time.time()
    print("Testing multipart files...")
    with open(front_path, "rb") as f:
        r_files = client.post("/api/v1/ekyc/card", files={"front_image": ("card.jpg", f.read(), "image/jpeg")})
    print(f"Multipart completed in {time.time()-t0:.2f}s, status={r_files.status_code}, identityNumber={r_files.json().get('extractedData', {}).get('identityNumber')}")
