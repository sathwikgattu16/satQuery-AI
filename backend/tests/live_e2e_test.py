"""
backend/tests/live_e2e_test.py
Live End-to-End HTTP client testing against running FastAPI backend on http://127.0.0.1:8000.
Owner: Member 1
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def run_live_tests():
    print("=== LIVE END-TO-END VERIFICATION (PORT 8000) ===\n")
    
    # 0. Health check
    r0 = requests.get(f"{BASE_URL}/")
    print(f"0. Health Check -> Status {r0.status_code}")
    print(f"   Payload: {json.dumps(r0.json(), indent=2)}\n")
    assert r0.status_code == 200

    svg_sample = b"<svg xmlns='http://www.w3.org/2000/svg'><rect width='100' height='100'/></svg>"

    # 1. Single Image VQA
    files1 = {"image": ("isro_cartosat_scene.svg", svg_sample, "image/svg+xml")}
    data1 = {
        "question": "What are the primary land cover classifications present in this scene?",
        "task_hint": "vqa"
    }
    r1 = requests.post(f"{BASE_URL}/api/analyze", files=files1, data=data1)
    print(f"1. Single Image VQA -> Status {r1.status_code}")
    j1 = r1.json()
    print(f"   Task: {j1['task']}")
    print(f"   Answer: {j1['answer']}")
    print(f"   Confidence: {j1['confidence']}")
    print(f"   Models Used: {j1['execution_summary']['models_used']}")
    print(f"   Trace Steps: {len(j1['execution_summary']['trace_steps'])} stages logged\n")
    assert r1.status_code == 200
    assert j1["task"] == "vqa"

    # 2. Single Image Captioning
    files2 = {"image": ("isro_cartosat_scene.svg", svg_sample, "image/svg+xml")}
    data2 = {"task_hint": "caption"}
    r2 = requests.post(f"{BASE_URL}/api/analyze", files=files2, data=data2)
    print(f"2. Single Image Captioning -> Status {r2.status_code}")
    j2 = r2.json()
    print(f"   Task: {j2['task']}")
    print(f"   Answer: {j2['answer']}")
    print(f"   Models Used: {j2['execution_summary']['models_used']}\n")
    assert r2.status_code == 200
    assert j2["task"] == "caption"

    # 3. Bi-temporal Change Detection
    files3 = {
        "image": ("t1_baseline_2022.svg", svg_sample, "image/svg+xml"),
        "image_t2": ("t2_observation_2024.svg", svg_sample, "image/svg+xml")
    }
    data3 = {"question": "What urban expansion or deforestation changes occurred between T1 and T2?"}
    r3 = requests.post(f"{BASE_URL}/api/analyze", files=files3, data=data3)
    print(f"3. Bi-temporal Change -> Status {r3.status_code}")
    j3 = r3.json()
    print(f"   Task: {j3['task']}")
    print(f"   Answer: {j3['answer']}")
    print(f"   Visualization: {j3['visualization']['title']} ({j3['visualization']['type']})")
    print(f"   Metrics: {j3['visualization']['metrics']}\n")
    assert r3.status_code == 200
    assert j3["task"] == "change"

    # 4. Optical + SAR Multimodal Fusion
    files4 = {
        "image": ("optical_cloudy_rgb.svg", svg_sample, "image/svg+xml"),
        "sar": ("sar_polarimetric_radar.svg", svg_sample, "image/svg+xml")
    }
    data4 = {"question": "Identify maritime vessels hidden beneath optical cloud cover using SAR."}
    r4 = requests.post(f"{BASE_URL}/api/analyze", files=files4, data=data4)
    print(f"4. Optical + SAR Fusion -> Status {r4.status_code}")
    j4 = r4.json()
    print(f"   Task: {j4['task']}")
    print(f"   Answer: {j4['answer']}")
    print(f"   Visualization: {j4['visualization']['title']} ({j4['visualization']['type']})")
    print(f"   Metrics: {j4['visualization']['metrics']}\n")
    assert r4.status_code == 200
    assert j4["task"] == "multimodal_fusion"

    # 5. Invalid Request (Missing primary image)
    r5 = requests.post(f"{BASE_URL}/api/analyze", data={"question": "No file test"})
    print(f"5. Invalid Request (Missing Image) -> Status {r5.status_code}")
    print(f"   Detail: {r5.json().get('detail')}\n")
    assert r5.status_code in (400, 422)

    print("=== ALL 5 LIVE HTTP FLOWS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_live_tests()
