import cv2
import base64
import requests
import numpy as np
import time

def test_health():
    try:
        r = requests.get("http://localhost:8000/health")
        print("Health Status:", r.json())
    except Exception as e:
        print("Health Check Failed:", e)

def test_inference():
    # Create a dummy frame (black image)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "TEST", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    _, buffer = cv2.imencode('.jpg', frame)
    encoded = base64.b64encode(buffer).decode('utf-8')
    base64_data = f"data:image/jpeg;base64,{encoded}"
    
    # We can't easily test WebSockets with requests, but we can test the /recognize endpoint
    try:
        files = {'file': ('test.jpg', buffer.tobytes(), 'image/jpeg')}
        r = requests.post("http://localhost:8000/recognize", files=files)
        print("Recognize Status:", r.status_code)
        print("Recognize Result:", r.json())
    except Exception as e:
        print("Inference Test Failed:", e)

if __name__ == "__main__":
    test_health()
    test_inference()
