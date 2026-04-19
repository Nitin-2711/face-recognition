# 📘 Face Mask Detection System Documentation

## 1. Introduction
The Face Mask Detection System is a high-performance computer vision platform designed to determine if an individual is wearing a protective face mask. Built with an emphasis on a premium user experience and efficient backend processing, it strips away unnecessary bloatware to focus entirely on fast, accurate compliance checking. 

## 2. System Architecture
The application runs on a clean **Client-Server Microservice Architecture**:
- **Client (Next.js):** Provides a modern, glassmorphism-themed UI. The client captures frames from the user's webcam or file system, packages them as binary blobs, and issues asynchronous requests to the AI Core.
- **Server (FastAPI):** A lightweight Python wrapper that intercepts incoming requests, decodes the HTTP attachments into numpy arrays, and passes them to the AI pipeline. 

## 3. How Mask Detection Works
The system follows a two-stage deep learning pipeline utilizing OpenCV and Keras:
1. **Face Localization:** When an image hits the server, a Single Shot MultiBox Detector (SSD) model (ResNet-10 architecture) searches the image for human faces, defining a bounding box.
2. **Mask Classification:** The detected facial region (ROI) is cropped, resized, and fed into a MobileNetV2 Neural Network. This classifier was trained to output two probabilities: `Mask` or `No Mask`.
3. The server then evaluates these probabilities and returns a simple Boolean response back to the client.

## 4. API Documentation

### `POST /detect`
**Description:** Analyzes an image and returns the mask status.  
**Content-Type:** `multipart/form-data`

**Input Format:**
- `file`: A binary image parameter (JPEG/PNG encoded).

**Response Format (JSON):**
```json
{
  "mask": true 
}
```
*(Returns `true` if a mask is detected, `false` if no mask is detected, or if no face is visible).*

## 5. Detailed Setup Guide

### Prerequisites
- Python 3.9+
- Node.js Version 18+

### Step-by-Step Installation
1. **Repository Setup:** Download or clone the project directory to your local file system.
2. **Backend Services:**
    - Navigate to `/backend`.
    - Install strictly required dependencies: `pip install fastapi uvicorn opencv-python tensorflow numpy python-multipart`.
    - Run the API on port `8000`: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`.
3. **Frontend Services:**
    - Navigate to `/frontend`.
    - Install React and Tailwind packages: `npm install`.
    - Boot the Next.js client on port `3000`: `npm run dev`.
4. Navigate to `http://localhost:3000` to interact with the system.

## 6. Troubleshooting

### Backend Not Running
**Symptom:** Terminal returns `Address already in use` or connection is refused.
**Resolution:** Verify port 8000 is open. Run `lsof -i :8000` on macOS/Linux and `kill -9 <PID>` to clear stuck Python processes. Ensure you are running the command from inside the `/backend` folder.

### Camera Not Working
**Symptom:** UI displays "Awaiting Source" indefinitely after clicking start.
**Resolution:** Modern browsers block camera access on unsecured HTTP connections unless accessed strictly via `localhost` or `127.0.0.1`. Ensure you access the app using `http://localhost:3000`. Check your browser permissions near the URL bar.

### No Detection Result
**Symptom:** Camera is running, but the Mask badge does not appear.
**Resolution:** 
1. Check the server terminal for errors. Confirm the Keras `.keras` and Caffe `.prototxt` models are successfully loaded upon startup.
2. Ensure `NEXT_PUBLIC_API_URL` exactly matches the backend host (defaults to `http://localhost:8000`).

## 7. Limitations
- **Occlusions:** Very thick frames or excessive hair covering the chin may yield false positives.
- **Lighting Reliability:** Extensively dark or back-lit environments will cause the SSD face detector to fail step 1, resulting in a `false` evaluation.

## 8. Future Scope
- **WebRTC Implementation:** Currently utilizing an HTTP Polling mechanism (every 1.5s). Translating edge processing to WebRTC WebSockets would deliver instantaneous hardware-level analysis.
- **WASM Acceleration:** Migrating the MobileNetV2 `.keras` model directly into the browser via TensorFlow.js or WebAssembly to eliminate server round-trip latency.
