# 😷 MaskGuard AI - Face Mask Detection System

A high-accuracy (~95%+) real-time face mask detection system built with deep learning (MobileNetV2) and a modern glassmorphism dashboard.

## 🚀 Features
- **Real-time Detection:** Process webcam feed with minimal latency.
- **High Accuracy:** Uses MobileNetV2 Transfer Learning for robust classification.
- **Smart Dashboard:** Premium UI with glassmorphism, real-time stats, and violation logs.
- **Automated Alerts:** Snapshots captured and logged automatically for violations.
- **Data Export:** Capability to track compliance over time.

---

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, TensorFlow/Keras, OpenCV.
- **Frontend:** Next.js, Tailwind CSS, Lucide Icons.
- **Model:** MobileNetV2 (Pre-trained on ImageNet).

---

## 📂 Project Structure
```text
/backend
  /app
    main.py         # FastAPI WebSocket & API
    detector.py     # Inference Logic
  /data             # Models, Violations, & Dataset
  setup.py          # Setup script (Downloads Face Detector)
  train.py          # Training script
  requirements.txt
/frontend
  /src
    /app            # Next.js Pages
    /components     # UI Components
```

---

## 🏁 Quick Start

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python setup.py          # Downloads pre-trained face detector
```

### 2. Model Training (Optional/Required)
To train the mask detector with your own data:
1. Place images in `backend/data/dataset/with_mask` and `backend/data/dataset/without_mask`.
2. Run training:
```bash
python train.py
```
*Note: A pre-trained model path is expected at `backend/data/mask_detector.model`.*

### 3. Run the Backend
```bash
python -m app.main
```

### 4. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

---

## 🎨 UI Preview
- **Dark Mode:** Deep slate & indigo theme.
- **Glassmorphism:** Frosted glass panels with blur effects.
- **Live Feed:** Overlaid bounding boxes (Green for Mask, Red for No Mask).

---

## ⚠️ Important Notes
- Ensure your webcam is available and permissions are granted in the browser.
- The system is optimized for **MobileNetV2**, providing a balance between speed and accuracy.
- For production, deploy the backend behind an SSL/TLS proxy for camera access.
