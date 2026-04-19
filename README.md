# 😷 Face Mask Detection System

A high-performance, real-time minimalist Face Mask Detection system built to quickly identify whether an individual is wearing a face mask. Featuring an ultra-modern 'glassmorphism' UI and a robust Python-based AI core, this project provides instant safety compliance checks.

## ✨ Features

- **Real-Time Mask Detection:** Streams webcam frames and processes them seamlessly with ultra-low latency.
- **Image Upload Analysis:** Audit static reference images through a localized deep learning API.
- **Premium User Interface:** A minimalist, portfolio-grade frontend built with Next.js and Tailwind CSS featuring dynamic color-coded result badges.
- **Microservice Architecture:** Independent frontend and backend layers ensuring scalability and separation of concerns.

## 🛠 Tech Stack

**Frontend:**
- [Next.js](https://nextjs.org/) (React Framework)
- [Tailwind CSS](https://tailwindcss.com/) (Styling & Glassmorphism)
- [Lucide React](https://lucide.dev/) (Icons)

**Backend:**
- [FastAPI](https://fastapi.tiangolo.com/) (Python API Wrapper)
- [OpenCV](https://opencv.org/) (Image Preprocessing & Face Detection)
- [TensorFlow / Keras](https://www.tensorflow.org/) (MobileNetV2 Mask Detection Model)

## 📸 Screenshots

> *(Insert a screenshot of the Next.js UI showing the `MASK DETECTED` green badge here)*
> 
> *(Insert a screenshot of the Next.js UI showing the `NO MASK` red badge here)*

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection
```

### 2. Backend Setup
```bash
cd backend
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Start the Fast API server
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
Open a new terminal window / tab.
```bash
cd frontend

# Install dependencies
npm install

# Start the Next.js development server
npm run dev
```

## 💻 Usage Instructions

1. Ensure both your backend (`localhost:8000`) and frontend (`localhost:3000`) are running.
2. Open your browser and navigate to `http://localhost:3000`.
3. Click **"Start Camera"** to initialize live scanning, or click **"Upload Photo"** to analyze a specific picture.
4. The system will process the image immediately and display a glowing Result Badge indicating Mask compliance.

## 📁 Folder Structure

```
face-mask-detection/
├── backend/                  # Python API Core
│   ├── app/
│   │   ├── main.py           # FastAPI entry point
│   │   └── detector.py       # Core OpenCV/TensorFlow logic
│   ├── data/                 # Caffe & Keras Models
│   └── requirements.txt      # Python dependencies
├── frontend/                 # Next.js Application
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx      # Unified minimalist UI
│   │       ├── layout.tsx
│   │       └── globals.css   # Global Tailwind styles
│   ├── package.json
│   └── tailwind.config.ts
└── README.md
```

## 🌟 Future Improvements
- Migration from HTTP Polling to WebRTC for even faster streaming.
- Incorporation of a lightweight edge model (.tflite) directly into the browser to reduce backend overhead.

## 👤 Author

Developed by **[Your Name/Nitin Kumar]**
- [LinkedIn](#)
- [GitHub](#)
- [Portfolio](#)
