# Project Documentation: FaceID Pro (AI Attendance System)

## 📋 Overview
FaceID Pro is a full-stack, AI-powered attendance management system. It uses computer vision to detect and recognize faces in real-time through a webcam, automatically logging attendance into a database. It features a modern administrative dashboard for data visualization and user management.

---

## 🚀 Key Features
1. **Real-time Recognition**: 
   - Powered by OpenCV and Dlib.
   - Draws bounding boxes and labels around detected faces.
2. **User Enrollment**:
   - Register new users via live webcam capture or photo upload.
   - Automatically generates facial encodings for 128 different facial landmarks.
3. **Automated Logging**:
   - Matches live faces against registered encodings.
   - Logs 'In-Time' and 'Date' instantly upon recognition.
4. **Intruder Detection**:
   - Captures and saves photos of "Unknown" faces for security.
5. **Interactive Dashboard**:
   - Visual charts (Recharts) showing attendance trends.
   - Records page with search and export (CSV) functionality.

---

## 🛠 Tech Stack
| Tier | Technology | Use Case |
| :--- | :--- | :--- |
| **Frontend** | Next.js 15 (App Router) | Modern UI, Server-side rendering |
| **Styling** | Tailwind CSS + Framer Motion | Premium Glassmorphism & Animations |
| **Backend** | FastAPI (Python) | High-performance asynchronous API |
| **Computer Vision** | OpenCV + Face_Recognition | Image processing & ML recognition |
| **Database** | SQLite + SQLAlchemy | Lightweight SQL storage |
| **Data Processing**| Pandas + NumPy | Handling logs and numerical data |

---

## ⚙️ How it Works (System Workflow)
1. **Registration**: The user's face is captured → Processed via `face_recognition` library → 128-d vector embedding (encoding) is generated → Saved in the backend.
2. **Scanning**: The system runs a loop capturing frames from the webcam.
3. **Matching**: Each frame is compared against saved encodings using "Euclidean Distance". If it's below a threshold (usually 0.6), the face is "Recognized".
4. **Recording**: Once recognized, the backend checks the last entry; if the user hasn't marked attendance today, a new row is inserted in SQLite.
5. **Frontend Sync**: The Next.js frontend fetches logs via REST API and updates the UI instantly using WebSockets or periodic polling.

---

## 📂 Project Structure
- `frontend/`: Next.js application with Tailwind CSS.
- `backend/app/`: FastAPI server, database models, and recognition logic.
- `backend/static/encodings/`: Stores saved facial data.
- `backend/static/captures/`: Stores images of unknown visitors.
