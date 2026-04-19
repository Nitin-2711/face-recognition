import os
import requests

def download_file(url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = url.split("/")[-1]
    filepath = os.path.join(folder, filename)
    
    if os.path.exists(filepath):
        print(f"[INFO] {filename} already exists.")
        return

    print(f"[INFO] Downloading {filename}...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"[INFO] {filename} downloaded.")
    else:
        print(f"[ERROR] Failed to download {filename}.")

def setup():
    # Face Detector Files (Caffe model)
    face_proto = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    face_model = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    detector_folder = "backend/data/face_detector"
    download_file(face_proto, detector_folder)
    download_file(face_model, detector_folder)
    
    # Violation folder
    os.makedirs("backend/data/violations", exist_ok=True)
    
    # Dataset placeholders
    os.makedirs("backend/data/dataset/with_mask", exist_ok=True)
    os.makedirs("backend/data/dataset/without_mask", exist_ok=True)
    
    print("[SUCCESS] Setup complete. Please put your training images in backend/data/dataset/ and run train.py")

if __name__ == "__main__":
    setup()
