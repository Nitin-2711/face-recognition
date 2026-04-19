import os
import requests
import bz2

def download_file(url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = url.split("/")[-1]
    filepath = os.path.join(folder, filename)
    
    if os.path.exists(filepath):
        print(f"[INFO] {filename} already exists.")
        return filepath

    print(f"[INFO] Downloading {filename}...")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"[INFO] {filename} downloaded.")
        return filepath
    else:
        print(f"[ERROR] Failed to download {filename}.")
        return None

def setup():
    # Face Detector Files
    face_proto = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    face_model = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    data_dir = "backend/data"
    detector_folder = os.path.join(data_dir, "face_detector")
    download_file(face_proto, detector_folder)
    download_file(face_model, detector_folder)
    
    # Facial Landmarks for Liveness (Blink Detection)
    landmarks_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_path = download_file(landmarks_url, data_dir)
    
    if bz2_path and bz2_path.endswith(".bz2"):
        print("[INFO] Decompressing landmarks...")
        dat_path = bz2_path.replace(".bz2", "")
        if not os.path.exists(dat_path):
            with bz2.BZ2File(bz2_path) as fr, open(dat_path, "wb") as fw:
                shutil_copyfileobj(fr, fw)
            print("[INFO] Landmarks decompressed.")
        os.remove(bz2_path)

    # Directories
    os.makedirs(os.path.join(data_dir, "violations"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "profiles"), exist_ok=True)
    
    print("[SUCCESS] Production setup complete.")

import shutil
def shutil_copyfileobj(fsrc, fdst, length=16*1024):
    while True:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)

if __name__ == "__main__":
    setup()
