import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
import os

def create_model():
    print("[INFO] Creating placeholder model architecture...")
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)
    
    data_dir = "backend/data"
    os.makedirs(data_dir, exist_ok=True)
    model_path = os.path.join(data_dir, "mask_detector.model")
    
    print(f"[INFO] Saving model to {model_path}...")
    model.save(model_path)
    print("[SUCCESS] Placeholder model generated.")

if __name__ == "__main__":
    create_model()
