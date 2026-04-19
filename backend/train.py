import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Hyperparameters
INIT_LR = 1e-4
FINE_TUNE_LR = 1e-5
EPOCHS = 40
BS = 32

def train_model(dataset_path="backend/data/dataset"):
    print("[INFO] Loading images...")
    imagePaths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                imagePaths.append(os.path.join(root, file))

    if not imagePaths:
        print("[ERROR] No images found. Check backend/data/dataset/ folder.")
        return

    data, labels = [], []
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(label)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)

    # Enhanced data augmentation for better generalization (reduces overfitting)
    aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest")

    # Load MobileNetV2 with ImageNet weights
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

    # Enhanced custom head for higher accuracy
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = BatchNormalization()(headModel) # Normalizes activations
    headModel = Dropout(0.5)(headModel) # Prevents overfitting
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    # STAGE 1: Train ONLY the custom head
    print("[INFO] Stage 1: Training the custom head...")
    for layer in baseModel.layers:
        layer.trainable = False

    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Callbacks for robust training
    checkpoint = ModelCheckpoint("backend/data/mask_detector.model", monitor="val_accuracy", 
                                verbose=1, save_best_only=True, mode="max")
    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, min_lr=1e-7)

    H1 = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        epochs=15, 
        callbacks=[early_stop, reduce_lr])

    # STAGE 2: Fine-tuning (Unfreeze deeper layers)
    print("[INFO] Stage 2: Fine-tuning MobileNetV2 blocks...")
    # Unfreeze the last 50 layers of MobileNetV2
    for layer in baseModel.layers[-50:]:
        layer.trainable = True

    # Re-compile with a MUCH lower learning rate for fine-tuning
    opt = Adam(learning_rate=FINE_TUNE_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    H2 = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        epochs=EPOCHS, 
        callbacks=[checkpoint, early_stop, reduce_lr])

    # Final Evaluation
    print("[INFO] Evaluating network after fine-tuning...")
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

if __name__ == "__main__":
    train_model()
