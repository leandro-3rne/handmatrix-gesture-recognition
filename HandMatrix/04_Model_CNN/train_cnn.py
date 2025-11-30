import os
import shutil
import sys
import numpy as np
import cv2

# ==========================================
# KONFIGURATION
# ==========================================
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Wir importieren tf_keras DIREKT als keras
# Das verhindert Verwechslungen mit Keras 3
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, models
import tf2onnx

# --- EINSTELLUNGEN ---
IMG_SIZE = 32
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 5

INPUT_DIR = "training_data_final"
ONNX_OUTPUT_PATH = "hand_cnn.onnx"


# --- DATEN LADEN ---
def load_dataset(directory):
    print(f"--- Lade Daten aus '{directory}' ---")
    images = []
    labels = []

    if not os.path.exists(directory):
        print(f"FEHLER: Ordner '{directory}' nicht gefunden!")
        return np.array([]), np.array([])

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            try:
                if '_' not in folder_name: continue
                label_part = folder_name.split('_')[0]
                if not label_part.isdigit(): continue
                label_id = int(label_part)
                if label_id < 0 or label_id >= NUM_CLASSES: continue

                print(f"Lade Klasse {label_id} ({folder_name})...")
                for img_name in os.listdir(folder_path):
                    if img_name.endswith(".png"):
                        path = os.path.join(folder_path, img_name)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                            images.append(img)
                            labels.append(label_id)
            except ValueError:
                continue

    if not images:
        print("FEHLER: Keine Bilder gefunden!")
        return np.array([]), np.array([])

    X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    y = np.array(labels)
    print(f"✅ Fertig! {len(X)} Bilder geladen.")
    return X, y


# --- HAUPTPROGRAMM ---
if __name__ == "__main__":
    # Sicherheitcheck
    print(f"Keras Version: {keras.__version__} (sollte mit .tf enden oder tf_keras sein)")

    X, y = load_dataset(INPUT_DIR)

    if len(X) > 0:
        # Modell definieren (Via tf_keras)
        model = models.Sequential([
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input_image"),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES, activation='softmax', name="output_class")
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("\n--- Starte Training ---")
        model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.1)

        print("\n--- Exportiere ONNX ---")

        # Da wir jetzt echtes tf_keras nutzen, geht der direkte Export!
        input_signature = [tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 1], tf.float32, name="input_image")]

        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13
        )

        with open(ONNX_OUTPUT_PATH, "wb") as f:
            f.write(model_proto.SerializeToString())

        print(f"\n✅ ERFOLG! '{ONNX_OUTPUT_PATH}' ist fertig.")
        print("Kopiere sie jetzt in deinen C++ Projektordner (C++ HandMatrix CNN (Externes Training)).")

    else:
        print("Keine Daten.")