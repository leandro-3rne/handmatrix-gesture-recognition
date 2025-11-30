import cv2
import os
import numpy as np

# --- KONFIGURATION ---
# Ordner, der vom C++ Data Collector erstellt wurde
INPUT_FOLDER = "training_data_raw"

# Zielordner für das Training (wird erstellt)
OUTPUT_FOLDER = "training_data_final"


def add_noise(image):
    row, col = image.shape
    mean = 0
    var = 10
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=0)


def augment_dataset():
    if not os.path.exists(INPUT_FOLDER):
        print(f"FEHLER: Der Ordner '{INPUT_FOLDER}' existiert nicht.")
        print("Bitte erst mit dem C++ Data Collector Bilder aufnehmen!")
        return

    print(f"--- STARTE AUGMENTATION ---")
    print(f"Quelle: {INPUT_FOLDER}")
    print(f"Ziel:   {OUTPUT_FOLDER}")

    count = 0

    # os.walk geht rekursiv durch alle Unterordner (0_faust, 1_peace, etc.)
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for filename in files:
            if not filename.endswith(".png"): continue

            # Pfade konstruieren
            input_path = os.path.join(root, filename)

            # Relativen Pfad (z.B. "0_faust") ermitteln, damit Struktur erhalten bleibt
            rel_path = os.path.relpath(root, INPUT_FOLDER)
            target_dir = os.path.join(OUTPUT_FOLDER, rel_path)

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            name_base = os.path.splitext(filename)[0]

            # 1. Original
            cv2.imwrite(os.path.join(target_dir, filename), img)

            # 2. Spiegeln
            img_flip = cv2.flip(img, 1)
            cv2.imwrite(os.path.join(target_dir, f"{name_base}_flip.png"), img_flip)

            # 3. Rotation
            for angle in [-10, 10]:
                img_rot = rotate_image(img, angle)
                cv2.imwrite(os.path.join(target_dir, f"{name_base}_rot{angle}.png"), img_rot)

            # 4. Noise
            img_noise = add_noise(img)
            cv2.imwrite(os.path.join(target_dir, f"{name_base}_noise.png"), img_noise)

            count += 5  # 1 Orig + 1 Flip + 2 Rot + 1 Noise

    print(f"--- FERTIG ---")
    print(f"Gesamtbilder: {count}")


if __name__ == "__main__":
    augment_dataset()