import os
import shutil
import random

# ==============================
# PATHS (IMPORTANT)
# ==============================
SOURCE_DIR = "data/malaria_dataset"          # where your original dataset is
BASE_DIR = "data/malaria_dataset_split"     # where the split will be saved

# ==============================
# SPLIT RATIOS
# ==============================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

CLASSES = ["Parasitized", "Uninfected"]

# ==============================
# CREATE FOLDERS
# ==============================
for split in ["Train", "Validation", "Test"]:
    for cls in CLASSES:
        path = os.path.join(BASE_DIR, split, cls)
        os.makedirs(path, exist_ok=True)

# ==============================
# SPLIT DATA
# ==============================
for cls in CLASSES:
    class_path = os.path.join(SOURCE_DIR, cls)

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    # Copy images
    for img in train_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(BASE_DIR, "Train", cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(BASE_DIR, "Validation", cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(BASE_DIR, "Test", cls, img)
        )

    print(f"✅ {cls} split done!")

print("\n🎉 Dataset successfully split into Train / Validation / Test!")