import shutil
import random
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

import pillow_heif
pillow_heif.register_heif_opener()


# =========================
# DEVICE (GPU / CPU)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# =========================
# KONFIGURASI
# =========================
BASE_DIR = Path("D:/Ratna/Skripsi/Dataset")
DATASET_NAMES = ["DP"]

K = 5
SEED = 42
CLEAN_OUTPUT = True
USE_AUGMENTATION = True   # <<< ubah True jika nanti ingin augment

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic"}

random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# HELPER FUNCTION
# =========================
def is_image_file(p: Path) -> bool:
    # Tambahkan pengecekan untuk mengabaikan file yang dimulai dengan "._" (biasanya file metadata dari macOS)
    return p.is_file() and p.suffix.lower() in IMG_EXTS and not p.stem.startswith("._")

def save_as_jpg(src: Path, dst: Path):
    try:
        img = Image.open(src).convert("RGB")
        dst = dst.with_suffix(".jpg")
        img.save(dst, "JPEG", quality=95)
    except Exception as e:
        print(f"[WARNING] Gagal memproses gambar {src}: {e}. Melewati file ini.")

def build_file_list(dataset_root: Path):
    X, y, class_names = [], [], []

    for cdir in sorted([d for d in dataset_root.iterdir() if d.is_dir()]):
        imgs = [p for p in cdir.iterdir() if is_image_file(p)]
        if not imgs:
            continue

        class_names.append(cdir.name)
        for img in imgs:
            X.append(img)
            y.append(cdir.name)

    return X, y, class_names


def ensure_class_dirs(base: Path, class_names):
    for cname in class_names:
        (base / cname).mkdir(parents=True, exist_ok=True)


def copy_files(files, dst_base: Path, label_map):
    for f in files:
        class_dir = dst_base / label_map[f]
        class_dir.mkdir(parents=True, exist_ok=True)

        if f.suffix.lower() == ".heic":
            dst = class_dir / f.stem
            save_as_jpg(f, dst)
        else:
            shutil.copy2(f, class_dir / f.name)


# =========================
# (OPSIONAL) AUGMENTASI PYTORCH
# =========================
shear_deg = 5.7  # ~10 derajat, setara keras dengan ImageDataGenerator

AUGS = [
    ("rot", transforms.RandomRotation(30)),

    ("zoom_out", transforms.RandomAffine(
        degrees=0,
        scale=(0.8, 0.8)
    )),

    ("zoom_in", transforms.RandomAffine(
        degrees=0,
        scale=(1.2, 1.2)
    )),

    ("shift_x", transforms.RandomAffine(
        degrees=0,
        translate=(0.10, 0.0)
    )),

    ("shift_y", transforms.RandomAffine(
        degrees=0,
        translate=(0.0, 0.10)
    )),

    ("shear", transforms.RandomAffine(
        degrees=0,
        shear=shear_deg
    )),

    ("bright", transforms.ColorJitter(
        brightness=(0.8, 1.2)
    )),
]

def augment_folder(folder: Path):
    for cdir in folder.iterdir():
        if not cdir.is_dir():
            continue

        imgs = [
            p for p in cdir.iterdir()
            if is_image_file(p) and "_aug_" not in p.stem
        ]

        for img_path in imgs:
            img = Image.open(img_path).convert("RGB")

            for tag, transform in AUGS:
                aug_img = transform(img)
                out_name = f"{img_path.stem}_aug_{tag}{img_path.suffix}"
                aug_img.save(cdir / out_name)


# =========================
# PROSES K-FOLD
# =========================
for ds_name in DATASET_NAMES:
    src_root = BASE_DIR / ds_name
    out_root = BASE_DIR / f"{ds_name}_KFold"

    if CLEAN_OUTPUT and out_root.exists():
        shutil.rmtree(out_root)

    X, y, class_names = build_file_list(src_root)
    if not X:
        print(f"[SKIP] Dataset kosong: {src_root}")
        continue

    label_map = {p: lbl for p, lbl in zip(X, y)}

    print(f"\n=== {ds_name} | total gambar: {len(X)} | kelas: {len(set(y))} ===")

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_dir = out_root / f"fold_{fold_idx}"
        train_dir = fold_dir / "train"
        val_dir   = fold_dir / "valid"

        ensure_class_dirs(train_dir, class_names)
        ensure_class_dirs(val_dir, class_names)

        train_files = [X[i] for i in train_idx]
        val_files   = [X[i] for i in val_idx]

        copy_files(train_files, train_dir, label_map)
        copy_files(val_files, val_dir, label_map)

        print(f"[KFOLD] {ds_name} fold_{fold_idx}: train={len(train_files)}, valid={len(val_files)}")

        if USE_AUGMENTATION:
            augment_folder(train_dir)
            print(f"[AUG] fold_{fold_idx} train augmented")

print("\nSelesai. Dataset siap untuk training PyTorch.")