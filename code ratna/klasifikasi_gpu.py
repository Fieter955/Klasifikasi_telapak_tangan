import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix

torch.backends.cudnn.benchmark = True

# =========================================================
# 1. KONFIGURASI GLOBAL
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # Tambahan untuk reproducibility di CUDA

DATA_ROOT = Path(r"D:/Ratna/Skripsi/Dataset")
DATASET_NAMES = ["G_KFold"]

K = 5
EPOCHS = 50
PATIENCE = 5

LR_LIST = [0.01]
BATCH_SIZES = [32, 64, 128]
OPTIMIZERS = ["adamw", "sgd"]

NUM_WORKERS = 2
OUTPUT_DIR = Path("output_full")
OUTPUT_DIR.mkdir(exist_ok=True)

USE_ALEXNET = True
USE_EFFICIENTNET = True
USE_MOBILENET_SMALL = True
USE_MOBILENET_LARGE = False  # Tambah Large

# =========================================================
# 2. DEVICE (FORCED GPU)
# =========================================================
def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU tidak tersedia! Pastikan CUDA terinstall dan GPU terdeteksi. Kode ini memaksa penggunaan GPU.")
    
    torch.cuda.set_device(0)  # Pastikan menggunakan GPU 0
    print("✅ CUDA AKTIF:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
    print(f"🔥 Device yang digunakan: {device}")
    return device

# =========================================================
# 3. CONFUSION MATRIX
# =========================================================
def plot_confusion_matrix(cm, class_names, title, save_path):
    acc = np.trace(cm) / np.sum(cm)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.title(f"{title}\nAccuracy={acc:.4f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# =========================================================
# 4. MODEL
# =========================================================

MODEL_SELECTION = []

if USE_ALEXNET:
    MODEL_SELECTION.append("alexnet")

if USE_EFFICIENTNET:
    MODEL_SELECTION.append("efficientnet")

if USE_MOBILENET_SMALL:
    MODEL_SELECTION.append("mobilenet_small")

if USE_MOBILENET_LARGE:
    MODEL_SELECTION.append("mobilenet_large")

# Helper modules for activations (diubah ke nn.Module agar bisa dipakai di Sequential)
class HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.relu6(x + 3) / 6

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Bottleneck block for MobileNet Small
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t, use_se=False, activation='RE'):
        super(Bottleneck, self).__init__()
        self.use_se = use_se
        self.stride = stride
        self.t = t
        self.activation = activation

        expanded = int(in_channels * t)  # Cast t (float) ke int untuk channel

        # Expansion
        if t != 1:
            self.expansion = nn.Sequential(
                nn.Conv2d(in_channels, expanded, 1, bias=False),
                nn.BatchNorm2d(expanded),
                nn.ReLU6(inplace=True) if activation == 'RE' else HardSwish()
            )
        else:
            self.expansion = None

        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded, expanded, 3, stride=stride, padding=1, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU6(inplace=True) if activation == 'RE' else HardSwish()
        )

        # SE
        if use_se:
            self.se = SEBlock(expanded)
        else:
            self.se = None

        # Projection
        self.projection = nn.Sequential(
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        if self.expansion:
            x = self.expansion(x)
        x = self.depthwise(x)
        if self.se:
            x = self.se(x)
        x = self.projection(x)
        if self.stride == 1 and x.shape[1] == residual.shape[1]:
            x += residual
        return x

# MobileNet Small
class MobileNetSmall(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetSmall, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )
        self.bottlenecks = nn.Sequential(
            Bottleneck(16, 16, stride=2, t=1, use_se=True, activation='RE'),  # 112x112x16
            Bottleneck(16, 16, stride=2, t=4, activation='RE'),  # 56x56x16
            Bottleneck(16, 24, stride=1, t=3, activation='RE'),  # 28x28x24
            Bottleneck(24, 24, stride=2, t=3, use_se=True, activation='RE'),  # 14x14x24
            Bottleneck(24, 40, stride=1, t=3, activation='HS', use_se=True),  # 14x14x40
            Bottleneck(40, 40, stride=1, t=3, activation='HS', use_se=True),  # 14x14x40
            Bottleneck(40, 40, stride=1, t=6, activation='HS', use_se=True),  # 14x14x40
            Bottleneck(40, 48, stride=1, t=2.5, activation='HS', use_se=True),  # 14x14x48 (kembali ke 2.5)
            Bottleneck(48, 48, stride=1, t=2.3, activation='HS', use_se=True),  # 14x14x48 (kembali ke 2.3)
            Bottleneck(48, 96, stride=1, t=2.3, activation='HS', use_se=True),  # 7x7x96 (kembali ke 2.3)
            Bottleneck(96, 96, stride=1, t=6, activation='HS', use_se=True),  # 7x7x96
            Bottleneck(96, 96, stride=1, t=6, activation='HS', use_se=True),  # 7x7x96
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            HardSwish()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(576, 1024, 1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# MBConv block for EfficientNet
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_connect_rate=0.0):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.se_ratio = se_ratio
        self.drop_connect_rate = drop_connect_rate

        # Expansion
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish()
            )
        else:
            self.expand_conv = None

        # Depthwise
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish()
        )

        # SE
        if se_ratio > 0:
            self.se = SEBlock(expanded_channels, reduction=int(1 / se_ratio))
        else:
            self.se = None

        # Projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        if self.expand_conv:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if self.se:
            x = self.se(x)
        x = self.project_conv(x)
        if self.stride == 1 and x.shape == identity.shape:
            if self.drop_connect_rate > 0:
                x = self.drop_connect(x, self.drop_connect_rate)
            x += identity
        return x

    def drop_connect(self, x, drop_connect_rate):
        keep_prob = 1.0 - drop_connect_rate
        batch_size = x.shape[0]
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        output = x / keep_prob * binary_tensor
        return output

# EfficientNetB0
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 3, 1, 1, 0.25),  # Block 1: input 32, output 16
            MBConvBlock(16, 24, 3, 2, 6, 0.25),  # Block 2: input 16, output 24
            MBConvBlock(24, 40, 5, 2, 6, 0.25),  # Block 3: input 24, output 40
            MBConvBlock(40, 80, 3, 2, 6, 0.25),  # Block 4: input 40, output 80
            MBConvBlock(80, 80, 5, 1, 6, 0.25),  # Block 5: input 80, output 80
            MBConvBlock(80, 112, 5, 2, 6, 0.25),  # Block 6: input 80, output 112
            MBConvBlock(112, 112, 5, 1, 6, 0.25),  # Block 7: input 112, output 112
            MBConvBlock(112, 112, 5, 1, 6, 0.25),  # Block 8: input 112, output 112
            MBConvBlock(112, 192, 5, 1, 6, 0.25),  # Block 9: input 112, output 192
            MBConvBlock(192, 192, 5, 1, 6, 0.25),  # Block 10: input 192, output 192
            MBConvBlock(192, 192, 5, 1, 6, 0.25),  # Block 11: input 192, output 192
            MBConvBlock(192, 192, 5, 1, 6, 0.25),  # Block 12: input 192, output 192
            MBConvBlock(192, 320, 3, 2, 6, 0.25),  # Block 13: input 192, output 320
            MBConvBlock(320, 1280, 1, 1, 6, 0.25),  # Block 14: input 320, output 1280
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, num_classes, 1, bias=True)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        return x

class AlexNetCustom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.5)  # Tambahkan dropout setelah max_pooling2d_2
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 500),
            nn.ReLU(),
            nn.Dropout(0.5),  # Tambahkan dropout setelah dense (ReLU)
            nn.Linear(500, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def build_model(name, num_classes):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if name == "alexnet":
        return AlexNetCustom(num_classes), tf

    if name == "efficientnet":
        # Menggunakan custom EfficientNetB0 sesuai spesifikasi
        return EfficientNetB0(num_classes), tf

    if name == "mobilenet_small":
        # Menggunakan custom MobileNetSmall sesuai spesifikasi
        return MobileNetSmall(num_classes), tf

    if name == "mobilenet_large":
        # Menggunakan MobileNet V3 Large standar dari PyTorch
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model, tf

    raise ValueError("Model tidak dikenal")

# =========================================================
# 5. OPTIMIZER
# =========================================================
def get_optimizer(name, params, lr):
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "adamw":
        return optim.AdamW(params, lr=lr)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)

# =========================================================
# 6. TRAINING (GRADIENT ACCUMULATION 16)
# =========================================================
ACCUM_STEPS = 16  # Gradient accumulation tetap 16 untuk semua batch sizes

def train_case(model, train_loader, valid_loader, device, optimizer, case_name, accum_steps):
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.3
    )

    best_acc = 0
    patience_counter = 0
    best_cm = None
    epoch_logs = []  # List untuk menyimpan log epoch

    # Path untuk menyimpan model terbaik
    model_save_path = OUTPUT_DIR / f"{case_name}_best_model.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            # Debug: Pastikan data di GPU
            assert x.device.type == 'cuda', f"Data training tidak di GPU: {x.device}"
            assert y.device.type == 'cuda', f"Label training tidak di GPU: {y.device}"
            
            loss = criterion(model(x), y)
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # ===== VALIDATION =====
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0

        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                # Debug: Pastikan data validasi di GPU
                assert x.device.type == 'cuda', f"Data validasi tidak di GPU: {x.device}"
                assert y.device.type == 'cuda', f"Label validasi tidak di GPU: {y.device}"
                
                out = model(x)
                val_loss += criterion(out, y).item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.argmax(1).cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        acc = np.trace(cm) / np.sum(cm)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{case_name} | Epoch {epoch} | Val Acc={acc:.4f} | Val Loss={val_loss:.4f} | LR={current_lr:.6f}")

        # Simpan log epoch ke list
        epoch_logs.append({
            "case_name": case_name,
            "epoch": epoch,
            "val_acc": acc,
            "lr": current_lr,
            "val_loss": val_loss
        })

        if acc > best_acc:
            best_acc = acc
            best_cm = cm
            patience_counter = 0
            # Simpan model terbaik
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 Model terbaik disimpan: {model_save_path}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("⛔ EarlyStopping")
            break

    return best_acc, best_cm, epoch_logs

# =========================================================
# 7. MAIN LOOP (DIPERBAIKI: Model -> Batch Size -> Optimizer -> Fold, dengan print rata-rata akurasi setelah semua fold)
# =========================================================
def main():
    device = get_device()
    print("🧠 Model yang digunakan:", MODEL_SELECTION)
    results = []
    epoch_logs_all = []  # List untuk semua epoch logs

    for dataset in DATASET_NAMES:
        for model_name in MODEL_SELECTION:
            for bs in BATCH_SIZES:
                for opt_name in OPTIMIZERS:
                    fold_accuracies = []  # List untuk menyimpan akurasi terbaik setiap fold
                    
                    for fold in range(1, K + 1):
                        train_dir = DATA_ROOT / dataset / f"fold_{fold}" / "train"
                        valid_dir = DATA_ROOT / dataset / f"fold_{fold}" / "valid"
                        classes = os.listdir(train_dir)

                        train_ds = datasets.ImageFolder(train_dir)
                        valid_ds = datasets.ImageFolder(valid_dir)

                        accum_steps = ACCUM_STEPS  # Tetap 16

                        for lr in LR_LIST:  # LR tetap satu, tapi loop tetap untuk fleksibilitas
                            model, tf = build_model(model_name, len(classes))
                            model.to(device)
                            # Debug: Pastikan model di GPU
                            assert next(model.parameters()).device.type == 'cuda', f"Model tidak di GPU: {next(model.parameters()).device}"
                            print(f"🔥 Model {model_name} on device: {next(model.parameters()).device}")

                            train_ds.transform = tf
                            valid_ds.transform = tf

                            train_loader = DataLoader(
                                train_ds, batch_size=bs,
                                shuffle=True, num_workers=NUM_WORKERS,
                                pin_memory=True  # Selalu True karena GPU dipaksa
                            )
                            valid_loader = DataLoader(
                                valid_ds, batch_size=bs,
                                shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True  # Selalu True karena GPU dipaksa
                            )

                            optimizer = get_optimizer(opt_name, model.parameters(), lr)
                            case = f"{dataset}_f{fold}_{model_name}_bs{bs}_{opt_name}"

                            acc, cm, epoch_logs = train_case(
                                model, train_loader, valid_loader,
                                device, optimizer, case, accum_steps
                            )

                            # Simpan akurasi fold ini
                            fold_accuracies.append(acc)

                            # Simpan epoch logs ke list global
                            epoch_logs_all.extend(epoch_logs)

                            plot_confusion_matrix(
                                cm, classes, case,
                                OUTPUT_DIR / f"{case}_cm.png"
                            )

                            results.append({
                                "dataset": dataset,
                                "fold": fold,
                                "model": model_name,
                                "batch_size": bs,
                                "optimizer": opt_name,
                                "accuracy": acc
                            })

                            # Simpan results ke CSV setiap selesai satu skenario
                            pd.DataFrame(results).to_csv(
                                OUTPUT_DIR / "results_full.csv", index=False
                            )
                            print(f"📊 Results disimpan ke CSV setelah skenario: {case}")

                            # Simpan epoch logs ke CSV setiap selesai satu skenario
                            pd.DataFrame(epoch_logs_all).to_csv(
                                OUTPUT_DIR / "epoch_logs.csv", index=False
                            )
                            print(f"📈 Epoch logs disimpan ke CSV setelah skenario: {case}")

                            # 🧹 BERSIH GPU
                            del model
                            del optimizer
                            torch.cuda.empty_cache()

                    # Setelah semua fold selesai untuk kombinasi ini, hitung dan print rata-rata akurasi
                    avg_acc = np.mean(fold_accuracies)
                    print(f"📈 Rata-rata akurasi terbaik {model_name} bs{bs} {opt_name} (5-fold): {avg_acc:.4f}")

if __name__ == "__main__":
    main()