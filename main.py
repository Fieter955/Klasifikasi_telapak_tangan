"""
main.py — FastAPI Backend untuk Klasifikasi Gambar
===================================================
Jalankan dengan:
    uvicorn main:app --reload --host 0.0.0.0 --port 8011

Lalu buka browser: http://localhost:8011
"""

import io
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models as tmodels

from model_def import (
    AlexNetCustom,
    EfficientNetB0,
    MobileNetSmall,
    PREPROCESS,
)

# ===========================================================
# KONFIGURASI APLIKASI
# ===========================================================

NUM_CLASSES = 30
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Model default jika tidak ditentukan oleh frontend
DEFAULT_ARCH = "efficientnet"
DEFAULT_MODEL_FILE = "restored_model.pth"

# Label numerik: 041 s/d 070
LABELS_NUMERIC: list[str] = [f"{i:03d}" for i in range(41, 71)]

# Label nama orang Indonesia (acak — 30 nama)
LABELS_NAMED: list[str] = [
    "Andi Pratama",      # 041
    "Budi Santoso",      # 042
    "Citra Dewi",        # 043
    "Dian Rahayu",       # 044
    "Eko Purnomo",       # 045
    "Fitri Wulandari",   # 046
    "Gilang Ramadan",    # 047
    "Hendra Wijaya",     # 048
    "Indah Permata",     # 049
    "Joko Susilo",       # 050
    "Kartika Sari",      # 051
    "Lestari Ningrum",   # 052
    "Muhammad Rizki",    # 053
    "Novi Anggraeni",    # 054
    "Okta Firmansyah",   # 055
    "Putri Maharani",    # 056
    "Qori Handayani",    # 057
    "Rizal Fauzan",      # 058
    "Sinta Melinda",     # 059
    "Teguh Prasetyo",    # 060
    "Umi Kalsum",        # 061
    "Vina Octavia",      # 062
    "Wahyu Hidayat",     # 063
    "Xena Septiana",     # 064
    "Yoga Anggara",      # 065
    "Zahra Nurlita",     # 066
    "Agus Kurniawan",    # 067
    "Bella Kusuma",      # 068
    "Cahyo Nugroho",     # 069
    "Dewi Anggraini",    # 070
]

# Deteksi device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# Cache model agar tidak reload setiap request
_model_cache: dict = {}


# ===========================================================
# FUNGSI LOAD MODEL
# ===========================================================

def load_model(architecture: str, model_file: str) -> nn.Module:
    """
    Muat model dari file .pt / .pth.
    Hasil di-cache di memori agar cepat saat request berikutnya.
    """
    cache_key = f"{architecture}::{model_file}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(
            f"File model '{model_file}' tidak ditemukan di folder models/."
        )

    # Bangun arsitektur
    if architecture == "alexnet":
        model = AlexNetCustom(NUM_CLASSES)
    elif architecture == "efficientnet":
        model = EfficientNetB0(NUM_CLASSES)
    elif architecture == "mobilenet_small":
        model = MobileNetSmall(NUM_CLASSES)
    elif architecture == "mobilenet_large":
        model = tmodels.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, NUM_CLASSES
        )
    else:
        raise ValueError(f"Arsitektur tidak dikenal: '{architecture}'")

    # Muat bobot
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model_cache[cache_key] = model
    print(f"[INFO] Model '{model_file}' ({architecture}) berhasil dimuat.")
    return model


# ===========================================================
# FASTAPI APP
# ===========================================================

app = FastAPI(
    title="Klasifikasi Gambar API",
    description="Backend untuk sistem klasifikasi gambar berbasis PyTorch",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------------

@app.get("/api/health", tags=["Sistem"])
def health_check():
    """Cek status server dan device yang aktif."""
    return {
        "status": "ok",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ----------------------------------------------------------
# DAFTAR MODEL YANG TERSEDIA
# ----------------------------------------------------------

@app.get("/api/models", tags=["Model"])
def get_available_models():
    """
    Scan folder models/ dan kembalikan daftar file .pt / .pth
    beserta arsitektur yang mungkin cocok berdasarkan nama file.
    """
    files = []
    if MODELS_DIR.exists():
        for f in sorted(MODELS_DIR.iterdir()):
            if f.suffix.lower() in (".pt", ".pth"):
                # Tebak arsitektur dari nama file
                name_lower = f.stem.lower()
                if "alexnet" in name_lower:
                    suggested_arch = "alexnet"
                elif "efficientnet" in name_lower:
                    suggested_arch = "efficientnet"
                elif "mobilenet_small" in name_lower or "mobilenetsmall" in name_lower:
                    suggested_arch = "mobilenet_small"
                elif "mobilenet_large" in name_lower or "mobilenetlarge" in name_lower:
                    suggested_arch = "mobilenet_large"
                else:
                    # Default jika tidak terdeteksi dari nama
                    suggested_arch = DEFAULT_ARCH if f.name == DEFAULT_MODEL_FILE else None
                
                files.append({
                    "filename": f.name,
                    "suggested_architecture": suggested_arch,
                    "size_mb": round(f.stat().st_size / 1_048_576, 2),
                    "is_default": f.name == DEFAULT_MODEL_FILE
                })
    return {
        "models": files,
        "default_architecture": DEFAULT_ARCH,
        "default_model_file": DEFAULT_MODEL_FILE
    }


# ----------------------------------------------------------
# DAFTAR LABEL
# ----------------------------------------------------------

@app.get("/api/labels", tags=["Label"])
def get_labels():
    """Kembalikan mapping label numerik dan label nama."""
    return {
        "numeric": LABELS_NUMERIC,
        "named": LABELS_NAMED,
        "mapping": [
            {"index": i, "numeric": n, "named": k}
            for i, (n, k) in enumerate(zip(LABELS_NUMERIC, LABELS_NAMED))
        ],
    }


# ----------------------------------------------------------
# PREDIKSI
# ----------------------------------------------------------

@app.post("/api/predict", tags=["Prediksi"])
async def predict(
    file: UploadFile = File(..., description="Gambar yang akan diklasifikasi"),
    architecture: str = Form(None, description="Arsitektur: alexnet | efficientnet | mobilenet_small | mobilenet_large"),
    model_file: str = Form(None, description="Nama file .pt / .pth dalam folder models/"),
    true_label: str = Form(
        None,
        description="Label benar (opsional). Format: '041' atau 'Andi Pratama'. "
                    "Digunakan untuk menentukan apakah prediksi benar (hijau) atau salah (merah).",
    ),
):
    """
    Klasifikasi satu gambar menggunakan model yang dipilih.
    """

    # 1. Baca & validasi gambar
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")

    # 2. Gunakan default jika tidak ditentukan
    arch = (architecture or DEFAULT_ARCH).strip()
    m_file = (model_file or DEFAULT_MODEL_FILE).strip()

    # 3. Muat model
    try:
        model = load_model(arch, m_file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal memuat model: {exc}")

    # 3. Preprocessing & inferensi
    tensor = PREPROCESS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs: list[float] = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    pred_idx: int = int(probs.index(max(probs)))
    confidence: float = round(probs[pred_idx] * 100, 4)

    # 4. Cek kebenaran (opsional)
    is_correct = None
    if true_label and true_label.strip():
        tl = true_label.strip()
        if tl in LABELS_NUMERIC:
            is_correct = LABELS_NUMERIC.index(tl) == pred_idx
        elif tl in LABELS_NAMED:
            is_correct = LABELS_NAMED.index(tl) == pred_idx
        # Coba parsing integer langsung (misal user kirim "5")
        elif tl.isdigit():
            idx = int(tl)
            if 0 <= idx < NUM_CLASSES:
                is_correct = idx == pred_idx

    # 5. Top-5
    sorted_idx = sorted(range(NUM_CLASSES), key=lambda i: probs[i], reverse=True)
    top5 = [
        {
            "rank": rank + 1,
            "index": i,
            "label_numeric": LABELS_NUMERIC[i],
            "label_named": LABELS_NAMED[i],
            "probability": round(probs[i] * 100, 4),
        }
        for rank, i in enumerate(sorted_idx[:5])
    ]

    return JSONResponse({
        "predicted_index": pred_idx,
        "predicted_label_numeric": LABELS_NUMERIC[pred_idx],
        "predicted_label_named": LABELS_NAMED[pred_idx],
        "confidence": confidence,
        "is_correct": is_correct,
        "top5": top5,
    })


# ----------------------------------------------------------
# HAPUS CACHE (berguna saat model di-update)
# ----------------------------------------------------------

@app.delete("/api/cache", tags=["Sistem"])
def clear_cache():
    """Hapus semua model yang ter-cache di memori."""
    count = len(_model_cache)
    _model_cache.clear()
    return {"message": f"{count} model dihapus dari cache."}


# ===========================================================
# SERVE STATIC FRONTEND (HARUS PALING AKHIR)
# ===========================================================

frontend_dir = Path(__file__).parent / "frontend"

@app.get("/", tags=["Frontend"])
async def read_root():
    """Sajikan halaman utama (static.html)."""
    file_path = frontend_dir / "static.html"
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse({"error": "File static.html tidak ditemukan di folder frontend/"}, status_code=404)

if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir)), name="static")


# ===========================================================
# ENTRY POINT
# ===========================================================

if __name__ == "__main__":
    import uvicorn
    import os
    # Railway/Cloud biasanya memberikan port melalui environment variable PORT
    port = int(os.environ.get("PORT", 8011))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
