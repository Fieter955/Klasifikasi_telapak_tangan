"""
main.py — FastAPI Backend untuk Klasifikasi Gambar
===================================================
Jalankan dengan:
    uvicorn main:app --reload --host 0.0.0.0 --port 8011

Lalu buka browser: http://localhost:8011
"""

import time
import io
import base64
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pillow_heif import register_heif_opener
from torchvision import models as tmodels

# Registrasi agar Pillow bisa membaca file HEIC/HEIF
# Catatan: Ini tidak memperlambat JPG karena Pillow hanya memanggil decoder HEIF 
# jika magic number file sesuai dengan format HEIF/HEIC.
register_heif_opener()

# Optimasi CUDA jika tersedia
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

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
# HELPER: OPTIMASI PREVIEW
# ===========================================================

def generate_preview_base64(img: Image.Image, max_size=(400, 400)) -> str:
    """
    Membuat preview Base64 yang ringan dengan meresize gambar terlebih dahulu.
    Ini SANGAT mengurangi waktu pemrosesan untuk gambar resolusi tinggi.
    """
    # Gunakan copy agar tidak merusak image asli yang akan di-preprocess
    preview = img.copy()
    preview.thumbnail(max_size)
    
    buffered = io.BytesIO()
    # JPEG quality 70 sudah cukup untuk preview dan lebih cepat
    preview.save(buffered, format="JPEG", quality=70)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


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
    version="1.1.0",
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
# KONVERSI GAMBAR (untuk preview HEIC)
# ----------------------------------------------------------

@app.post("/api/convert", tags=["Sistem"])
async def convert_image(
    file: UploadFile = File(..., description="Gambar untuk dikonversi ke JPEG")
):
    """
    Menerima gambar apa pun (termasuk HEIC) dan mengembalikan Base64 JPEG.
    Digunakan agar frontend bisa menampilkan preview HEIC secara ringan.
    """
    contents = await file.read()
    try:
        start_time = time.time()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_data_uri = generate_preview_base64(img)
        
        print(f"[DEBUG] /api/convert: {round(time.time() - start_time, 4)}s")
        return JSONResponse({"image_data_uri": img_data_uri})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal mengonversi gambar: {e}")


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
    t_start = time.time()

    # 1. Baca & validasi gambar
    contents = await file.read()
    t_read = time.time()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")
    t_open = time.time()

    # 1a. Konversi gambar ke base64 secara OPTIMAL (Resize dulu)
    img_data_uri = generate_preview_base64(img)
    t_preview = time.time()

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
    t_load = time.time()

    # 4. Preprocessing & inferensi
    tensor = PREPROCESS(img).unsqueeze(0).to(device)
    t_preprocess = time.time()
    
    with torch.no_grad():
        logits = model(tensor)
        # Ensure probs is always a list of floats, handling 1D or 2D logits
        probs_tensor = torch.softmax(logits, dim=1).squeeze()
        if probs_tensor.dim() == 0:
            probs = [probs_tensor.item()]
        else:
            probs = probs_tensor.cpu().tolist()
    t_inference = time.time()

    pred_idx: int = int(probs.index(max(probs)))
    confidence: float = round(probs[pred_idx] * 100, 4)
    
    # Logging performa untuk diagnosa user
    print(f"--- PERFORMANCE LOG ---")
    print(f"Read File    : {round(t_read - t_start, 4)}s")
    print(f"Open Image   : {round(t_open - t_read, 4)}s")
    print(f"Preview Gen  : {round(t_preview - t_open, 4)}s (Optimized)")
    print(f"Load Model   : {round(t_load - t_preview, 4)}s (Cached)")
    print(f"Preprocess   : {round(t_preprocess - t_load, 4)}s")
    print(f"Inference    : {round(t_inference - t_preprocess, 4)}s")
    print(f"TOTAL        : {round(t_inference - t_start, 4)}s")
    print(f"-----------------------")

    # 5. Cek kebenaran (opsional)
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
            if 0 <= idx < len(LABELS_NUMERIC):
                is_correct = idx == pred_idx

    # 6. Top-5 & All Probabilities
    num_probs = len(probs)
    sorted_idx = sorted(range(num_probs), key=lambda i: probs[i], reverse=True)
    
    def get_label_numeric(i):
        return LABELS_NUMERIC[i] if i < len(LABELS_NUMERIC) else f"Unknown({i})"
    
    def get_label_named(i):
        return LABELS_NAMED[i] if i < len(LABELS_NAMED) else f"Unknown Label {i}"

    top5 = [
        {
            "rank": rank + 1,
            "index": i,
            "label_numeric": get_label_numeric(i),
            "label_named": get_label_named(i),
            "probability": round(probs[i] * 100, 4),
        }
        for rank, i in enumerate(sorted_idx[:5])
    ]

    all_probs = [
        {
            "index": i,
            "label_numeric": get_label_numeric(i),
            "label_named": get_label_named(i),
            "probability": round(probs[i] * 100, 4),
        }
        for i in range(num_probs)
    ]

    return JSONResponse({
        "predicted_index": pred_idx,
        "predicted_label_numeric": get_label_numeric(pred_idx),
        "predicted_label_named": get_label_named(pred_idx),
        "confidence": confidence,
        "is_correct": is_correct,
        "top5": top5,
        "all_probs": all_probs,
        "image_data_uri": img_data_uri,
        "performance": {
            "total_time": round(t_inference - t_start, 4),
            "inference_time": round(t_inference - t_preprocess, 4)
        }
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
