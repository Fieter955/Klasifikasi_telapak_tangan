# Gemini Context & Instructions (GEMINI.md)

File ini memberikan panduan teknis kepada AI (Gemini CLI) untuk memahami struktur dan aturan khusus dalam proyek ini.

## 📌 Arsitektur Proyek
- **Backend**: FastAPI (Python) melayani API di `/api/*` dan Static Frontend di `/`.
- **Port Default**: `8011`.
- **Frontend File**: `frontend/static.html` (di-mount ke root `/`).

## 🧠 Aturan Model (PENTING)
1. **Penyimpanan**: Model harus dalam bentuk file tunggal `.pth` di dalam folder `models/`.
2. **Format Internal**: File `.pth` harus berupa ZIP archive dengan folder root internal bernama `archive/`. Jika model rusak/terekstrak, gunakan script Python `zipfile` untuk membungkusnya kembali (Jangan gunakan PowerShell `Compress-Archive` karena masalah backslash).
3. **Mapping Arsitektur**:
   - `AlexNetCustom`: Mencari layer `features` dan `classifier`.
   - `EfficientNetB0`: Menggunakan blok `MBConvBlock` dan `stem`.
   - `MobileNetSmall`: Menggunakan blok `bottlenecks` dan `conv1`.
4. **Label**: Menggunakan 30 kelas (041 - 070). Mapping tersedia di `main.py` dalam variabel `LABELS_NAMED`.

## 🛠️ Alur Kerja AI
- Jika user melaporkan error "Missing key in state_dict", periksa apakah arsitektur yang dipilih sudah sesuai dengan model yang di-load.
- Jangan mengubah port `8011` di `main.py` kecuali diminta secara eksplisit.
- Selalu gunakan `FileResponse` untuk melayani `static.html`.

## ⚠️ Known Issues
- File model `restored_model.pth` adalah hasil restorasi manual. Jika terjadi error "Invalid Magic Number", file tersebut kemungkinan corrupt dan perlu di-copy ulang dari sumber asli (folder `output_full`).
