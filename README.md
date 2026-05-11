# VisionID — Image Classification System

Sistem klasifikasi gambar berbasis Deep Learning (PyTorch) dengan antarmuka FastAPI dan Frontend Modern.

## 🚀 Fitur Utama
- **Multi-Architecture Support**: Mendukung AlexNet, EfficientNet-B0, dan MobileNet (Small/Large).
- **FastAPI Backend**: API yang cepat dengan dokumentasi otomatis (Swagger).
- **Modern UI**: Frontend elegan menggunakan Tailwind CSS dengan fitur drag-and-drop.
- **Real-time Inference**: Prediksi cepat dengan dukungan GPU/CUDA.
- **Top-5 Predictions**: Menampilkan 5 hasil prediksi teratas beserta skor kepercayaan.

## 🛠️ Struktur Proyek
- `main.py`: Entry point aplikasi FastAPI dan static file server.
- `model_def.py`: Definisi arsitektur model PyTorch (AlexNet, EfficientNet, MobileNet).
- `models/`: Folder tempat menyimpan file model `.pth`.
- `frontend/`: Folder berisi aset frontend (`static.html`).
- `code ratna/`: Script untuk augmentasi data dan training model.

## 📦 Instalasi
1. Pastikan Python 3.9+ sudah terinstall.
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jika menggunakan GPU, pastikan versi `torch` sesuai dengan CUDA Anda.

## 🚦 Cara Menjalankan
Jalankan perintah berikut di root direktori:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8011
```
Buka browser di: `http://localhost:8011`

## 🧠 Konfigurasi Model
Pastikan file `.pth` diletakkan di dalam folder `models/`. Jika model tersimpan dalam format folder (hasil ekstrak), gunakan script restorasi untuk membungkusnya kembali menjadi file tunggal yang kompatibel dengan `torch.load()`.
