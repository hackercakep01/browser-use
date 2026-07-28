# Deploying Browser-Use via Easypanel

Panduan ini mejelaskan cara men-deploy **Browser-Use** pada server Anda menggunakan **Easypanel** (Docker/Docker Swarm control panel).

---

## 🚀 Metode Deployment (Pilih Salah Satu)

### Metode 1: Deploy Menggunakan GitHub & Dockerfile (Rekomendasi)

1. **Buka Dashboard Easypanel** Anda.
2. Buat **Project** baru (misal: `browser-use`).
3. Tambahkan Service baru berjenis **App**.
4. Di bagian **Source**:
   - Pilih **GitHub**.
   - Masukkan Repository URL: `https://github.com/hackercakep01/browser-use` (atau repo fork Anda).
   - Branch: `main`.
5. Di bagian **Build**:
   - Build Type: **Dockerfile**.
   - Dockerfile Path: `Dockerfile`.
6. Di bagian **Environment Variables**:
   ```env
   PORT=8000
   HOST=0.0.0.0
   BROWSER_USE_API_KEY=your_browser_use_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```
7. Di bagian **Ports**:
   - Published Port: `8000` (atau port domain pilihan Anda).
   - Target Port: `8000`.
8. Di bagian **Volumes**:
   - Name: `browser-use-data`.
   - Mount Path: `/data`.
9. Klik **Deploy**.

---

### Metode 2: Deploy Menggunakan Template `easypanel.json`

1. Buka project Anda di Easypanel.
2. Pilih **Templates / Schema Import**.
3. Copy isi dari file [`easypanel.json`](./easypanel.json) di repository ini dan paste ke form Easypanel.
4. Masukkan API Key Anda di bagian Environment Variables.
5. Klik **Create & Deploy**.

---

### Metode 3: Deploy Menggunakan Docker Compose Service

1. Di Easypanel, buat service baru berjenis **Compose**.
2. Paste isi dari file [`docker-compose.yml`](./docker-compose.yml).
3. Atur Environment Variables (`BROWSER_USE_API_KEY`, `OPENAI_API_KEY`, dll.).
4. Klik **Deploy**.

---

## 🔀 Integrasi 9router / Custom OpenAI Endpoint

Browser-Use telah dilengkapi dengan dukungan penuh untuk **9router** dan Custom OpenAI-compatible API Gateway:

1. Buka Web UI Dashboard di `http://<your-easypanel-domain-or-ip>:8000/`.
2. Di bagian **LLM Model Provider**, pilih **9router / Custom OpenAI Compatible Endpoint**.
3. Masukkan **API Base URL** Anda (contoh: `https://terbaik-9router.3obhmi.easypanel.host/v1`).
4. Klik tombol **📥 Import Models** untuk secara otomatis mengambil dan mengimpor daftar model AI yang tersedia dari 9router Anda.
5. Pilih model yang diimpor dari dropdown dan jalankan tugas browser automation!

### Contoh Payload REST API dengan 9router:
```json
{
  "task": "Extract top posts summary from Hacker News",
  "llm_provider": "9router",
  "api_base_url": "https://terbaik-9router.3obhmi.easypanel.host/v1",
  "model_name": "gpt-4o",
  "api_key": "your_9router_api_key_optional"
}
```

---

## 📊 Logging Permanen, Filter Tanggal & Export JSON

Seluruh log eksekusi disimpan secara permanen di direktori `/data/logs/` yang terhubung dengan persistent volume `browser-use-data`.

- **Filter Tanggal & Status**:
  Gunakan kontrol Start Date, End Date, dan Status pada Web UI atau kirim query parameter pada API:
  `GET /api/v1/tasks?start_date=2026-07-01&end_date=2026-07-28&status=completed`
- **Download JSON Log**:
  Klik tombol **📥 Download JSON** pada Web UI atau akses endpoint API:
  `GET /api/v1/tasks/export?start_date=2026-07-01&end_date=2026-07-28`

---

## 🔍 Verification & Health Check

Setelah status service berubah menjadi **Healthy**:
- **Health Check Endpoint**: `http://<your-easypanel-domain-or-ip>:8000/health`
- **Web UI Dashboard**: Buka `http://<your-easypanel-domain-or-ip>:8000/` di browser untuk mengakses Dashboard Browser-Use interaktif.
- **Import Models API Endpoint**: `POST /api/v1/models`
- **Run Task API Endpoint**: `POST /api/v1/run`
- **Task Log Search API Endpoint**: `GET /api/v1/tasks`
- **JSON Log Export API Endpoint**: `GET /api/v1/tasks/export`

---

## ⚡ Performa & Production Recommendation

- **ChatBrowserUse**: Direkomendasikan sebagai model utama karena kecepatan, akurasi tinggi, dan token cost terendah untuk tugas browser automation.
- **Browser Use Cloud (`use_cloud=True`)**:
  Untuk menghindari deteksi bot/captcha dan meningkatkan kecepatan eksekusi di server cloud, tambahkan `BROWSER_USE_API_KEY` dan aktifkan `use_cloud: true` di payload API atau konfigurasi `Browser(use_cloud=True)`.
