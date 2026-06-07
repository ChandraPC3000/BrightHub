# 💡 The Data Statesman: 360° Business Intelligence Dashboard
### **Samchan Tech Production // Powered by Data, Big Data & AI for Indonesia 2045**

---

## 🚀 Overview Proyek
Dashboard ini dibangun sebagai sistem kecerdasan bisnis (*Business Intelligence*) terintegrasi untuk mengoptimalkan parameter **"Toko Sehat"** pada jaringan outlet **Bright Store & Bright Cafe** di bawah Direktorat Pemasaran Ritel Pertamina Patra Niaga. 

Sistem ini tidak hanya menyajikan data historis keuangan, melainkan menggabungkan pendekatan ekonometrika, analisis opini publik (*customer voice*), dan simulasi taktis dampak ekonomi dari kampanye pemasaran nasional dalam satu antarmuka interaktif.

---

## 📂 Arsitektur Fitur (3 Tab Utama)

### 📊 Tab 1: Financial & Sales Forecast
*   **Algoritma Time-Series:** Mengintegrasikan model **ARIMA(1,1,1)** dan **Holt-Winters Exponential Smoothing** untuk memproyeksikan *Revenue, EBITDA, dan EAT* hingga periode Januari 2030.
*   **Anti-Manis Policy:** Dilengkapi dengan uji stasioneritas *Augmented Dickey-Fuller* (ADF Test) dan otomatisasi *Differencing* (d=1) untuk mengeliminasi bias tren linear yang terlalu optimis, sehingga menghasilkan proyeksi fluktuatif yang jujur dan berbasis data riil.
*   **Manajemen Risiko:** Menggunakan batasan *80% Confidence Interval (Stochastic Error Margin)* pada *Lower Bound* untuk membantu manajemen menyusun perencanaan anggaran (*budgeting*) berbasis skenario terburuk (*worst-case scenario*).

### 💬 Tab 2: Customer Sentiment & Public Opinion Analytics
*   **Multi-Platform Data Gathering:** Mengolah data hasil *scraping* opini publik dari berbagai lini media sosial (X, Instagram, TikTok) dan YouTube Review.
*   **Business Intelligence Connectivity:** Menghubungkan sebab-akibat pergerakan grafik finansial dengan persepsi riil konsumen di lapangan (misal: analisis korelasi antara penurunan omzet dengan lonjakan sentimen negatif terkait isu *supply chain* atau pelayanan).
*   **Audit Opini Viral:** Menyediakan *data logging interaktif* untuk menyaring komentar-komentar yang memiliki tingkat *engagement* tinggi demi efisiensi evaluasi manajemen.

### 🏃 Tab 3: Strategic Campaign (Bright marathON / Bright ON)
*   **Demand Generation Simulator:** Menyajikan rekomendasi strategi aktivasi massa nasional dengan rute ikonik Jakarta (Monas ➔ Bundaran HI ➔ GBK) untuk merajut seluruh ekosistem *Own Brand* (Bright Store, Bright Cafe, Olimart, Bright Gas, dan MyPertamina).
*   **Dynamic ROI Calculator:** Simulator pengembalian investasi interaktif yang mengkalkulasi proyeksi *Gross GMV*, *Net Revenue PPN (15% take-rate)*, dan pertumbuhan pemicu pengguna baru (*Daily Active Users*) aplikasi MyPertamina berdasarkan parameter jumlah pengunjung.
*   **Decision Matrix Expansion:** Menyediakan opsi komparasi kebijakan taktis antara melakukan *Pilot Project (Tap-In di Pertamina Eco RunFest Holding)* atau mengeksekusi *Standalone Mega Event*.

---

## ⚙️ Spesifikasi Teknologi & Library
Aplikasi ini dikembangkan menggunakan bahasa pemrograman **Python 3.10+** dengan dependensi utama:
*   `streamlit` (Kerangka kerja antarmuka web interaktif)
*   `pandas` & `numpy` (Manipulasi, pembersihan, dan rekayasa data)
*   `statsmodels` (Uji statistik ADF, modeling ARIMA, dan Exponential Smoothing)
*   `matplotlib` & `seaborn` (Visualisasi grafik terintegrasi)
*   `scikit-learn` (Evaluasi metrik kesalahan model / RMSE)
*   `openpyxl` (Mesin pembaca berkas spreadsheet Excel)

---

## 💻 Panduan Instalasi Lokal & Penggunaan

1. **Clone Repositori:**
```bash
   git clone [https://github.com/username_lu/the-data-statesman-dashboard.git](https://github.com/username_lu/the-data-statesman-dashboard.git)
   cd the-data-statesman-dashboard
