# Bright Hub: 360° Business Intelligence Dashboard
### **Sistem Informasi Proyeksi Finansial dan Manajemen Risiko Direktorat Pemasaran Ritel**

---

## 📈 Deskripsi Proyek
Bright Hub Dashboard adalah sebuah sistem aplikasi berbasis kecerdasan bisnis (*Business Intelligence*) yang dirancang secara khusus untuk melakukan monitoring, evaluasi, dan proyeksi tingkat kesehatan jaringan usaha non-fuel retail di bawah PT Pertamina Patra Niaga. 

Aplikasi ini mengintegrasikan pendekatan analitik runtun waktu (*time-series forecasting*), analisis kualitatif sentimen pasar konsumen, serta model kalkulasi efisiensi pengembalian investasi (*Return on Investment*) program pemasaran ke dalam satu platform pelaporan interaktif.

---

## 📂 Struktur Fungsi Utama Modul

### 1. Modul Proyeksi Penjualan & Finansial
*   **Implementasi Ekonometrika:** Menggunakan pemodelan algoritma statistika **ARIMA(1,1,1)** dan **Holt-Winters Exponential Smoothing** untuk menyusun proyeksi indikator keuangan (*Revenue, EBITDA, EAT*) dengan target estimasi jangka panjang.
*   **Metodologi Anti-Bias (Realistik):** Menerapkan pengujian statistik *Augmented Dickey-Fuller* (ADF Test) serta penyesuaian nilai *Differencing* (d=1) guna mengeliminasi bias tren pertumbuhan linear yang tidak realistis pada data retail.
*   **Mitigasi Risiko Perencanaan Anggaran:** Menyediakan visualisasi area ketidakpastian berbasis *80% Confidence Interval (Stochastic Error Margin)* untuk memfasilitasi manajemen dalam penyusunan target penganggaran berdasarkan skenario terburuk (*worst-case scenario*).

### 2. Modul Analisis Sentimen Konsumen
*   **Pemetaan Komparatif Lintas Saluran:** Mengompilasi data opini konsumen dari berbagai platform media digital untuk mengidentifikasi variabel kepuasan publik.
*   **Root Cause Analysis:** Menghubungkan visualisasi fluktuasi grafik keuangan internal dengan faktor eksternal berupa tingkat komplain publik (misal: analisis hubungan antara penurunan transaksi harian dengan kendala distribusi produk/supply chain).

### 3. Modul Strategi Kampanye Pemasaran (Bright ON)
*   **Demand Generation Simulator:** Menyediakan visualisasi usulan lintasan lari (Monas – Bundaran HI – GBK) sebagai pusat aktivasi promosi ekosistem *Own Brand* (*Bright Store, Bright Cafe, Olimart, Bright Gas*).
*   **Interactive ROI Calculator:** Alat simulator pengambil keputusan finansial untuk menghitung proyeksi perputaran uang (*Gross GMV*), estimasi pendapatan bagi hasil korporasi, serta percepatan akuisisi pengguna harian aktif aplikasi transaksi digital.

---

## 💻 Panduan Penggunaan Sistem Secara Lokal

1. **Unduh Repositori Dokumen:**
```bash
   git clone [https://github.com/username_anda/bright-hub-dashboard.git](https://github.com/username_anda/bright-hub-dashboard.git)
   cd bright-hub-dashboard
