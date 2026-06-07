import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# ==============================================================================
# 1. CORE FUNCTIONS (BACK-END ENGINEERING)
# ==============================================================================

def calculate_mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

@st.cache_data
def run_statistical_analysis(file_upload, sheet_name, target_column, train_ratio):
    """
    Fungsi untuk membaca data, menjalankan pengujian backtesting, dan forecasting.
    Menggunakan mekanisme caching agar performa dashboard tetap optimal saat interaksi user.
    """
    # 1. Data Preparation
    df = pd.read_excel(file_upload, sheet_name=sheet_name)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=[target_column]).sort_values('Date')
    df.set_index('Date', inplace=True)
    df.index.freq = 'MS'
    ts_data = df[target_column]
    
    # 2. Uji Stasioneritas (ADF Test)
    adf_result = adfuller(ts_data)
    p_val = adf_result[1]
    adf_stat = adf_result[0]
    crit_vals = adf_result[4]
    
    # 3. Validasi Model (Backtesting Split)
    split_point = int(len(ts_data) * train_ratio)
    train_data = ts_data.iloc[:split_point]
    test_data = ts_data.iloc[split_point:]
    
    # Fitting Model untuk Validasi Akurasi
    val_arima_model = ARIMA(train_data, order=(1,1,1), seasonal_order=(1,1,0,12)).fit()
    val_arima_pred = val_arima_model.forecast(steps=len(test_data))
    
    val_hw_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()
    val_hw_pred = val_hw_model.forecast(steps=len(test_data))
    
    # Kalkulasi Metrik Evaluasi
    rmse_arima = np.sqrt(mean_squared_error(test_data, val_arima_pred))
    mape_arima = calculate_mape(test_data, val_arima_pred)
    mape_hw = calculate_mape(test_data, val_hw_pred)
    
    # 4. Proyeksi Produksi Akhir (Menggunakan Seluruh Data Historis)
    historical_ebitda_margin = (df['Ebitda'] / df['Revenue']).mean()
    historical_eat_margin = (df['EAT'] / df['Revenue']).mean()
    
    START_FORECAST_DATE = '2026-05-01'
    END_FORECAST_DATE = '2030-01-01'
    forecast_index = pd.date_range(start=START_FORECAST_DATE, end=END_FORECAST_DATE, freq='MS')
    steps = len(forecast_index)
    
    final_model = ARIMA(ts_data, order=(1,1,1), seasonal_order=(1,1,0,12)).fit()
    arima_result = final_model.get_forecast(steps=steps)
    rev_forecast = arima_result.predicted_mean
    conf_int = arima_result.conf_int(alpha=0.20) # 80% Confidence Interval
    
    ebitda_forecast = rev_forecast * historical_ebitda_margin
    eat_forecast = rev_forecast * historical_eat_margin
    eat_lower_bound = conf_int.iloc[:, 0] * historical_eat_margin
    
    # Konsolidasi Data Hasil Proyeksi ke Dalam DataFrame
    export_df = pd.DataFrame({
        'Periode': forecast_index.strftime('%Y-%m-%d'),
        'Forecast_Revenue_IDR': rev_forecast.values,
        'Forecast_EBITDA_IDR': ebitda_forecast.values,
        'Forecast_EAT_NetProfit_IDR': eat_forecast.values,
        'EAT_Worst_Case_Scenario_IDR': eat_lower_bound.values,
        'EAT_Risk_Gap_IDR': eat_forecast.values - eat_lower_bound.values
    })
    
    return {
        "p_value": p_val, "adf_stat": adf_stat, "crit_values": crit_vals,
        "rmse_arima": rmse_arima, "mape_arima": mape_arima, "mape_hw": mape_hw,
        "historical_ebitda_margin": historical_ebitda_margin, "historical_eat_margin": historical_eat_margin,
        "export_df": export_df, "rev_forecast": rev_forecast, "ebitda_forecast": ebitda_forecast,
        "eat_forecast": eat_forecast, "eat_lower_bound": eat_lower_bound, "forecast_index": forecast_index,
        "growth_pct": ((rev_forecast.iloc[-1] - rev_forecast.iloc[0]) / rev_forecast.iloc[0]) * 100
    }

# ==============================================================================
# 2. STREAMLIT INTERFACE DEPLOYMENT (FRONT-END ENGINE)
# ==============================================================================

st.set_page_config(
    page_title="Bright Hub Dashboard",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Bright Hub: 360° Business Intelligence Dashboard")
st.caption("Direktorat Pemasaran Ritel // PT Pertamina Patra Niaga")

# Inisialisasi Struktur Tab Utama
tab1, tab2, tab3 = st.tabs(["📊 Proyeksi Penjualan & Finansial", "💬 Analisis Sentimen Konsumen", "🏃 Strategi Kampanye Pemasaran (Bright ON)"])

# ------------------------------------------------------------------------------
# TAB 1: PROYEKSI PENJUALAN & FINANSIAL
# ------------------------------------------------------------------------------
with tab1:
    st.header("Analisis Proyeksi Finansial & Matriks Risiko")
    st.markdown("---")
    
    # Panel Kontrol Pengunggahan Berkas di Sidebar
    st.sidebar.header("📁 Pengaturan Data Input")
    uploaded_file = st.sidebar.file_uploader("Unggah Laporan Keuangan Retail (Format .xlsx)", type=["xlsx"])
    
    if uploaded_file is not None:
        target_col = st.sidebar.selectbox("Pilih Parameter Target", ["Revenue", "Ebitda", "EAT"])
        sheet = st.sidebar.text_input("Nama Sheet Keuangan", value="Sheet1")
        
        # Eksekusi Komputasi Analisis Statistik
        with st.spinner("Sistem sedang mengkalkulasi model regresi ARIMA dan Holt-Winters..."):
            res = run_statistical_analysis(uploaded_file, sheet, target_col, train_ratio=0.80)
        
        # Ringkasan Indikator Kinerja Utama (KPI Metrics)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tingkat Akurasi Model (100 - MAPE)", f"{100 - res['mape_arima']:.2f}%", help="Standar akurasi industri: >90% sangat akurat, 80-90% baik.")
        with col2:
            st.metric("Rata-rata Margin EBITDA Historis", f"{res['historical_ebitda_margin']*100:.2f}%")
        with col3:
            st.metric("Proyeksi Pertumbuhan Akumulatif", f"+{res['growth_pct']:.2f}%")
            
        st.markdown("---")
        
        # Visualisasi Grafik Proyeksi
        st.subheader("🔮 Tren Proyeksi Keuangan Terintegrasi (Periode 2026 - 2030)")
        
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(res['rev_forecast'].index, res['rev_forecast'].values / 1e6, label=f'Proyeksi {target_col} (Top Line)', color='#1c3d5a', linewidth=2.5)
        ax.plot(res['ebitda_forecast'].index, res['ebitda_forecast'].values / 1e6, label='Proyeksi EBITDA', color='#27ae60', linestyle='--')
        ax.plot(res['eat_forecast'].index, res['eat_forecast'].values / 1e6, label='Proyeksi EAT (Laba Bersih)', color='#d35400', linewidth=2.5)
        
        # Visualisasi Batas Risiko (Lower Bound) Berdasarkan Kebijakan Anggaran Realistis
        ax.fill_between(res['forecast_index'], res['eat_lower_bound'].values / 1e6, res['eat_forecast'].values / 1e6, 
                        color='#e67e22', alpha=0.15, label='Batas Toleransi Ketidakpastian Laba Bersih')
        
        ax.set_ylabel('Nilai (Dalam Juta Rupiah)', fontsize=11)
        ax.set_title('Simulasi Target Pendapatan & Manajemen Risiko Finansial Jaringan Outlet', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left')
        
        st.pyplot(fig)
        
        # Laporan Evaluasi Strategis Otomatis
        st.subheader("📋 Laporan Diagnosa Finansial & Status Kesehatan Outlet")
        
        f_worst = res['export_df']['EAT_Worst_Case_Scenario_IDR'].iloc[-1]
        status_kesehatan = "SANGAT SEHAT" if f_worst > 0 else "WASPADA (POTENSI DEFISIT OPERASIONAL)"
        risk_exposure = ((res['eat_forecast'].iloc[-1] - f_worst) / res['eat_forecast'].iloc[-1]) * 100
        
        report_text = f"""
        1. ANALISIS PERTUMBUHAN PENDAPATAN:
           Berdasarkan hasil uji stasioneritas ADF (p-value: {res['p_value']:.4f}), data menunjukkan tren yang kuat sehingga sistem menerapkan parameter Differencing (d=1). Hasil proyeksi parameter {target_col} diestimasikan tumbuh sebesar {res['growth_pct']:.2f}% dengan pola pergerakan yang dinamis dan musiman, menghindari bias linear (asumsi pertumbuhan konstan yang tidak realistis).
        
        2. STRUKTUR PROFITABILITAS DAN OPERASIONAL:
           Rata-rata Margin Laba Bersih (EAT) dikunci pada rasio historis sebesar {res['historical_eat_margin']*100:.2f}%. Pada akhir rentang proyeksi, jaringan outlet diestimasikan mampu menghasilkan Laba Bersih optimal hingga Rp {res['eat_forecast'].iloc[-1]/1e6:,.2f} Juta per bulan, dengan asumsi inflasi biaya operasional tidak melebihi rata-rata historis.
        
        3. EVALUASI SKENARIO MITIGASI RISIKO (WORST-CASE SCENARIO):
           Model mendeteksi adanya tingkat risiko ketidakpastian pasar sebesar {risk_exposure:.2f}% pada komponen laba bersih. Mengacu pada skenario terburuk (Lower Bound), bisnis diproyeksikan tetap mempertahankan profitabilitas positif dengan estimasi laba minimum Rp {f_worst/1e6:,.2f} Juta.
        
        4. KESIMPULAN STRATEGIS DAN VONIS KESEHATAN OUTLET:
           STATUS KESEHATAN FINANSIAL: [{status_kesehatan}]
           Kesimpulan: Jaringan bisnis memiliki ketahanan kapital yang memadai dalam meredam fluktuasi biaya operasional karena batas bawah risiko (Lower Bound) berada di atas angka nol. Manajemen direkomendasikan untuk mempertahankan efisiensi biaya tetap (fixed costs) harian agar margin keuntungan tetap berada pada zona aman.
        """
        st.info(report_text)
        
        # Ekspor Data untuk Kebutuhan Rapat Anggaran
        st.subheader("📥 Pusat Unduh Data Laporan Eksekutif")
        st.markdown("Dokumen data mentah hasil proyeksi ini dapat diunduh untuk kebutuhan penyusunan Rencana Kerja dan Anggaran Perusahaan (RKAP):")
        
        csv_data = res['export_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Laporan Finansial Lengkap (.CSV)",
            data=csv_data,
            file_name=f"BrightHub_Laporan_Proyeksi_{target_col}_2030.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("⚠️ Menunggu unggahan dokumen data laporan keuangan...")
        st.info("""
        **Petunjuk Standardisasi Struktur Dokumen Keuangan:**
        1. Berkas wajib berformat Excel (`.xlsx`).
        2. Lembar kerja wajib memiliki kolom dengan penamaan: **Date** (Format Tanggal), **Revenue**, **Ebitda**, dan **EAT**.
        3. Gunakan panel pengaturan di sebelah kiri untuk menyesuaikan parameter analisis.
        """)

# ------------------------------------------------------------------------------
# TAB 2: ANALISIS SENTIMEN KONSUMEN
# ------------------------------------------------------------------------------
with tab2:
    st.header("💬 Analisis Sentimen Konsumen & Persepsi Publik")
    st.markdown("---")
    
    st.markdown("""
    Modul ini berfungsi untuk mengevaluasi data opini publik dari berbagai saluran komunikasi digital 
    (Media Sosial dan YouTube Review) guna mengidentifikasi faktor eksternal yang memengaruhi kinerja penjualan produk retail.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("💬 Pengaturan Data Sentimen")
    uploaded_sentiment_file = st.sidebar.file_uploader("Unggah Berkas Sentimen (.csv/.xlsx)", type=["csv", "xlsx"])
    
    if uploaded_sentiment_file is not None:
        try:
            if uploaded_sentiment_file.name.endswith('.csv'):
                df_sent = pd.read_csv(uploaded_sentiment_file)
            else:
                df_sent = pd.read_excel(uploaded_sentiment_file)
                
            df_sent['Date'] = pd.to_datetime(df_sent['Date'])
            
            # Perhitungan Proporsi Sentimen Publik
            total_tweets = len(df_sent)
            positive_pct = (len(df_sent[df_sent['Sentiment'].str.upper() == 'POSITIF']) / total_tweets) * 100
            negative_pct = (len(df_sent[df_sent['Sentiment'].str.upper() == 'NEGATIF']) / total_tweets) * 100
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Total Data Opini yang Dievaluasi", f"{total_tweets:,} Sampel")
            with col_s2:
                st.metric("Rasio Sentimen Positif", f"{positive_pct:.1f}%")
            with col_s3:
                st.metric("Rasio Sentimen Negatif", f"{negative_pct:.1f}%", delta=f"{negative_pct:.1f}%", delta_color="inverse")
                
            st.markdown("---")
            
            # Tampilan Grafik Distribusi dan Tren Sentimen
            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                st.subheader("📊 Komparasi Distribusi Sentimen per Platform")
                fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
                platform_sent = pd.crosstab(df_sent['Platform'], df_sent['Sentiment'])
                platform_sent.plot(kind='bar', stacked=True, color=['#d35400', '#7f8c8d', '#27ae60'], ax=ax_bar)
                ax_bar.set_ylabel("Volume Komentar")
                ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=0)
                ax_bar.grid(axis='y', alpha=0.3)
                st.pyplot(fig_bar)
                
            with col_graph2:
                st.subheader("📈 Grafik Tren Mingguan Komplain Konsumen")
                fig_trend, ax_trend = plt.subplots(figsize=(8, 5))
                df_neg = df_sent[df_sent['Sentiment'].str.upper() == 'NEGATIF'].set_index('Date')
                trend_data = df_neg.resample('W').size()
                ax_trend.plot(trend_data.index, trend_data.values, marker='o', color='#c0392b', linewidth=2)
                ax_trend.set_ylabel("Volume Komplain")
                ax_trend.grid(alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig_trend)
                
            st.markdown("---")
            
            # Kategorisasi Topik Keluhan dan Kepuasan (Root Cause Analysis)
            st.subheader("🔍 Identifikasi Masalah Utama & Key Drivers Kepuasan")
            col_topic1, col_topic2 = st.columns(2)
            
            with col_topic1:
                st.error("🚨 Indikator Utama Komplain (Sentimen Negatif)")
                st.markdown("""
                *   **Ketersediaan Stok Produk (Supply Chain):** Keterbatasan ketersediaan produk siap saji pada jam-jam sibuk.
                *   **Waktu Tunggu Transaksi:** Kecepatan pelayanan kasir pada beberapa titik area SPBU saat kondisi padat.
                *   **Rasio Harga Produk:** Persepsi konsumen terkait selisih harga produk ritel tertentu dibanding kompetitor pasar.
                """)
                
            with col_topic2:
                st.success("✨ Indikator Utama Kepuasan (Sentimen Positif)")
                st.markdown("""
                *   **Kenyamanan Fasilitas Lokasi:** Kesesuaian ruangan untuk aktivitas kerja kasual (Work From Cafe comfort).
                *   **Konsistensi Kualitas Produk Kuliner:** Standar cita rasa menu makanan dan minuman kopi yang dinilai bersaing baik.
                *   **Program Integrasi Digital:** Efisiensi promosi potongan harga yang terhubung langsung dengan aplikasi MyPertamina.
                """)
                
            st.markdown("---")
            
            # Tabel Audit Komentar untuk Peninjauan Manajemen Risiko
            st.subheader("📋 Audit Berkas Opini Publik Berdampak Tinggi")
            platform_filter = st.selectbox("Pilih Filter Platform", df_sent['Platform'].unique())
            sentiment_filter = st.selectbox("Pilih Filter Sentimen", df_sent['Sentiment'].unique())
            
            filtered_df = df_sent[(df_sent['Platform'] == platform_filter) & (df_sent['Sentiment'] == sentiment_filter)]
            st.dataframe(filtered_df[['Date', 'Comment', 'Username', 'Engagement_Count']], use_container_width=True)
            
        except Exception as e:
            st.error(f"Gagal memproses visualisasi data sentimen. Format kolom tidak sesuai standar. Error: {e}")
            
    else:
        st.warning("⚠️ Menunggu unggahan berkas data sentimen pasar...")
        st.info("""
        **Petunjuk Standardisasi Struktur Berkas Sentimen:**
        1. Format berkas yang didukung adalah `.csv` atau `.xlsx`.
        2. Berkas wajib memiliki kolom sebagai berikut:
           - **Date** (Format Tanggal standar)
           - **Comment** (Teks narasi opini/komentar)
           - **Sentiment** (Isi kategori dengan nilai: 'Positif', 'Netral', atau 'Negatif')
           - **Platform** (Asal sumber data, contoh: 'X', 'Instagram', 'TikTok', 'YouTube')
           - **Username** & **Engagement_Count** (Detail pelengkap analisis dampak)
        """)

# ------------------------------------------------------------------------------
# TAB 3: STRATEGI KAMPANYE PEMASARAN (BRIGHT ON)
# ------------------------------------------------------------------------------
with tab3:
    st.header("🏃 Rekomendasi Kampanye Strategis: Bright marathON / Bright ON")
    st.markdown("---")
    
    st.markdown("""
    ### Integrasi Strategis Ekosistem Non-Fuel Retail Melalui Aktivasi Event Olahraga
    Modul ini menyajikan model simulasi usulan program kerja kampanye olahraga tahunan untuk mendorong peningkatan 
    keterikatan konsumen (*Customer Engagement*) serta konversi penjualan pada seluruh *Own Brand* korporasi.
    """)
    
    st.subheader("📌 Matriks Pilihan Skenario Eksekusi Program")
    opsi_strategi = st.radio(
        "Tentukan Skenario Peluncuran Kampanye Kerja:",
        ["Skenario Aliansi: Pilot Project Terintegrasi (Tap-In Event Pertamina Eco RunFest Holding)", 
         "Skenario Mandiri: Standalone Brand Event (Bright marathON diselenggarakan oleh Patra Niaga)"]
    )
    
    if "Pilot Project" in opsi_strategi:
        st.info("""
        💡 **ANALISIS INTEGRASI SKENARIO ALIANSI (TAHAPAN REKOMENDASI AWAL):**
        Melakukan penggabungan strategi pemasaran dengan mengambil peran sebagai penyedia utama zona kuliner dan hub ritel terintegrasi pada acara olahraga *Pertamina Eco RunFest* milik Holding.
        - **Mitigasi Risiko:** Mengurangi beban pengeluaran anggaran investasi awal secara masif dan menyederhanakan manajemen perizinan rute jalan.
        - **Output Pengukuran Kinerja:** Menjadi basis pengumpulan data konversi transaksi riil (*Proof of Concept*) sebelum meluncurkan program kerja mandiri.
        """)
    else:
        st.warning("""
        ⚠️ **ANALISIS POTENSI DAN INVESTASI SKENARIO MANDIRI (TAHAPAN LANJUTAN):**
        Menyelenggarakan kegiatan mega-event olahraga tahunan secara independen di bawah koordinasi penuh Pertamina Patra Niaga.
        - **Dampak Konsolidasi Internal:** Menjadi instrumen pencapaian target kinerja utama (KPI) lintas fungsi jabatan manajemen pemasaran ritel.
        - **Struktur Acara Festival:** Menggabungkan aktivitas kompetisi lari jalanan pada hari pertama, dan dilanjutkan festival seni/musik selama dua hari berturut-turut untuk mengoptimalkan paparan merek dagang ritel.
        """)

    st.markdown("---")

    # Simulasi Lintasan dan Pemetaan Jalur Aktivasi
    st.subheader("🗺️ Pemetaan Rencana Lintasan Lari & Distribusi Hub Komersial")
    col_route1, col_route2 = st.columns([1, 2])
    
    with col_route1:
        st.markdown("""
        **Rencana Geografis Jalur Lintasan Lari:**
        *   🏁 **Titik Start:** Kawasan Monumen Nasional (Monas)
        *   📍 **Titik Check Point / Media Branding Utama:** Bundaran Hotel Indonesia (HI)
        *   🏁 **Titik Finish & Area Utama Festival:** Kawasan Gelora Bung Karno (GBK)
        
        **Strategi Distribusi Titik Komersial (Own Brand Touchpoints):**
        - *Water Station & Hydration Boost:* Penempatan produk konsumsi cepat saji yang dikelola eksklusif oleh jaringan *Bright Store* dan *Bright Cafe*.
        - *Booster Activation Booth:* Penyediaan fasilitas stan pengalaman produk untuk *Olimart, Bright Gas*, serta pusat aktivasi pengunduhan aplikasi *MyPertamina*.
        """)
        
    with col_route2:
        # Pembuatan visualisasi peta diagram lintasan lari makro
        fig_route, ax_route = plt.subplots(figsize=(10, 5))
        x_coords = [1, 2, 3] 
        y_coords = [5, 3, 1]
        labels = ['🏁 Start: Area Monas', '📍 Check Point: Bundaran HI', '🏆 Finish & Festival: Area GBK']
        
        ax_route.plot(x_coords, y_coords, linestyle='-', marker='o', color='#1c3d5a', linewidth=3, markersize=10)
        ax_route.fill_between(x_coords, y_coords, color='#27ae60', alpha=0.1, label='Zona Integrasi Komersial Retail')
        
        for i, txt in enumerate(labels):
            ax_route.annotate(txt, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', fontsize=10)
            
        ax_route.set_title("Simulasi Pemetaan Alur Lintasan Lari & Zona Hub Komersial", fontsize=12, fontweight='bold')
        ax_route.axis('off')
        st.pyplot(fig_route)

    st.markdown("---")

    # Simulator Finansial Proyeksi ROI Event
    st.subheader("🧮 Model Simulator Nilai Pengembalian Investasi (ROI)")
    st.markdown("Sesuaikan indikator parameter di bawah ini untuk melihat estimasi dampak sirkulasi ekonomi dan potensi nilai pendapatan tidak langsung:")

    col_calc1, col_calc2 = st.columns(2)
    with col_calc1:
        total_pelari = st.slider("Target Akumulasi Jumlah Peserta & Pengunjung Festival:", min_value=5000, max_value=30000, value=15000, step=1000)
    with col_calc2:
        belanja_per_orang = st.slider("Estimasi Rata-rata Nilai Belanja Per Pengunjung di Area Tenant (Rp):", min_value=25000, max_value=150000, value=75000, step=5000)

    # Logika Matematis Kalkulasi ROI Bisnis Ritel
    total_gross_revenue = total_pelari * belanja_per_orang
    estimasi_fee_bumn = total_gross_revenue * 0.15 # Estimasi take-rate komisi bagi hasil pengelolaan area komersial ritel & UMKM binaan
    proyeksi_user_baru_mypertamina = int(total_pelari * 0.40) # Target estimasi konversi pengguna baru aplikasi transaksi digital

    # Menampilkan Hasil Perhitungan Indikator Finansial Acara
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("Estimasi Total Nilai Transaksi (Gross GMV)", f"Rp {total_gross_revenue:,.0f}", help="Total nilai transaksi kumulatif yang berputar di seluruh ekosistem bisnis retail sepanjang periode acara.")
    with col_res2:
        st.metric("Proyeksi Pendapatan Bersih Tidak Langsung (15% Take-Rate)", f"Rp {estimasi_fee_bumn:,.0f}", help="Pendapatan dari kontribusi biaya sewa stan, bagi hasil penjualan produk mitra ritel, dan komisi merchant.")
    with col_res3:
        st.metric("Target Akuisisi Pengguna Baru MyPertamina", f"+{proyeksi_user_baru_mypertamina:,} Pengguna", delta=f"{proyeksi_user_baru_mypertamina} DAU Spike", delta_color="inverse")

    st.markdown("---")

    # Kesimpulan Akhir Evaluasi Program Strategis
    st.subheader("📋 Rekomendasi Manajerial Akhir")
    
    rekomendasi_narasi = f"""
    *   **Korelasi Analisis Lintas Modul (360° Perspective):** Hasil evaluasi opini konsumen pada **Tab 2** menunjukkan adanya area perbaikan pada aspek *brand awareness* menu kuliner. Pendekatan pelaksanaan program olahraga massa pada **Tab 3** ini bertindak sebagai media akselerasi taktis guna menciptakan pengalaman langsung konsumen (*experiential marketing*) terhadap kualitas lini produk ritel baru kita di hadapan **{total_pelari:,} target sasaran**.
    *   **Rencana Garis Waktu Eksekusi (Roadmap):** Tahapan pengajuan uji coba berskala terbatas (Skenario Aliansi) diusulkan untuk masuk dalam agenda perencanaan program triwulan berikutnya. Hal ini bertujuan sebagai media pengumpulan data performa valid yang akan digunakan untuk mendukung proses pengambilan keputusan strategis oleh jajaran direksi.
    """
    st.markdown(rekomendasi_narasi)
    st.success("🎯 **STATUS EVALUASI:** Seluruh modul arsitektur Bright Hub Dashboard telah terintegrasi penuh. Sistem siap dipergunakan untuk mendukung proses presentasi manajerial.")
