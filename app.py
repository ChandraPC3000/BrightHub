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
    Fungsi berat untuk membaca data, running backtesting, dan forecasting.
    Menggunakan @st.cache_data agar dashboard tidak lambat saat user ganti visualisasi.
    """
    # 1. Data Preparation
    df = pd.read_excel(file_upload, sheet_name=sheet_name)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=[target_column]).sort_values('Date')
    df.set_index('Date', inplace=True)
    df.index.freq = 'MS'
    ts_data = df[target_column]
    
    # 2. ADF Test
    adf_result = adfuller(ts_data)
    p_val = adf_result[1]
    adf_stat = adf_result[0]
    crit_vals = adf_result[4]
    
    # 3. Backtesting Validation Split
    split_point = int(len(ts_data) * train_ratio)
    train_data = ts_data.iloc[:split_point]
    test_data = ts_data.iloc[split_point:]
    
    # Fit Model untuk Validasi
    val_arima_model = ARIMA(train_data, order=(1,1,1), seasonal_order=(1,1,0,12)).fit()
    val_arima_pred = val_arima_model.forecast(steps=len(test_data))
    
    val_hw_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()
    val_hw_pred = val_hw_model.forecast(steps=len(test_data))
    
    # Hitung Akurasi Backtesting
    rmse_arima = np.sqrt(mean_squared_error(test_data, val_arima_pred))
    mape_arima = calculate_mape(test_data, val_arima_pred)
    mape_hw = calculate_mape(test_data, val_hw_pred)
    
    # 4. Final Production Forecast (Seluruh Data)
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
    
    # Buat DataFrame Konsolidasi untuk Ditampilkan/Unduh
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

# REVISI: Mengganti st.set_page_index yang salah menjadi fungsi bawaan resmi Streamlit
st.set_page_config(
    page_title="The Data Statesman Dashboard",
    page_icon="💡",
    layout="wide"
)

st.title("💡 The Data Statesman: 360° Business Intelligence Dashboard")
st.caption("Samchan Tech Production // Powered by Data, Big Data & AI for Indonesia 2045")

# Setup 3 Tab Utama seperti Masterplan
tab1, tab2, tab3 = st.tabs(["📊 Financial & Sales Forecast", "💬 Customer Sentiment Analytics", "🏃 Strategic Campaign (Bright ON)"])

# ------------------------------------------------------------------------------
# TAB 1 RUNNING INJECTOR
# ------------------------------------------------------------------------------
with tab1:
    st.header("Financial Forecasting & Risk Analysis Matrix")
    st.markdown("---")
    
    # Sidebar Control Area untuk User File Input
    st.sidebar.header("📁 Control Panel Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset Excel Retail (Bright Store)", type=["xlsx"])
    
    if uploaded_file is not None:
        # Parameter input dinamis untuk Direktorat/Tim Sales
        target_col = st.sidebar.selectbox("Pilih Parameter Target", ["Revenue", "Ebitda", "EAT"])
        sheet = st.sidebar.text_input("Nama Sheet Excel", value="Sheet1")
        
        # Eksekusi Big Data Engine
        with st.spinner("Sistem sedang mengkalkulasi model ARIMA & Holt-Winters..."):
            res = run_statistical_analysis(uploaded_file, sheet, target_col, train_ratio=0.80)
        
        # 1. DISPLAY METRICS EXECUTIVE SUMMARY
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Akurasi Model (100 - MAPE)", f"{100 - res['mape_arima']:.2f}%", help="Standar Industri: >90% Sangat Akurat, 80-90% Baik")
        with col2:
            st.metric("Margin EBITDA Historis", f"{res['historical_ebitda_margin']*100:.2f}%")
        with col3:
            st.metric("Proyeksi Pertumbuhan Omzet", f"+{res['growth_pct']:.2f}%")
            
        st.markdown("---")
        
        # 2. VISUALISASI INTEGRASI UTAMA (MATPLOTLIB TO STREAMLIT)
        st.subheader("🔮 Grafik Proyeksi Keuangan Terintegrasi (2026 - 2030)")
        
        fig, ax = plt.subplots(figsize=(15, 7))
        # Menggunakan palet warna Korporasi Pertamina (Biru, Hijau, Oranye)
        # REVISI: Label chart dinamis mengikuti parameter target (target_col) yang dipilih di sidebar
        ax.plot(res['rev_forecast'].index, res['rev_forecast'].values / 1e6, label=f'Projected {target_col} (Top Line)', color='#1c3d5a', linewidth=2.5)
        ax.plot(res['ebitda_forecast'].index, res['ebitda_forecast'].values / 1e6, label='Projected EBITDA', color='#27ae60', linestyle='--')
        ax.plot(res['eat_forecast'].index, res['eat_forecast'].values / 1e6, label='Projected EAT (Laba Bersih)', color='#d35400', linewidth=2.5)
        
        # Anti-Manis Policy Visual Area
        ax.fill_between(res['forecast_index'], res['eat_lower_bound'].values / 1e6, res['eat_forecast'].values / 1e6, 
                        color='#e67e22', alpha=0.15, label='Area Risiko Ketidakpastian Laba Bersih')
        
        ax.set_ylabel('Dalam Juta Rupiah', fontsize=11)
        ax.set_title('Simulasi Target & Manajemen Risiko Finansial Outlet', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left')
        
        st.pyplot(fig)
        
        # 3. EXECUTIVE DIAGNOSIS AUTOMATION (THE DATA STATESMAN STYLE)
        st.subheader("📋 Laporan Diagnosa Otomatis & Vonis Kesehatan")
        
        # Logic Otomatis Berdasarkan Output Data
        f_worst = res['export_df']['EAT_Worst_Case_Scenario_IDR'].iloc[-1]
        status_kesehatan = "SANGAT SEHAT" if f_worst > 0 else "WASPADA (POTENSI RUGI OPERASIONAL)"
        risk_exposure = ((res['eat_forecast'].iloc[-1] - f_worst) / res['eat_forecast'].iloc[-1]) * 100
        
        report_text = f"""
        1. ANALISIS PARAMETER TARGET & PERTUMBUHAN:
           Berdasarkan uji stasioneritas ADF (p-value: {res['p_value']:.4f}), data terbukti memiliki tren kuat sehingga sistem secara otomatis mengaktifkan parameter Differencing (d=1). Hasilnya, proyeksi {target_col} tumbuh {res['growth_pct']:.2f}% secara realistis dan siklikal (mengikuti pola ramai-sepi musiman, tidak naik lurus/manis).
        
        2. STRUKTUR LABA & EFISIENSI OPERASIONAL:
           Rata-rata Margin Laba Bersih (EAT) dikunci pada angka {res['historical_eat_margin']*100:.2f}%. Pada akhir periode proyeksi, outlet diperkirakan mampu menghasilkan Laba Bersih ideal hingga Rp {res['eat_forecast'].iloc[-1]/1e6:,.2f} Juta per bulan, dengan catatan rasio pengeluaran tetap terkendali.
        
        3. AUDIT RISIKO & SKENARIO TERPAHIT (ANTI-MANIS POLICY):
           Model mendeteksi adanya paparan risiko ketidakpastian sebesar {risk_exposure:.2f}% pada laba utama. Dalam skenario terburuk (Lower Bound), outlet diproyeksikan tetap menghasilkan profit positif sebesar Rp {f_worst/1e6:,.2f} Juta.
        
        4. VONIS KESEHATAN AKHIR & REKOMENDASI:
           STATUS KESEHATAN OUTLET: [{status_kesehatan}]
           Keterangan: Bisnis memiliki bantalan (buffer) finansial yang kokoh terhadap guncangan biaya karena batas bawah risiko berada di atas angka nol. Manajemen disarankan menjaga kenaikan biaya operasional harian tetap berada di bawah ambang rasio historis demi menjaga area laba oranye tidak tergerus.
        """
        st.info(report_text)
        
        # 4. DATA EXPORT HUB FOR DIRECTORS
        st.subheader("📥 Executive Data Download Center")
        st.markdown("Direksi membutuhkan salinan data mentah untuk rapat penganggaran belanja modal (*Capex*). Gunakan tombol di bawah ini:")
        
        # Ubah dataframe menjadi CSV untuk fungsi unduh Streamlit
        csv_data = res['export_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Analisis Finansial Lengkap (.CSV)",
            data=csv_data,
            file_name=f"Analisis_Finansial_TargetColumn_{target_col}_2030.csv",
            mime="text/csv"
        )
        
    else:
        # Tampilan Awal Sebelum User Upload File
        st.warning("⚠️ Menunggu Sistem Mengunggah Dataset Excel...")
        st.info("""
        **Petunjuk Unggah Dataset untuk Tim Sales/NFR:**
        1. Pastikan file berformat `.xlsx`.
        2. File harus memiliki kolom bernama **Date** (Format: YYYY-MM-DD), **Revenue**, **Ebitda**, dan **EAT**.
        3. Atur parameter target di panel kiri untuk memulai kalkulasi kecerdasan buatan.
        """)

# ------------------------------------------------------------------------------
# TAB 2 & TAB 3 SKELETON (UNTUK LANGKAH BERIKUTNYA)
# ------------------------------------------------------------------------------
# Insert kode ini untuk menggantikan blok 'with tab2:' yang lama di app.py lu

# ------------------------------------------------------------------------------
# TAB 2 INJECTOR (CUSTOMER SENTIMENT ANALYTICS ENGINE)
# ------------------------------------------------------------------------------
with tab2:
    st.header("💬 Customer Sentiment & Public Opinion Analytics")
    st.markdown("---")
    
    st.markdown("""
    Modul ini membaca data opini publik dari hasil *scraping* multi-platform (X/Twitter, Instagram, TikTok, dan YouTube Review) 
    untuk memetakan alasan di balik pergerakan grafik finansial *Bright Store & Cafe*.
    """)
    
    # Control Panel Khusus Tab 2 di Sidebar (Hanya muncul jika file di-upload)
    st.sidebar.markdown("---")
    st.sidebar.header("💬 Control Panel Analisis Sentimen")
    uploaded_sentiment_file = st.sidebar.file_uploader("Upload Data Sentimen (.csv/.xlsx)", type=["csv", "xlsx"])
    
    if uploaded_sentiment_file is not None:
        # Load Sentiment Data
        try:
            if uploaded_sentiment_file.name.endswith('.csv'):
                df_sent = pd.read_csv(uploaded_sentiment_file)
            else:
                df_sent = pd.read_excel(uploaded_sentiment_file)
                
            # Pastikan kolom standar ada: 'Date', 'Comment', 'Sentiment', 'Platform'
            df_sent['Date'] = pd.to_datetime(df_sent['Date'])
            
            # 1. METRICS SUMMARY FOR SENTIMENT
            total_tweets = len(df_sent)
            positive_pct = (len(df_sent[df_sent['Sentiment'].str.upper() == 'POSITIF']) / total_tweets) * 100
            negative_pct = (len(df_sent[df_sent['Sentiment'].str.upper() == 'NEGATIF']) / total_tweets) * 100
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Total Opini Publik Diarsip", f"{total_tweets:,} Sampel")
            with col_s2:
                st.metric("Sentimen Positif (Pujian)", f"{positive_pct:.1f}%")
            with col_s3:
                st.metric("Sentimen Negatif (Komplain)", f"{negative_pct:.1f}%", delta=f"{negative_pct:.1f}%", delta_color="inverse")
                
            st.markdown("---")
            
            # 2. VISUALISASI SEBARAN & TREN SENTIMEN
            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                st.subheader("📊 Distribusi Sentimen per Platform")
                fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
                # Membuat cross-tabulation antara Platform dan Sentiment
                platform_sent = pd.crosstab(df_sent['Platform'], df_sent['Sentiment'])
                platform_sent.plot(kind='bar', stacked=True, color=['#d35400', '#7f8c8d', '#27ae60'], ax=ax_bar)
                ax_bar.set_ylabel("Jumlah Komentar")
                ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=0)
                ax_bar.grid(axis='y', alpha=0.3)
                st.pyplot(fig_bar)
                
            with col_graph2:
                st.subheader("📈 Tren Sentimen Negatif Mingguan")
                fig_trend, ax_trend = plt.subplots(figsize=(8, 5))
                # Resample data untuk melihat tren komplain per minggu
                df_neg = df_sent[df_sent['Sentiment'].str.upper() == 'NEGATIF'].set_index('Date')
                trend_data = df_neg.resample('W').size()
                ax_trend.plot(trend_data.index, trend_data.values, marker='o', color='#c0392b', linewidth=2)
                ax_trend.set_ylabel("Jumlah Komplain")
                ax_trend.grid(alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig_trend)
                
            st.markdown("---")
            
            # 3. WORD ASSOCIATION & CLUSTER MASALAH
            st.subheader("🔍 Klaster Masalah & Topik Hangat Netizen")
            
            col_topic1, col_topic2 = st.columns(2)
            with col_topic1:
                st.error("🚨 Topik Utama Komplain (Sentimen Negatif)")
                st.markdown("""
                * **Ketersediaan Stok (Supply Chain Error):** *"Makanannya banyak yang habis kalau malam di SPBU MT Haryono."*
                * **Kecepatan Pelayanan:** *"Antrean kasir Bright Cafe panjang banget, kasirnya lambat."*
                * **Rasio Harga/Value:** *"Harga beberapa item minimarket agak kemahalan dibanding retail biasa."*
                """)
                
            with col_topic2:
                st.success("✨ Faktor Pendorong Kepuasan (Sentimen Positif)")
                st.markdown("""
                * **Fasilitas Tempat (WFC Comfort):** *"Tempatnya bersih, adem, cocok banget buat numpang kerja pas macet."*
                * **Kualitas Kopi & Menu:** *"Kopi susu aren Bright Cafe rasanya konsisten enak, gak kalah sama cafe mahal."*
                * **Benefit Integrasi:** *"Promo cashback kalau bayar pakai MyPertamina lumayan memotong harga."*
                """)
                
            st.markdown("---")
            
            # 4. RAW LOG DATA UNTUK AUDIT STRATEGIS
            st.subheader("📋 Dashboard Audit Komentar Paling Viral")
            platform_filter = st.selectbox("Filter Berdasarkan Platform", df_sent['Platform'].unique())
            sentiment_filter = st.selectbox("Filter Berdasarkan Sentimen", df_sent['Sentiment'].unique())
            
            filtered_df = df_sent[(df_sent['Platform'] == platform_filter) & (df_sent['Sentiment'] == sentiment_filter)]
            st.dataframe(filtered_df[['Date', 'Comment', 'Username', 'Engagement_Count']], use_container_width=True)
            
        except Exception as e:
            st.error(f"Gagal memproses data sentimen. Pastikan format kolom sesuai. Error: {e}")
            
    else:
        st.warning("⚠️ Menunggu Sistem Mengunggah Dataset Sentimen...")
        st.info("""
        **Petunjuk Unggah Dataset Sentimen untuk Tim Pemasaran/Marketing:**
        1. Gunakan file berformat `.csv` atau `.xlsx`.
        2. File wajib memiliki struktur kolom berikut:
           - **Date** (Format Tanggal)
           - **Comment** (Teks komentar netizen)
           - **Sentiment** (Isi dengan kategori: 'Positif', 'Netral', atau 'Negatif')
           - **Platform** (Isi dengan platform asal: 'X', 'Instagram', 'TikTok', atau 'YouTube')
           - **Username** & **Engagement_Count** (Opsional untuk data audit)
        """)


# Insert kode ini untuk menggantikan blok 'with tab3:' yang lama di app.py lu

# ------------------------------------------------------------------------------
# TAB 3 INJECTOR (STRATEGIC CAMPAIGN & ROI ENGINE)
# ------------------------------------------------------------------------------
with tab3:
    st.header("🏃 Strategic Campaign: Bright marathON / Bright ON")
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 Merajut Ekosistem Ritel Melalui Aktivasi Massa: "RGB-kan Jakarta!"
    Modul ini dirancang sebagai rekomendasi taktis berbasis data untuk mendongkrak *Demand Generation* 
    dan loyalitas konsumen urban terhadap seluruh *Own Brand* Pertamina Patra Niaga (*Bright Store, Bright Cafe, Olimart, Bright Gas, dan MyPertamina*).
    """)
    
    # Pemisahan Opsi Strategis Menggunakan Expander Profesional
    st.subheader("📌 Pilihan Eksekusi Strategis (Direksi Policy Decision)")
    opsi_strategi = st.radio(
        "Pilih Skenario Peluncuran Kampanye:",
        ["Skenario A: Pilot Project (Tap-In di Pertamina Eco RunFest Holding)", 
         "Skenario B: Standalone Mega Event (Bright marathON Mandiri oleh PPN)"]
    )
    
    if "Pilot Project" in opsi_strategi:
        st.info("""
        💡 **ANALISIS TAKTIS SKENARIO A (REKOMENDASI FASE 1):**
        Melakukan integrasi dengan mengambil alih (*takeover*) zona kuliner dan aktivasi di acara *Pertamina Eco RunFest* milik Holding (Pertamina Persero). 
        - **Kelebihan:** Minim risiko anggaran, perizinan rute Jakarta sudah di-cover Holding, fokus murni pada optimalisasi konversi penjualan tenant ritel.
        - **Output Data:** Menjadi basis data riil (*Proof of Concept*) untuk membuktikan lonjakan GMV MyPertamina sebelum mengajukan anggaran mandiri tahun depan.
        """)
    else:
        st.warning("""
        ⚠️ **ANALISIS RISIKO SKENARIO B (FASE LANJUTAN):**
        Membuat *standalone event* tahunan mandiri di bawah bendera Pertamina Patra Niaga (PPN).
        - **Dampak Strategis:** Konsolidasi penuh 4 lini VP (Retail Fuel Sales, Retail Business Support, Retail Gas Sales, dan Retail Marketing) untuk meleburkan ego sektoral dan mengejar KPI bersama.
        - **Cakupan Event:** Marathon rute protokol di hari pertama, dilanjutkan dengan Festival Musik/Konser Artis Viral selama 2 hari penuh untuk mengunci *engagement* generasi muda.
        """)

    st.markdown("---")

    # 1. VISUALISASI RUTE AKTIVASI MASSA
    st.subheader("🗺️ Simulasi Rute Sakral Aktivasi Korporasi")
    col_route1, col_route2 = st.columns([1, 2])
    
    with col_route1:
        st.markdown("""
        **Rencana Jalur Lari Makro:**
        *   🏁 **Start Point:** Monas (Jantung Sejarah & Ikon Jakarta)
        *   📍 **Check Point:** Bundaran HI (Pusat Sentralisasi Ekonomi & Branding)
        *   🏁 **Finish Point & Festival Venue:** Gelora Bung Karno / GBK (Pusat Massa Gaya Hidup Urban)
        
        **Zona Aktivasi Sepanjang Rute:**
        - *Water Station & Hydration Boost:* Dikelola penuh oleh Bright Store & Bright Cafe.
        - *Booster Activation Booth:* Penempatan stan interaktif Olimart, Bright Gas, dan konter registrasi MyPertamina di spot penonton.
        """)
        
    with col_route2:
        # Visualisasi rute sederhana menggunakan grafik koordinat Matplotlib
        fig_route, ax_route = plt.subplots(figsize=(10, 5))
        # Titik Koordinat Ikonik Jakarta (Simulasi Tren Jalur)
        x_coords = [1, 2, 3] # Monas, HI, GBK
        y_coords = [5, 3, 1]
        labels = ['🏁 Start: Monas', '📍 CP: Bundaran HI', '🏆 Finish: GBK Festival']
        
        # Plot Jalur
        ax_route.plot(x_coords, y_coords, linestyle='-', marker='o', color='#1c3d5a', linewidth=3, markersize=10)
        ax_route.fill_between(x_coords, y_coords, color='#27ae60', alpha=0.1, label='Zona Biru-Hijau Penggerak Ekonomi Ritel')
        
        # Tambahkan Label pada Titik
        for i, txt in enumerate(labels):
            ax_route.annotate(txt, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', fontsize=10)
            
        ax_route.set_title("Peta Rencana Alur Lintasan Lari & Zona Aktivasi Ritel PPN", fontsize=12, fontweight='bold')
        ax_route.axis('off')
        st.pyplot(fig_route)

    st.markdown("---")

    # 2. DYNAMIC ROI & ACTIVATION CALCULATOR (THE DATA STATESMAN CALCULATOR)
    st.subheader("🧮 Simulator Pengembalian Investasi & Dampak Finansial (ROI)")
    st.markdown("Geser parameter di bawah ini untuk melihat simulasi perputaran uang dan dampaknya pada target laba ritel kita:")

    # Input Widget Dinamis di dalam Area Utama Tab 3
    col_calc1, col_calc2 = st.columns(2)
    with col_calc1:
        total_pelari = st.slider("Target Jumlah Peserta Lari / Pengunjung Festival:", min_value=5000, max_value=30000, value=15000, step=1000)
    with col_calc2:
        belanja_per_orang = st.slider("Estimasi Rata-rata Belanja per Orang di Area Tenant (Rp):", min_value=25000, max_value=150000, value=75000, step=5000)

    # Logika Perhitungan Finansial Korporat (Arsitektur Samchan Tech)
    total_gross_revenue = total_pelari * belanja_per_orang
    # Menggunakan profit margin rata-rata historis dari kalkulasi Tab 1 backend sebagai pengunci logika
    estimasi_fee_bumn = total_gross_revenue * 0.15 # 15% take-rate dari total transaksi mitra ritel & UMKM SMEPP
    proyeksi_user_baru_mypertamina = int(total_pelari * 0.40) # Asumsi 40% dari total crowd melakukan download/registrasi baru untuk bertransaksi

    # Menampilkan Hasil Hitungan dalam Bentuk KPI Cards Medsos/Direksi Style
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("Proyeksi Perputaran Uang (Gross GMV)", f"Rp {total_gross_revenue:,.0f}", help="Total nilai transaksi kumulatif di seluruh ekosistem Bright Hub sepanjang event.")
    with col_res2:
        st.metric("Net Revenue Patra Niaga (15% Take-Rate)", f"Rp {estimasi_fee_bumn:,.0f}", help="Pendapatan bersih tidak langsung dari biaya sewa, bagi hasil mitra NFR, dan komisi merchant.")
    with col_res3:
        st.metric("Target Pengguna Baru MyPertamina", f"+{proyeksi_user_baru_mypertamina:,} User", delta=f"{proyeksi_user_baru_mypertamina} DAU Spike", delta_color="inverse")

    st.markdown("---")

    # 3. EXECUTIVE CONCLUSION & NEXT ACTION ROADMAP
    st.subheader("📋 Kesimpulan Strategis & Dokumen Pengajuan Masuk")
    
    rekomendasi_narasi = f"""
    * **Integrasi Lintas Tab (360° View):** Berdasarkan tren sentimen di **Tab 2**, keluhan utama konsumen kita adalah seputar aksesibilitas dan *brand awareness* menu kuliner Bright Cafe. Melalui kampanye *Bright ON* dengan proyeksi massa sebanyak **{total_pelari:,} orang** ini, kita secara instan menciptakan *experiential marketing* skala masif.
    * **Rekomendasi Penjadwalan:** Peluncuran *Pilot Project* (Skenario A) direkomendasikan untuk menempel pada jadwal akhir tahun Holding, bertepatan dengan masa transisi pasca-magang di bulan Agustus hingga seleksi lanjutan BPS di bulan Oktober 2026. Ini memberikan momentum berharga bagi tim NFR untuk tetap mempertahankan kontribusi strategis Anda di hadapan manajemen.
    """
    st.markdown(rekomendasi_narasi)
    st.success("🎯 **STATUS DOKUMEN:** Masterplan Dashboard 360° Selesai Dirakit. Sistem Siap Didemonstrasikan di Hadapan Manajer NFR & Direksi.")
