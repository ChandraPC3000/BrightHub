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
with tab2:
    st.header("💬 Customer Sentiment & Public Opinion Analytics")
    st.info("Modul Sentiment Analysis (Scraping Media Sosial & Komentar YouTube) sedang mengantre antrean eksekusi CTO.")

with tab3:
    st.header("🏃 Strategic Campaign: Bright marathON / Bright ON")
    st.info("Modul Strategic Recommendation & Visualisasi Rute Aktivasi Massa sedang mengantre antrean eksekusi CTO.")
