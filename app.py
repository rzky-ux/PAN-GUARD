import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="PAN Guard Dashboard", page_icon="🌾", layout="wide")
st.title("🌾 PAN Guard: Dashboard Pemantauan & Peringatan Dini")
st.markdown("#### *Cakupan Wilayah: Provinsi Lampung (2021 - 2025)*")
st.markdown("Sistem cerdas untuk monitoring risiko, proyeksi harga gabah, dan mitigasi gagal panen.")

# ==========================================
# 2. FUNGSI UNTUK MEMBACA DATA ASLI
# ==========================================
@st.cache_data
def load_data():
    # Membaca CSV dengan melewati 3 baris pertama (metadata template)
    df = pd.read_csv("data_panen.csv", header=3)
    
    # Mengisi baris tahun yang kosong secara otomatis (forward fill)
    df['Tahun'] = df['Tahun'].ffill()
    
    # Memetakan nama bulan menjadi angka untuk konversi tanggal
    bulan_map = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
        'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }
    df['Bulan_Num'] = df['Bulan'].map(bulan_map)
    
    # Membuat kolom Tanggal standar berbasis datetime
    df['Tanggal'] = pd.to_datetime(dict(year=df['Tahun'].astype(int), month=df['Bulan_Num'], day=1))
    
    # Menyesuaikan nama kolom asli agar lebih bersih di dashboard
    df = df.rename(columns={
        "Curah Hujan\n(mm/bulan)": "Curah Hujan (mm)",
        "Suhu Rata-rata\n(°C)": "Suhu Rata-rata (°C)",
        "NTP\n(Indeks)": "NTP",
        "Produksi Padi\n(Ton)": "Produksi Padi (Ton)",
        "Harga GKP\n(Rp/Kg)": "Harga Gabah (Rp/Kg)"
    })
    
    # Membersihkan format angka (menghapus simbol 'Rp' dan tanda koma pemisah ribuan)
    cols_to_clean = ["Harga Gabah (Rp/Kg)", "Produksi Padi (Ton)"]
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('Rp', '', regex=False).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Mengurutkan data berdasarkan waktu secara kronologis
    df = df.sort_values("Tanggal").reset_index(drop=True)
    return df

# Memuat data ke aplikasi
try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat file 'data_panen.csv'. Pastikan file berada di direktori/folder yang sama dengan app.py. Error: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR FILTER WAKTU
# ==========================================
st.sidebar.header("⚙️ Filter Data")
min_year = int(df["Tanggal"].dt.year.min())
max_year = int(df["Tanggal"].dt.year.max())
tahun_terpilih = st.sidebar.slider("Pilih Rentang Tahun", min_year, max_year, (min_year, max_year))

# Filter dataset berdasarkan pilihan tahun user
df_filtered = df[(df["Tanggal"].dt.year >= tahun_terpilih[0]) & 
                 (df["Tanggal"].dt.year <= tahun_terpilih[1])]

# Tab Navigasi Utama
tab1, tab2, tab3 = st.tabs(["📊 Monitoring Risiko", "📈 Proyeksi Harga Gabah", "⚠️ Peringatan Dini"])

# ==========================================
# TAB 1: MONITORING RISIKO REAL-TIME
# ==========================================
with tab1:
    st.header("📊 Monitoring Risiko Real-Time")
    
    if not df_filtered.empty:
        # Mengambil baris data paling akhir dari hasil filter sebagai kondisi terkini
        data_terkini = df_filtered.iloc[-1]
        
        st.subheader(f"Kondisi Terakhir pada Data Filter ({data_terkini['Bulan']} {int(data_terkini['Tahun'])})")
        
        # Tampilan Grid Metrik Utama
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Curah Hujan", f"{data_terkini['Curah Hujan (mm)']:.1f} mm")
        col2.metric("Suhu Rata-rata", f"{data_terkini['Suhu Rata-rata (°C)']:.1f} °C")
        col3.metric("NTP (Nilai Tukar Petani)", f"{data_terkini['NTP']:.2f}")
        col4.metric("Harga Gabah (GKP)", f"Rp {data_terkini['Harga Gabah (Rp/Kg)']:,.0f}")

        st.markdown("---")
        
        # Visualisasi Grafis Tren Historis
        st.subheader("Grafik Tren Variabel Makro & Iklim")
        
        fig_harga_ntp = px.line(df_filtered, x="Tanggal", y=["Harga Gabah (Rp/Kg)", "NTP"], 
                                title="Pergerakan Harga Gabah vs Nilai Tukar Petani (NTP)")
        st.plotly_chart(fig_harga_ntp, use_container_width=True)
        
        fig_hujan = px.bar(df_filtered, x="Tanggal", y="Curah Hujan (mm)", 
                           title="Fluktuasi Curah Hujan Bulanan", color_discrete_sequence=['#2ecc71'])
        st.plotly_chart(fig_hujan, use_container_width=True)
    else:
        st.warning("Tidak ada data tersedia pada rentang tahun yang dipilih.")

# ==========================================
# TAB 2: PROYEKSI HARGA GABAH
# ==========================================
with tab2:
    st.header("📈 Proyeksi Harga Gabah Mendatang")
    st.markdown("Estimasi harga gabah tingkat petani menggunakan algoritma Regresi Linear berdasarkan data aktual iklim dan ekonomi.")
    
    # Membersihkan baris kosong khusus untuk pemodelan ML
    df_ml = df.dropna(subset=["Curah Hujan (mm)", "Suhu Rata-rata (°C)", "NTP", "Produksi Padi (Ton)", "Harga Gabah (Rp/Kg)"])
    
    if len(df_ml) > 5:
        # Menentukan fitur prediktor (X) dan target (y)
        X = df_ml[["Curah Hujan (mm)", "Suhu Rata-rata (°C)", "NTP", "Produksi Padi (Ton)"]]
        y = df_ml["Harga Gabah (Rp/Kg)"]
        
        # Training Model
        model = LinearRegression()
        model.fit(X, y)
        
        st.subheader("Simulasi Skenario Parameter")
        st.markdown("Sesuaikan nilai di bawah ini untuk melihat bagaimana perubahan parameter memengaruhi estimasi harga:")
        
        col_p1, col_p2 = st.columns(2)
        col_p3, col_p4 = st.columns(2)
        
        # Set nilai default menggunakan nilai rata-rata data historis
        input_hujan = col_p1.number_input("Curah Hujan Target (mm)", value=float(X["Curah Hujan (mm)"].mean()))
        input_suhu = col_p2.number_input("Suhu Rata-rata Target (°C)", value=float(X["Suhu Rata-rata (°C)"].mean()))
        input_ntp = col_p3.number_input("Indeks NTP Target", value=float(X["NTP"].mean()))
        input_produksi = col_p4.number_input("Volume Produksi Padi Target (Ton)", value=float(X["Produksi Padi (Ton)"].mean()))
        
        if st.button("Jalankan Proyeksi Harga"):
            hasil_prediksi = model.predict([[input_hujan, input_suhu, input_ntp, input_produksi]])
            st.success(f"### 💰 **Hasil Estimasi Harga Gabah:** Rp {hasil_prediksi[0]:,.2f} / Kg")
    else:
        st.error("Kuantitas baris data yang valid tidak mencukupi untuk melakukan pemodelan Machine Learning.")

# ==========================================
# TAB 3: PERINGATAN DINI (EARLY WARNING SYSTEM)
# ==========================================
with tab3:
    st.header("⚠️ Sistem Peringatan Dini")
    st.markdown("Pemberitahuan otomatis saat indikator berada di luar batas parameter ideal ketahanan pangan.")
    
    # Menentukan threshold / ambang batas bahaya (bisa dikustomisasi)
    # Contoh: Curah hujan ekstrem > 300mm, Suhu tinggi > 28.5°C, NTP rendah < 100
    kondisi_kritis = df_filtered[
        (df_filtered["Curah Hujan (mm)"] > 300) | 
        (df_filtered["Suhu Rata-rata (°C)"] > 28.5) | 
        (df_filtered["NTP"] < 100)
    ]
    
    if not kondisi_kritis.empty:
        st.error(f"🚨 Terdeteksi {len(kondisi_kritis)} periode anomali/risiko tinggi dalam rentang tahun yang difilter.")
        
        # Menampilkan hingga 5 rekam jejak kondisi kritis terbaru
        for indeks, baris in kondisi_kritis.tail(5).iterrows():
            label_waktu = f"{baris['Bulan']} {int(baris['Tahun'])}"
            st.markdown(f"**Periode: {label_waktu}**")
            
            if baris['Curah Hujan (mm)'] > 300:
                st.warning(f"🌧️ **Curah Hujan Tinggi:** Terdeteksi {baris['Curah Hujan (mm)']:.1f} mm. Waspada potensi banjir lahan pertanian.")
            if baris['Suhu Rata-rata (°C)'] > 28.5:
                st.warning(f"🔥 **Suhu Di Atas Normal:** Terdeteksi {baris['Suhu Rata-rata (°C)']:.1f} °C. Berisiko memicu kekeringan atau meningkatnya hama.")
            if baris['NTP'] < 100:
                st.warning(f"📉 **NTP Defisit:** Indeks NTP berada di tingkat {baris['NTP']:.2f}. Kesejahteraan petani berisiko menurun karena biaya produksi lebih besar dari pendapatan.")
            st.markdown("---")
    else:
        st.success("✅ Seluruh indikator berada dalam batas normal. Tidak ada aktivitas peringatan dini untuk rentang waktu ini.")