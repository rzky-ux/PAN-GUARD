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
    df = pd.read_csv("data-panen.csv", header=3)
    
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
    
    # 1. Membersihkan format kolom nominal uang / volume produksi
    cols_to_clean = ["Harga Gabah (Rp/Kg)", "Produksi Padi (Ton)"]
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('Rp', '', regex=False).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 2. Memastikan kolom iklim dan NTP terkonversi menjadi numerik (Float)
    # Jika data menggunakan tanda koma (,) untuk desimal, diganti menjadi titik (.) agar bisa dibaca Python
    cols_to_numeric = ["Curah Hujan (mm)", "Suhu Rata-rata (°C)", "NTP"]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Mengurutkan data berdasarkan waktu secara kronologis
    df = df.sort_values("Tanggal").reset_index(drop=True)
    return df

# Memuat data ke aplikasi
try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat file 'data-panen.csv'. Pastikan file berada di direktori/folder yang sama dengan app.py. Error: {e}")
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
    
    # Bersihkan baris bernilai NaN khusus pada tampilan agar tidak error saat format string
    df_metrics = df_filtered.dropna(subset=["Curah Hujan (mm)", "Suhu Rata-rata (°C)", "NTP", "Harga Gabah (Rp/Kg)"])
    
    if not df_metrics.empty:
        # Mengambil baris data paling akhir dari hasil filter sebagai kondisi terkini
        data_terkini = df_metrics.iloc[-1]
        
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
        st.warning("Tidak ada data valid tersedia pada rentang tahun yang dipilih.")

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
# TAB 3: PERINGATAN DINI & REKOMENDASI MITIGASI
# ==========================================
with tab3:
    st.header("⚠️ Sistem Peringatan Dini & Rekomendasi")
    st.markdown("Pemberitahuan otomatis beserta panduan mitigasi saat indikator berada di luar batas parameter ideal ketahanan pangan.")
    
    # 1. Pastikan semua kolom yang mau dianalisis tidak kosong
    kolom_ewis = ["Curah Hujan (mm)", "Suhu Rata-rata (°C)", "NTP", "Harga Gabah (Rp/Kg)", "Produksi Padi (Ton)"]
    df_ewis = df_filtered.dropna(subset=kolom_ewis)
    
    # 2. Menentukan threshold / ambang batas bahaya untuk SEMUA variabel
    # Anda bisa menyesuaikan angka-angka ini sesuai dengan data historis riil di Lampung
    kondisi_kritis = df_ewis[
        (df_ewis["Curah Hujan (mm)"] > 300) |  # Curah hujan ekstrem tinggi
        (df_ewis["Curah Hujan (mm)"] < 100) |  # Curah hujan ekstrem rendah (kemarau)
        (df_ewis["Suhu Rata-rata (°C)"] > 28.5) | # Suhu kepanasan
        (df_ewis["NTP"] < 100) |               # Kesejahteraan defisit
        (df_ewis["Harga Gabah (Rp/Kg)"] < 5000) | # Harga gabah anjlok di bawah HPP
        (df_ewis["Produksi Padi (Ton)"] < 50000)  # Produksi anjlok (sesuaikan angkanya dengan skala data Anda)
    ]
    
    if not kondisi_kritis.empty:
        st.error(f"🚨 Terdeteksi {len(kondisi_kritis)} periode anomali/risiko tinggi dalam rentang tahun yang difilter.")
        
        # Menampilkan hingga 5 rekam jejak kondisi kritis terbaru
        for indeks, baris in kondisi_kritis.tail(5).iterrows():
            label_waktu = f"{baris['Bulan']} {int(baris['Tahun'])}"
            st.markdown(f"### **Periode: {label_waktu}**")
            
            # --- A. VARIABEL CURAH HUJAN ---
            if baris['Curah Hujan (mm)'] > 300:
                st.warning(f"🌧️ **Curah Hujan Ekstrem Tinggi:** ({baris['Curah Hujan (mm)']:.1f} mm) - Waspada genangan dan banjir lahan pertanian.")
                with st.expander("💡 Rekomendasi Mitigasi Banjir"):
                    st.markdown("""
                    * **Tata Kelola Air:** Segera bersihkan dan perdalam saluran drainase/irigasi agar air cepat surut.
                    * **Panen Dini:** Jika padi sudah masuk fase masak kuning (>80%), percepat jadwal panen untuk menghindari gabah membusuk.
                    """)
            elif baris['Curah Hujan (mm)'] < 100:
                st.warning(f"🏜️ **Curah Hujan Rendah / Kemarau:** ({baris['Curah Hujan (mm)']:.1f} mm) - Waspada kekeringan lahan sawah.")
                with st.expander("💡 Rekomendasi Mitigasi Kekeringan"):
                    st.markdown("""
                    * **Pompanisasi Irigasi:** Manfaatkan pompa air untuk mengambil sumber air dari sungai terdekat atau sumur bor (sumur pantek).
                    * **Alih Komoditas Sementara:** Jika air sangat minim, pertimbangkan beralih menanam palawija (jagung/kedelai) yang lebih hemat air.
                    """)
            
            # --- B. VARIABEL SUHU RATA-RATA ---
            if baris['Suhu Rata-rata (°C)'] > 28.5:
                st.warning(f"🔥 **Suhu Di Atas Normal:** ({baris['Suhu Rata-rata (°C)']:.1f} °C) - Berisiko memicu meledaknya populasi hama dan penyakit.")
                with st.expander("💡 Rekomendasi Antisipasi Hama & Suhu"):
                    st.markdown("""
                    * **Efisiensi Irigasi (AWD):** Terapkan sistem pengairan berselang (*Alternate Wetting and Drying*) untuk menjaga kelembapan mikroklimat tanaman.
                    * **Waspada Hama:** Tingkatkan frekuensi pengecekan lahan (scouting) terhadap wereng cokelat dan penggerek batang.
                    """)
            
            # --- C. VARIABEL NTP (NILAI TUKAR PETANI) ---
            if baris['NTP'] < 100:
                st.warning(f"📉 **NTP Defisit:** ({baris['NTP']:.2f}) - Biaya produksi dan kebutuhan hidup petani lebih tinggi dari pendapatan panen.")
                with st.expander("💡 Rekomendasi Keputusan Ekonomi Petani"):
                    st.markdown("""
                    * **Efisiensi Pupuk:** Kurangi penggunaan pupuk kimia yang mahal, substitusi sebagian dengan pupuk organik (kompos jerami) untuk menekan biaya produksi.
                    * **Diversifikasi Pendapatan:** Terapkan sistem mina padi atau tumpangsari di pematang sawah untuk tambahan penghasilan harian.
                    """)
            
            # --- D. VARIABEL HARGA GABAH ---
            if baris['Harga Gabah (Rp/Kg)'] < 5000:
                st.warning(f"💰 **Harga Gabah Anjlok:** (Rp {baris['Harga Gabah (Rp/Kg)']:,.0f}/Kg) - Harga jual berada di tingkat yang sangat merugikan petani.")
                with st.expander("💡 Rekomendasi Strategi Penjualan"):
                    st.markdown("""
                    * **Sistem Resi Gudang (Tunda Jual):** Simpan GKG (Gabah Kering Giling) di lumbung atau KUD. Jangan langsung dijual saat panen raya; tunggu hingga pasokan pasar berkurang dan harga naik.
                    * **Penyerapan Bulog:** Hubungi satgas pangan daerah atau mitra Bulog untuk penyerapan gabah sesuai HPP (Harga Pembelian Pemerintah) guna menghindari tengkulak.
                    * **Peningkatan Nilai Tambah:** Jika memungkinkan, giling gabah menjadi beras premium secara mandiri atau berkelompok untuk margin keuntungan yang lebih besar.
                    """)
                    
            # --- E. VARIABEL PRODUKSI PADI ---
            # Asumsi: Jika produksi bulanan turun di bawah 50.000 ton, sistem memberi peringatan gagal panen/paceklik.
            if baris['Produksi Padi (Ton)'] < 50000:
                st.warning(f"🌾 **Volume Produksi Kritis:** ({baris['Produksi Padi (Ton)']:,.0f} Ton) - Indikasi paceklik atau risiko gagal panen masif.")
                with st.expander("💡 Rekomendasi Antisipasi Paceklik"):
                    st.markdown("""
                    * **Klaim Asuransi (AUTP):** Segera ajukan klaim Asuransi Usaha Tani Padi jika penurunan produksi disebabkan oleh hama atau bencana alam.
                    * **Bantuan Benih/Pupuk:** Kelompok tani agar segera berkoordinasi dengan Dinas Pertanian setempat untuk mendapatkan bantuan benih cadangan nasional (CBP) untuk musim tanam berikutnya.
                    """)
            
            st.markdown("---")
    else:
        st.success("✅ Seluruh indikator berada dalam batas normal. Tidak ada peringatan dini yang aktif pada rentang waktu ini.")