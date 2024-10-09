# Proyek Analisis Data: [Bike Sharing Dataset]
# Nama: [Umi Inayatul Hidayah]
# Email: [umihidayah003@gmail.com]
# ID Dicoding: [Umi Inayatul Hidayah]

# Menentukan Pertanyaan:
# 1. Pada jam berapa terjadi penyewaan sepeda dengan jumlah tertinggi?
# 2. Apakah penyewaan sepeda terpengearuh oleh kondisi cuaca?


# Menyiapkan DataFrame
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st 
from babel.numbers import format_currency
from sklearn.preprocessing import StandardScaler
sns.set(style='dark')

# Memuat dataset yang diunggah
hour_df = pd.read_csv('hour.csv')
day_df = pd.read_csv('day.csv')

# Tampilkan beberapa baris pertama dari dataset
print("Dataset Hour:")
print(hour_df.head())
print("\nDataset Day:")
print(day_df.head())

hour_df.info()
day_df.info()

# memeriksa nilai null pada dataset
print("\nNilai Null pada Dataset Hour:")
print(hour_df.isnull().sum())
print("\nNilai Null pada Dataset Day:")
print(day_df.isnull().sum())

# Penghapusan Duplikasi Jika Ada
hour_df.drop_duplicates(inplace=True)
day_df.drop_duplicates(inplace=True)

# Memastikan Tidak Terdapat Missing Values
hour_df.fillna(0, inplace=True)
day_df.fillna(0, inplace=True)

# Hitung Rata-Rata Penyewaan Tiap Jam
hourly_rentals = hour_df.groupby('hr')['cnt'].mean().reset_index()

# Pembuatan Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=hourly_rentals, x='hr', y='cnt', palette='Blues_d')
plt.title('Rata-Rata Penyewaan Sepeda per Jam')
plt.xlabel('Jam')
plt.ylabel('Rata-Rata Penyewaan')
plt.xticks(rotation=45)
plt.show()

"""### Pertanyaan 2:"""

# Memvisualisasikan Hubungan Variabel penyewaan sepeda dengan cuaca
plt.figure(figsize=(14,6))

# Melakukan Scatter Plot untuk penyewaan sepeda dan suhu
plt.subplot(1, 2, 1)
sns.scatterplot(data=day_df, x='temp', y='cnt')
plt.title('Pengaruh suhu terhadap penyewaan sepeda')
plt.xlabel('Suhu(Normalized)')
plt.ylabel('Total Penyewaan')

# Melakukan Scatter plot untuk penyewaan sepeda dan kelembaban
plt.subplot(1, 2, 2)
sns.scatterplot(data=day_df, x='hum', y='cnt')
plt.title('Pengaruh kelembaban terhadap total penyewaan sepeda')
plt.xlabel('Kelembaban (Normalized)')
plt.ylabel('Total Penyewaan')

plt.tight_layout()
plt.show()

# Penerapan Clustering dengan K-Means
# Skala data sebelum dilakukan clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(hourly_rentals[['cnt']])

# Menerapkan K-Means Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Melakukan scalling pada data
scaler = StandardScaler()
features = ['hr', 'cnt']
hourly_data_scaled = scaler.fit_transform(hourly_rentals[features])

# Penerapan K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
hourly_rentals['cluster'] = kmeans.fit_predict(hourly_data_scaled)

# Hasil cluster
hourly_rentals.head()

# Visualisasi Hasil Dari Clustering
sns.scatterplot(x='hr', y='cnt', hue='cluster', data=hourly_rentals, palette='cool')
plt.title('Clustering Penyewaan Sepeda per Jam')
plt.xlabel('Jam')
plt.ylabel('Rata-Rata Penyewaan')
plt.show()

# """## Conclusion

# - Berdasarkan pertanyaan pertama maka penyewaan sepeda paling tertinggi terjadi pada jam sibuk yaitu sekitar pukul 08.00 pagi dan 05.00 - 06.00 sore
# - Berdasarkan pertanyaan kedua maka penyewaan sepeda menunjukkan korelasi yang positif dengan faktor suhu, yaitu dimana saat cuaca yang lebih hangat maka jumlah penyewaan sepeda akan meningkat. Akan tetapi, kelembaban tidak menunjukkan hubungan yang kuat.

# Baca Dataset
hourly_rentals = pd.read_csv('hour.csv')

# Bagian judul
st.title('Dashboard Penyewaan Sepeda')

# Bagian Sidebar
st.sidebar.title('Pengaturan Tampilan')

# Filter Pemilihan jam berdasarkan rentang tertentu
jam_rentang = st.sidebar.slider('Pilih rentang jam', min_value=0, max_value=23, value=(0, 23))

# tampilan informasi ringkas
st.write('### Ringkasan Data')
st.write(hourly_rentals.describe())

# filter berdasarkan jam
filtered_data = hourly_rentals[(hourly_rentals['hr'] >= jam_rentang[0]) & (hourly_rentals['hr'] <= jam_rentang[1])]

# Plot rata rata penyewaan sepeda tiap sejam
st.write('### Rata-Rata Penyewaan Sepeda per Jam')
plt.figure(figsize=(10, 6))
sns.barplot(data=filtered_data, x='hr', y='cnt', palette='Blues_d')
plt.title('Rata-Rata Penyewaan Sepeda per Jam')
plt.xlabel('Jam')
plt.ylabel('Jumlah Penyewaan')
st.pyplot(plt)

# Plot pengaruh suhu terhadap penyewaan sepeda
st.write('### Pengaruh Suhu terhadap Penyewaan Sepeda ')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=hourly_rentals, x='temp', y='cnt', hue='weathersit', palette='coolwarm')
plt.title('Pengaruh Suhu terhadap Penyewaan Sepeda')
plt.xlabel('Suhu (Normalized)')
plt.ylabel('Jumlah Penyewaan')
st.pyplot(plt)

# Menampilkan data filter
st.write('### Data Penyewaan Sepeda pe Jam {jam_rentang[0]} sampai {jam_rentang[1]}')
st.dataframe(filtered_data)

# Menambahkan kesimpulan
st.write('### Kesimpulan:')
st.write('Berdasarkan analisis, penyewaan sepeda cenderung tinggi terjadi pada jam sibuk yaitu sekitar pukul 08.00 pagi dan 05.00 - 06.00 sore dan dipengaruhi oleh suhu serta kondisi cuaca')

# link akses streamlit 
# https://dicoding1-jrqzg79p39gbp2je4sezlb.streamlit.app/ 
