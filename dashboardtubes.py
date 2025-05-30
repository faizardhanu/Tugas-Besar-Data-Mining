# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# Konfigurasi halaman
st.set_page_config(page_title="Diabetes Analysis", layout="centered")

st.title("🧬 Dashboard Analisis Risiko Diabetes")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Data Awal")
    st.write(df.head())

    # Encode kolom kategorikal
    df_copy = df.copy()
    le_dict = {}
    for col in df_copy.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col])
        le_dict[col] = le

    # Fitur dan standardisasi
    selected_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease', 'smoking_history']
    X = df_copy[selected_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method
    st.subheader("📈 Elbow Method")
    inertia = []
    range_k = range(2, 11)
    for k in range_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range_k, inertia, 'bo-')
    ax1.set_xlabel("Jumlah Cluster (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    st.pyplot(fig1)

    # Clustering dan PCA
    st.subheader("🔍 Visualisasi Cluster (PCA)")
    k = st.slider("Pilih jumlah cluster", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_copy['cluster'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_copy['PC1'] = X_pca[:, 0]
    df_copy['PC2'] = X_pca[:, 1]

    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df_copy, x='PC1', y='PC2', hue='cluster', palette='Set2', ax=ax2)
    ax2.set_title("PCA Clustering")
    st.pyplot(fig2)

    st.subheader("📋 Rata-rata Atribut per Cluster")
    st.write(df_copy.groupby('cluster')[selected_cols].mean().round(2))

    # Prediksi Logistic Regression
    if 'diabetes' in df.columns:
        st.subheader("🔎 Prediksi Diabetes (Logistic Regression)")
        y = df_copy['diabetes']
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_scaled, y)

        # Form input prediksi
        st.markdown("### Input Atribut Kesehatan:")
        age = st.slider("Umur", 1, 100, 30)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        hba1c = st.slider("HbA1c", 3.0, 15.0, 6.0)
        glucose = st.slider("Glukosa Darah", 70, 300, 120)
        hypertension = st.radio("Hipertensi", [0, 1])
        heart_disease = st.radio("Penyakit Jantung", [0, 1])
        smoking = st.selectbox("Riwayat Merokok", options=list(le_dict['smoking_history'].classes_))
        smoking_encoded = le_dict['smoking_history'].transform([smoking])[0]

        # Prediksi
        input_data = np.array([[age, bmi, hba1c, glucose, hypertension, heart_disease, smoking_encoded]])
        input_scaled = scaler.transform(input_data)
        prob = logreg.predict_proba(input_scaled)[0][1]
        pred = logreg.predict(input_scaled)[0]

        st.metric("Probabilitas Diabetes", f"{prob:.2%}")
        st.success("✅ Positif Diabetes" if pred == 1 else "❎ Negatif Diabetes")
