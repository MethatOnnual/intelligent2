import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# 📌 โหลด Dataset
DATASET_PATH = "FINAL_USO.csv"
df = pd.read_csv(DATASET_PATH)

# 📌 โหลดโมเดล KNN
KNN_MODEL_PATH = "knn_gold_model.pkl"

def load_knn_model():
    if os.path.exists(KNN_MODEL_PATH):
        return joblib.load(KNN_MODEL_PATH)
    return None

knn_model = load_knn_model()

# ✅ สร้างฟังก์ชัน `show()` และใส่โค้ดทั้งหมดไว้ข้างใน
def show():
   
    # 🎯 Header
    st.title("📈 วิเคราะห์ตลาดหุ้นด้วย KNN")
    st.markdown("### 🌍 แสดงแนวโน้มราคาหุ้นและข้อมูลการซื้อขาย")

    # 📌 ใช้ Columns ให้ UI ดูดี
    col1, col2 = st.columns(2)

    # 🔹 กราฟแนวโน้มราคาหุ้น
    with col1:
        st.subheader("📊 แนวโน้มราคาหุ้น")
        fig = px.line(df, x='Date', y=['USO_Close', 'SP_high', 'GDX_Close'], title="แนวโน้มราคาหุ้น")
        st.plotly_chart(fig, use_container_width=True)

    # 🔹 กราฟ Volume
    with col2:
        st.subheader("💰 ปริมาณการซื้อขาย")
        fig_volume = px.line(df, x='Date', y=['USO_Volume', 'GDX_Volume'], title="ปริมาณการซื้อขาย")
        st.plotly_chart(fig_volume, use_container_width=True)

    # 🎯 Stock Price Prediction (KNN Model)
    st.title("🔮 ทำนายราคาหุ้น (KNN Model)")
    st.markdown("กรอกข้อมูลเพื่อทำนายราคาหุ้น")

    # 📌 UI ให้ดูทันสมัยด้วย Columns
    col1, col2 = st.columns(2)

    with col1:
        feature1 = st.number_input("📈 เปิดตลาด (Open)", min_value=0.0, max_value=500.0, value=160.0)
        feature2 = st.number_input("📊 ราคาสูงสุด (High)", min_value=0.0, max_value=500.0, value=180.0)

    with col2:
        feature3 = st.number_input("📉 ราคาต่ำสุด (Low)", min_value=0.0, max_value=500.0, value=150.0)
        feature4 = st.number_input("💰 ปริมาณซื้อขาย (Volume)", min_value=0.0, max_value=100000000.0, value=8000000.0)

    input_data = np.array([[feature1, feature2, feature3, feature4]])

    # 🔘 ปุ่มทำนาย
    if st.button("🔮 ทำนายราคา"):
        if knn_model:
            prediction = knn_model.predict(input_data)[0]
            st.success(f"📊 KNN ทำนายราคาปิด: {prediction:.2f}")
        else:
            st.error("❌ ไม่พบโมเดล KNN กรุณาตรวจสอบไฟล์ scaler_gold.pkl")

    
