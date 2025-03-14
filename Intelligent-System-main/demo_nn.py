import streamlit as st
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# ตั้งค่า path ของไฟล์โมเดล
MODEL_PATH = "neural_network_model.h5"

# ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_keras_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            # st.sidebar.success("✅ โหลดโมเดลสำเร็จ!")
            return model
        except Exception as e:
            st.sidebar.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
            return None
    else:
        st.sidebar.warning("⚠️ ไม่พบไฟล์โมเดล โปรดอัปโหลดไฟล์ก่อน!")
        return None

# ฟังก์ชันหลักสำหรับ Streamlit
def show():
    st.title("🏦 การพยากรณ์ลูกค้าที่จะออกจากธนาคาร")
    st.write("🔍 กรอกข้อมูลเพื่อให้โมเดลคาดการณ์ว่าลูกค้าจะออกจากธนาคารหรือไม่")

    # โหลดโมเดล
    model = load_keras_model()

    # ฟีเจอร์ที่ใช้และคำอธิบายภาษาไทย
    feature_labels = {
        "CreditScore": "🎯 คะแนนเครดิต",
        "Age": "👤 อายุ (ปี)",
        "Tenure": "📅 ระยะเวลาที่เป็นลูกค้า (ปี)",
        "Balance": "💰 ยอดเงินคงเหลือในบัญชี",
        "NumOfProducts": "📦 จำนวนผลิตภัณฑ์ที่ใช้",
        "HasCrCard": "💳 มีบัตรเครดิตหรือไม่ (0=ไม่มี, 1=มี)",
        "IsActiveMember": "🟢 เป็นสมาชิกที่ใช้งานอยู่หรือไม่ (0=ไม่ใช่, 1=ใช่)",
        "EstimatedSalary": "💵 รายได้ประมาณการ",
        "Geography_France": "🇫🇷 อยู่ในฝรั่งเศสหรือไม่ (0=ไม่ใช่, 1=ใช่)",
        "Geography_Germany": "🇩🇪 อยู่ในเยอรมนีหรือไม่ (0=ไม่ใช่, 1=ใช่)",
        "Geography_Spain": "🇪🇸 อยู่ในสเปนหรือไม่ (0=ไม่ใช่, 1=ใช่)",
        "Gender_Male": "🚹 เพศชายหรือไม่ (0=ไม่ใช่, 1=ใช่)"
    }

    # จัดเรียงเป็น Grid Layout
    st.subheader("📊 กรอกข้อมูลลูกค้า")
    feature_values = []
    cols = st.columns(3)  # ใช้ 3 คอลัมน์

    feature_names = list(feature_labels.keys())  # ดึงชื่อฟีเจอร์ทั้งหมด

    for i, feature_name in enumerate(feature_names):
        with cols[i % 3]:  # จัดวาง input ให้อยู่ใน 3 คอลัมน์
            feature = st.number_input(
                f"{feature_labels[feature_name]}",
                min_value=0.0, max_value=1000000.0, value=00.0
            )
            feature_values.append(feature)

    # แปลงค่า input เป็นรูปแบบที่โมเดลต้องการ
    input_data = np.array([feature_values]).reshape(1, len(feature_names))

    # พยากรณ์ผลลัพธ์เมื่อกดปุ่ม
    st.subheader("🔮 ผลลัพธ์การพยากรณ์")
    if model is not None:
        if st.button("📊 ทำนายผล", use_container_width=True):
            prediction = model.predict(input_data)[0][0]
            result_text = "✅ ลูกค้ามีแนวโน้ม **คงอยู่** กับธนาคาร" if prediction < 0.5 else "❌ ลูกค้ามีแนวโน้ม **จะออกจากธนาคาร**"
            # st.success(f"📌 ผลลัพธ์โมเดล: {prediction:.4f}")
            st.warning(result_text)
    else:
        st.warning("⚠️ กรุณาอัปโหลดไฟล์โมเดลก่อนทำการพยากรณ์")

  
