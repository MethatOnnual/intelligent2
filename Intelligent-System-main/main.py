import streamlit as st
import pandas as pd
import home  # ตรวจสอบว่า home.py อยู่ในโฟลเดอร์เดียวกัน
import about # ตรวจสอบว่า about.py อยู่ในโฟลเดอร์เดียวกัน
import app   # นำเข้า app.py (ใช้สำหรับ CSV)
import demo_ml  # ✅ ตรวจสอบว่า demo_ml.py อยู่ในโฟลเดอร์เดียวกัน
import demo_nn  # ✅ ตรวจสอบว่า demo_nn.py อยู่ในโฟลเดอร์เดียวกัน

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI & ML Guide", page_icon="🤖", layout="wide")

# สร้างเมนู Sidebar
st.sidebar.header("เมนู", divider=True)
page = st.sidebar.radio("ไปที่", ["🏠หน้าหลัก","📊 app", "Demo ML", "Demo NN"])

# 👉 ตรวจสอบให้แน่ใจว่าเรียกใช้ `show()` จากแต่ละไฟล์ถูกต้อง
if page == "🏠หน้าหลัก":
    home.show()  # ✅ แก้ไขให้เรียก demo_nn.show()
elif page == "📊 app":
    app.show()  # ✅ แก้ไขให้เรียก demo_nn.show()  
elif page == "Demo ML":
    demo_ml.show()  # ✅ ตรวจสอบว่ามีฟังก์ชัน show() จริงใน demo_ml.py

elif page == "Demo NN":
    demo_nn.show()  # ✅ แก้ไขให้เรียก demo_nn.show()

# Footer
st.sidebar.markdown("---")
st.sidebar.header("Intelligent System Project", divider=True)
st.sidebar.header("Member")
st.sidebar.write("นางสาว อรธนัท ปิ่นทองธรรม 6404062616047")
st.sidebar.write("นาย เมธัส อ่อนนวล 6404062616055")
