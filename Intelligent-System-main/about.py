import streamlit as st
import about as about

    
def show():
   
    st.header("Welcome to Intellgent System and ML & Neural Network Web")

    # 👉 สร้าง Tabs Menu
    tab1, tab2, tab3 = st.tabs(["📌 บทนำ", "📊 ข้อมูล", "⚙️ ตั้งค่า"])