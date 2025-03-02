import streamlit as st

st.set_page_config(
    page_title="Multi-Page App",
    page_icon="📊",
    layout="wide",
)

st.sidebar.title("🏠 Home")
st.sidebar.write("Chọn ứng dụng từ sidebar!")


st.write(" Ứng dụng: ")
st.markdown("- **Linear Regression**")