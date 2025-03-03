import streamlit as st

# Cấu hình trang
st.set_page_config(page_title="Multi-Page App", page_icon="📊", layout="wide")

# Sidebar điều hướng
st.sidebar.title("🧭 Điều hướng Ứng dụng")
menu = st.sidebar.radio(
    "Chọn ứng dụng:",
    ["🌟 Trang Chính", "📊 Linear Regression", "🖊️ MNIST Classification", "🔍 Clustering Algorithms"],
)

# Xử lý nội dung trang chính
if menu == "🌟 Trang Chính":
    st.title("🌟 Ứng dụng Đa năng với Streamlit")
    st.write("Chào mừng bạn đến với ứng dụng đa trang!")

elif menu == "📊 Linear Regression":
    from pages import linear_regression
    linear_regression.show()

elif menu == "🖊️ MNIST Classification":
    from pages import mnist_classification
    mnist_classification.show()

elif menu == "🔍 Clustering Algorithms":
    from pages import clustering_algorithms
    clustering_algorithms.show()
