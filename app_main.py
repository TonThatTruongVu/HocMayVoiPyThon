import streamlit as st

# Thiết lập cấu hình trang với layout rộng và biểu tượng
st.set_page_config(
    page_title="Multi-Page App",
    page_icon="📊",
    layout="wide",
)

# Sidebar với giao diện tùy chỉnh
st.sidebar.markdown(
    """
    <div style='background-color: #1E1E1E; padding: 10px; border-radius: 5px; text-align: center;'>
        <h2 style='color: #FFFFFF;'>🧭 Điều hướng Ứng dụng</h2>
        <p style='color: #D3D3D3;'>Chọn một ứng dụng để trải nghiệm!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Tạo thanh chọn ứng dụng trong sidebar
menu = st.sidebar.radio(
    "Chọn ứng dụng:",
    ["🌟 Trang Chính", "📊 Linear Regression", "🖊️ MNIST Classification", "🔍 Clustering Algorithms"],
)

# Hiển thị nội dung tương ứng
if menu == "🌟 Trang Chính":
    st.title("🌟 Ứng dụng Đa năng với Streamlit")
    st.markdown(
        """
        <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px;'>
            <h3 style='color: #4DA8DA;'>📋 Danh sách Ứng dụng</h3>
            <p style='color: #D3D3D3;'>Khám phá các ứng dụng thú vị bên dưới:</p>
            <ul style='list-style-type: none; padding-left: 0; color: #FFFFFF;'>
                <li>➡️ <strong style='color: #FF6F61;'>Linear Regression</strong>: Phân tích hồi quy tuyến tính.</li>
                <li>➡️ <strong style='color: #FF6F61;'>MNIST Classification</strong>: Phân loại chữ số viết tay.</li>
                <li>➡️ <strong style='color: #FF6F61;'>Clustering Algorithms</strong>: Các thuật toán phân cụm.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

elif menu == "📊 Linear Regression":
    st.title("📊 Linear Regression")
    st.write("Nội dung về hồi quy tuyến tính...")

elif menu == "🖊️ MNIST Classification":
    st.title("🖊️ MNIST Classification")
    st.write("Nội dung về phân loại chữ số viết tay...")

elif menu == "🔍 Clustering Algorithms":
    st.title("🔍 Clustering Algorithms")
    st.write("Nội dung về thuật toán phân cụm...")

# Footer với nền đen và chữ xám nhạt
st.markdown(
    "<p style='text-align: center; color: #A9A9A9; font-size: 12px;'>Được xây dựng với Streamlit</p>",
    unsafe_allow_html=True
)
