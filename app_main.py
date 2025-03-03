import streamlit as st

# Thiết lập cấu hình trang với layout rộng và biểu tượng
st.set_page_config(
    page_title="Multi-Page App",
    page_icon="📊",
    layout="wide",
)

# Sidebar với tiêu đề nổi bật
st.sidebar.markdown(
    """
    <h2 style='text-align: center; color: #FF4B4B;'>🧭 Điều hướng Ứng dụng</h2>
    <p style='text-align: center;'>Chọn một ứng dụng để trải nghiệm!</p>
    """,
    unsafe_allow_html=True
)

# Nội dung chính với phong cách đẹp hơn
st.title("🌟 Ứng dụng Đa năng với Streamlit")
st.markdown(
    """
    <div style='background-color: #F0F2F6; padding: 20px; border-radius: 10px;'>
        <h3 style='color: #1F77B4;'>📋 Danh sách Ứng dụng</h3>
        <p>Khám phá các ứng dụng thú vị bên dưới:</p>
        <ul style='list-style-type: none; padding-left: 0;'>
            <li>➡️ <strong style='color: #FF4B4B;'>Linear Regression</strong>: Phân tích hồi quy tuyến tính.</li>
            <li>➡️ <strong style='color: #FF4B4B;'>MNIST Classification</strong>: Phân loại chữ số viết tay.</li>
            <li>➡️ <strong style='color: #FF4B4B;'>Clustering Algorithms</strong>: Các thuật toán phân cụm.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Thêm một dòng footer nhỏ
st.markdown(
    "<p style='text-align: center; color: #888888; font-size: 12px;'>Được xây dựng với ❤️ bởi Streamlit</p>",
    unsafe_allow_html=True
)