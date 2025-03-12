import streamlit as st

# Thiết lập cấu hình trang với layout rộng và biểu tượng
st.set_page_config(
    page_title="Multi-Page App",
    page_icon="📊",
    layout="wide",
)

# Sidebar với nền đen và chữ trắng
st.sidebar.markdown(
    """
    <div style='background-color: #1E1E1E; padding: 10px; border-radius: 5px;'>
        <h2 style='text-align: center; color: #FFFFFF;'>🧭 Điều hướng Ứng dụng</h2>
        <p style='text-align: center; color: #D3D3D3;'>Chọn một ứng dụng để trải nghiệm!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Nội dung chính với nền đen và chữ trắng
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
            <li>➡️ <strong style='color: #FF6F61;'>Assignment - PCA & t-SNE MNIST</strong>: Các kỹ thuật giảm chiều dữ liệu trên tập MNIST.</li>
            <li>➡️ <strong style='color: #FF6F61;'>NeuralNetwork_MNIST</strong>: .</li>

        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
# Footer với nền đen và chữ xám nhạt
st.markdown(
    "<p style='text-align: center; color: #A9A9A9; font-size: 12px;'>Được xây dựng với Streamlit</p>",
    unsafe_allow_html=True
)
