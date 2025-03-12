import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import mlflow
import os
import time
from datetime import datetime
from mlflow.tracking import MlflowClient

# Hàm kết nối MLflow
def input_mlflow():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/TonThatTruongVu/MNIST_PCA_t-SNE.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "TonThatTruongVu"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "519c4a864e131de52197f54d170c130beb15ffd5"
    mlflow.set_experiment("MNIST_PCA_t_SNE")
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI

# Hàm tải dữ liệu MNIST từ OpenML
@st.cache_data
def load_mnist_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32) / 255.0
    return X, y

# Tab hiển thị dữ liệu
def data():
    st.header("📘 Dữ Liệu MNIST từ OpenML")
    
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.X = None
        st.session_state.y = None

    if st.button("⬇️ Tải dữ liệu từ OpenML"):
        with st.spinner("⏳ Đang tải dữ liệu MNIST từ OpenML..."):
            X, y = load_mnist_data()
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.data_loaded = True
            st.success("✅ Dữ liệu đã được tải thành công!")

    if st.session_state.data_loaded:
        X, y = st.session_state.X, st.session_state.y
        st.write("""
            **Thông tin tập dữ liệu MNIST:**
            - Tổng số mẫu: {}
            - Kích thước mỗi ảnh: 28 × 28 pixels (784 đặc trưng)
            - Số lớp: 10 (chữ số từ 0-9)
        """.format(X.shape[0]))

        st.subheader("Một số hình ảnh mẫu")
        n_samples = 10
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        for i, idx in enumerate(indices):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(X[idx].reshape(28, 28), cmap='gray')
            axes[row, col].set_title(f"Label: {y[idx]}")
            axes[row, col].axis("off")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("ℹ️ Nhấn nút 'Tải dữ liệu từ OpenML' để tải và hiển thị dữ liệu.")

# Hàm giải thích PCA
def explain_pca():
    st.markdown("## 🧠 PCA - Phân tích Thành phần Chính")

    st.markdown("""
    **PCA (Principal Component Analysis)** là một kỹ thuật giảm chiều tuyến tính giúp chuyển dữ liệu từ không gian nhiều chiều sang không gian ít chiều hơn, đồng thời giữ lại phần lớn thông tin quan trọng (phương sai).  
    - **Mục tiêu**: Tìm các hướng chính (principal components) mà dữ liệu biến thiên nhiều nhất, sau đó chiếu dữ liệu lên các hướng này.
    - **Ứng dụng**: Trực quan hóa dữ liệu, giảm kích thước dữ liệu để tăng tốc các thuật toán học máy.
    """)

    st.markdown("### 🔹 **PCA hoạt động như thế nào?**")
    st.markdown("""
    Hãy tưởng tượng bạn có dữ liệu 2D với các điểm nằm rải rác nhưng chủ yếu phân bố theo một hướng chéo. PCA sẽ tìm hướng chính mà dữ liệu biến thiên mạnh nhất và biến đổi dữ liệu sang hệ tọa độ mới dựa trên hướng đó.
    """)

    np.random.seed(42)
    x = np.random.rand(100) * 10
    y = x * 0.8 + np.random.randn(100) * 2
    X = np.column_stack((x, y))

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], color="blue", alpha=0.5, label="Dữ liệu ban đầu")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    **Hình trên**: Dữ liệu phân bố chủ yếu theo hướng chéo. PCA sẽ tìm ra hướng này để giảm chiều từ 2D xuống 1D.
    """)

    st.markdown("### 🔹 **Các bước thực hiện PCA**")
    st.markdown("""
    1. **Chuẩn hóa dữ liệu (trừ trung bình)**  
       - Tính trung bình của từng chiều:  
         $$ \\mu = \\frac{1}{n} \\sum_{i=1}^{n} x_i $$  
       - Dịch chuyển dữ liệu về gốc tọa độ:  
         $$ X_{\\text{norm}} = X - \\mu $$  
       - **Mục đích**: Đảm bảo trung tâm dữ liệu nằm tại $(0, 0)$, giúp phân tích không bị lệch.

    2. **Tính ma trận hiệp phương sai**  
       - Công thức:  
         $$ C = \\frac{1}{n-1} X_{\\text{norm}}^T X_{\\text{norm}} $$  
       - **Ý nghĩa**:  
         - $C_{ii}$ (đường chéo): Phương sai của chiều $i$.  
         - $C_{ij}$ (ngoài đường chéo): Hiệp phương sai giữa chiều $i$ và $j$, đo mức độ tương quan.

    3. **Tìm trị riêng và vector riêng**  
       - Giải phương trình:  
         $$ C v = \\lambda v $$  
       - Trong đó:  
         - $\\lambda$: Trị riêng, biểu thị độ lớn phương sai theo hướng tương ứng.  
         - $v$: Vector riêng, biểu thị hướng của thành phần chính.

    4. **Chọn thành phần chính**  
       - Sắp xếp các trị riêng từ lớn đến nhỏ, chọn $k$ trị riêng lớn nhất và vector riêng tương ứng để tạo ma trận $U_k$:  
         $$ U_k = [v_1, v_2, ..., v_k] $$

    5. **Chiếu dữ liệu lên không gian mới**  
       - Công thức:  
         $$ X_{\\text{new}} = X_{\\text{norm}} U_k $$  
       - Kết quả là dữ liệu mới với số chiều giảm xuống $k$.
    """)

    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], color="blue", alpha=0.5, label="Dữ liệu ban đầu")
    origin = np.mean(X, axis=0)
    for i in range(2):
        ax.arrow(origin[0], origin[1], eigenvectors[0, i] * 3, eigenvectors[1, i] * 3,
                 head_width=0.3, head_length=0.3, color="red", label=f"Trục {i+1}")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    **Hình trên**: Các mũi tên đỏ là các trục chính mà PCA tìm ra. Trục dài hơn (Trục 1) là hướng có phương sai lớn nhất.
    """)

    st.markdown("### ✅ **Ưu điểm của PCA**")
    st.markdown("""
    - Giảm chiều hiệu quả, giữ được thông tin chính (phương sai lớn).
    - Tăng tốc xử lý cho các mô hình học máy.
    - Loại bỏ nhiễu bằng cách bỏ qua các chiều có phương sai nhỏ.
    """)

    st.markdown("### ❌ **Nhược điểm của PCA**")
    st.markdown("""
    - Chỉ hiệu quả với dữ liệu có cấu trúc tuyến tính.
    - Các thành phần chính không còn ý nghĩa trực quan như đặc trưng gốc.
    - Nhạy cảm với dữ liệu chưa chuẩn hóa (cần scale trước nếu các chiều có đơn vị khác nhau).
    """)
    
def explain_tsne():
    st.markdown("## 🌌 t-SNE - Giảm chiều Phi tuyến")

    st.markdown("""
    **t-SNE (t-Distributed Stochastic Neighbor Embedding)** là một kỹ thuật giảm chiều phi tuyến, tập trung vào việc bảo toàn cấu trúc cục bộ của dữ liệu (khoảng cách giữa các điểm gần nhau).  
    - **Mục tiêu**: Chuyển dữ liệu từ không gian cao chiều (ví dụ: 784 chiều của MNIST) xuống 2D hoặc 3D để trực quan hóa.
    - **Ứng dụng**: Chủ yếu dùng để khám phá và hiển thị dữ liệu phức tạp.
    """)

    st.markdown("### 🔹 **Tham số quan trọng trong t-SNE**")
    st.markdown("""
    - **`n_components`**:  
      - **Ý nghĩa**: Số chiều mà dữ liệu sẽ được giảm xuống (thường là 2 hoặc 3 để trực quan hóa).  
      - **Giá trị**: Một số nguyên dương (ví dụ: 2 cho 2D, 3 cho 3D).  
      - **Tác động**:  
        - 2 hoặc 3: Phù hợp để vẽ biểu đồ trực quan.  
        - Không hỗ trợ giá trị lớn hơn vì t-SNE chủ yếu dùng cho trực quan hóa.  
      - **Trong code này**: Bạn chọn từ 1 đến 3 để hiển thị dữ liệu dưới dạng 1D, 2D, hoặc 3D.
    """)

    st.markdown("### 🔹 **t-SNE hoạt động như thế nào?**")
    st.markdown("""
    t-SNE không tìm các hướng tuyến tính như PCA, mà cố gắng giữ các điểm gần nhau trong không gian gốc cũng gần nhau trong không gian mới. Nó làm điều này bằng cách so sánh và tối ưu hóa phân phối xác suất giữa hai không gian.
    """)

    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) + np.array([2, 2])
    cluster2 = np.random.randn(50, 2) + np.array([-2, -2])
    X = np.vstack((cluster1, cluster2))

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=['blue']*50 + ['orange']*50, alpha=0.5, label="Dữ liệu ban đầu")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend(["Cụm 1", "Cụm 2"])
    st.pyplot(fig)

    st.markdown("""
    **Hình trên**: Dữ liệu 2D với hai cụm. t-SNE sẽ cố gắng giữ hai cụm này tách biệt khi giảm chiều.
    """)

    st.markdown("### 🔹 **Các bước thực hiện t-SNE**")
    st.markdown("""
    1. **Tính xác suất tương đồng trong không gian gốc**  
       - Với mỗi cặp điểm $x_i$ và $x_j$, tính xác suất $p_{j|i}$ rằng $x_j$ là hàng xóm của $x_i$:  
         $$ p_{j|i} = \\frac{\\exp(-\\| x_i - x_j \\|^2 / 2\\sigma^2)}{\\sum_{k \\neq i} \\exp(-\\| x_i - x_k \\|^2 / 2\\sigma^2)} $$  
       - $\\sigma$: Độ rộng của phân phối Gaussian, phụ thuộc vào số lượng hàng xóm được xem xét.  
       - **Ý nghĩa**: Các điểm gần nhau có xác suất lớn hơn.

    2. **Tính xác suất trong không gian mới**  
       - Trong không gian giảm chiều, dùng phân phối t-Student để tính $q_{j|i}$:  
         $$ q_{j|i} = \\frac{(1 + \\| y_i - y_j \\|^2)^{-1}}{\\sum_{k \\neq i} (1 + \\| y_i - y_k \\|^2)^{-1}} $$  
       - **Ý nghĩa**: Phân phối t-Student có đuôi dài, giúp phân bố các điểm xa nhau hợp lý hơn.

    3. **Tối ưu hóa sự khác biệt**  
       - Đo sự khác biệt giữa $P$ và $Q$ bằng **KL-divergence**:  
         $$ KL(P||Q) = \\sum_{i \\neq j} p_{ij} \\log \\frac{p_{ij}}{q_{ij}} $$  
       - Dùng gradient descent để điều chỉnh tọa độ $y_i$ sao cho $KL$ nhỏ nhất.
    """)

    st.markdown("### ✅ **Ưu điểm của t-SNE**")
    st.markdown("""
    - Tạo các cụm dữ liệu rõ ràng, dễ nhìn trong không gian 2D/3D.
    - Phù hợp với dữ liệu phi tuyến tính (PCA không làm được).
    - Rất tốt để trực quan hóa dữ liệu phức tạp như MNIST.
    """)

    st.markdown("### ❌ **Nhược điểm của t-SNE**")
    st.markdown("""
    - Tốn nhiều thời gian tính toán, đặc biệt với dữ liệu lớn.
    - Nhạy cảm với cách thiết lập ban đầu (cần chọn cẩn thận).
    - Không bảo toàn cấu trúc toàn cục, chỉ tập trung vào cục bộ.
    - Không phù hợp để giảm chiều cho học máy (chỉ dùng để trực quan hóa).
    """)

# Hàm thực hiện giảm chiều và trực quan hóa
def dimensionality_reduction():
    st.title("📉 Giảm Chiều Dữ liệu MNIST")

    if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
        st.warning("⚠ Vui lòng tải dữ liệu từ tab 'Dữ Liệu' trước khi tiếp tục!")
        return

    X, y = st.session_state["X"], st.session_state["y"]
    st.write(f"Tổng số mẫu: {X.shape[0]}, Số chiều ban đầu: {X.shape[1]}")

    num_samples = st.slider("Chọn số lượng mẫu:", 1000, X.shape[0], 5000, step=1000)
    X_subset, y_subset = X[:num_samples], y[:num_samples]

    method = st.radio("Chọn phương pháp giảm chiều:", ["PCA", "t-SNE"])
    n_components = st.slider("Số chiều giảm xuống:", 1, 3, 2)

    run_name = st.text_input("🔹 Nhập tên Run:", "")  # Để trống để người dùng tự nhập

    if st.button("🚀 Chạy Giảm Chiều"):
        if not run_name:
            st.error("⚠ Vui lòng nhập tên Run trước khi tiếp tục!")
            return

        with st.spinner(f"Đang thực hiện {method}..."):
            # Khởi tạo thanh trạng thái
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Cập nhật trạng thái: Bắt đầu
            status_text.text("Bắt đầu quá trình giảm chiều...")
            progress_bar.progress(0.1)

            input_mlflow()
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("method", method)
                mlflow.log_param("n_components", n_components)
                mlflow.log_param("num_samples", num_samples)
                mlflow.log_param("original_dim", X.shape[1])

                start_time = time.time()

                if method == "PCA":
                    # Giai đoạn 1: Khởi tạo PCA
                    status_text.text("Khởi tạo PCA...")
                    reducer = PCA(n_components=n_components)
                    progress_bar.progress(0.3)

                    # Giai đoạn 2: Fit và transform dữ liệu
                    status_text.text("Đang giảm chiều dữ liệu với PCA...")
                    X_reduced = reducer.fit_transform(X_subset)
                    progress_bar.progress(0.7)
                    #explained_variance_ratio là Phương sai 
                    if n_components > 1:
                        explained_variance = np.sum(reducer.explained_variance_ratio_)
                        mlflow.log_metric("explained_variance_ratio", explained_variance)
                else:
                    # Giai đoạn 1: Khởi tạo t-SNE
                    status_text.text("Khởi tạo t-SNE...")
                    perplexity = min(30, num_samples - 1)
                    mlflow.log_param("perplexity", perplexity)
                    reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
                    progress_bar.progress(0.3)

                    # Giai đoạn 2: Fit và transform dữ liệu
                    status_text.text("Đang giảm chiều dữ liệu với t-SNE (có thể lâu hơn PCA)...")
                    X_reduced = reducer.fit_transform(X_subset)
                    progress_bar.progress(0.7)
                    #kl_divergence là một thước đo sự khác biệt giữa hai phân phối xác suất, được sử dụng trong t-SNE để tối ưu hóa việc giảm chiều
                    #là một thước đo sự khác biệt giữa hai phân phối xác suất, được sử dụng trong t-SNE để tối ưu hóa việc giảm chiều
                    if hasattr(reducer, "kl_divergence_"):
                        mlflow.log_metric("KL_divergence", reducer.kl_divergence_)

                # Giai đoạn 3: Trực quan hóa
                status_text.text("Đang tạo biểu đồ trực quan...")
                elapsed_time = time.time() - start_time
                mlflow.log_metric("elapsed_time", elapsed_time)

                if n_components == 1:
                    fig = px.line(x=range(len(X_reduced)), y=X_reduced.flatten(), color=y_subset,
                                  title=f"{method} giảm chiều xuống 1D",
                                  labels={'x': "Mẫu", 'y': "Giá trị thành phần"})
                elif n_components == 2:
                    fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], color=y_subset,
                                     title=f"{method} giảm chiều xuống 2D",
                                     labels={'x': "Thành phần 1", 'y': "Thành phần 2"})
                else:
                    fig = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                                        color=y_subset,
                                        title=f"{method} giảm chiều xuống 3D",
                                        labels={'x': "Thành phần 1", 'y': "Thành phần 2", 'z': "Thành phần 3"})
                st.plotly_chart(fig)
                progress_bar.progress(0.9)

                # Giai đoạn 4: Lưu dữ liệu và hoàn tất
                status_text.text("Đang lưu dữ liệu và hoàn tất...")
                os.makedirs("logs", exist_ok=True)
                reduced_data_path = f"logs/{method}_{n_components}D_X_reduced.npy"
                np.save(reduced_data_path, X_reduced)
                mlflow.log_artifact(reduced_data_path)

                # Lưu run_name vào session_state để dùng trong tab MLflow
                st.session_state["last_run_name"] = run_name

                progress_bar.progress(1.0)
                status_text.text("Hoàn thành!")

                st.success(f"✅ Hoàn thành {method} trong {elapsed_time:.2f} giây!")
                st.markdown(f"🔗 [Xem kết quả trên MLflow]({st.session_state['mlflow_url']})")

# Hàm hiển thị thông tin MLflow Experiments (Hiển thị tất cả run_name và chi tiết run vừa chạy)
def show_experiment_selector():
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'> MLflow Experiments </h1>", unsafe_allow_html=True)
    if 'mlflow_url' in st.session_state:
        st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
    else:
        st.warning("⚠️ URL MLflow chưa được khởi tạo!")

    input_mlflow()
    experiment_name = "MNIST_PCA_t_SNE"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Không tìm thấy Experiment '{experiment_name}'!", icon="🚫")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'🟢 Active' if selected_experiment.lifecycle_stage == 'active' else '🔴 Deleted'}")
    st.write(f"**Artifact Location:** `{selected_experiment.artifact_location}`")

    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này!", icon="🚨")
        return

    # Hiển thị tất cả run_name dưới dạng danh sách chọn
    st.subheader("🏃‍♂️ Tất cả Runs trong Experiment")
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_data = mlflow.get_run(run_id)
        run_name = run_data.info.run_name if run_data.info.run_name else f"Run_{run_id[:8]}"
        run_info.append((run_name, run_id))

    # Loại bỏ trùng lặp trong danh sách run_name để hiển thị trong selectbox
    run_name_to_id = dict(run_info)  # Từ điển ánh xạ run_name -> run_id (giữ run_id cuối cùng nếu trùng)
    run_names = list(run_name_to_id.keys())  # Danh sách run_name không trùng lặp

    st.write("**Chọn Run để xem chi tiết:**")
    selected_run_name = st.selectbox("Danh sách Run Names:", run_names, key="run_selector")

    # Tìm run tương ứng với run_name được chọn
    selected_run_id = run_name_to_id[selected_run_name]
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.markdown(f"<h3 style='color: #28B463;'>📌 Chi tiết Run Được Chọn: {selected_run_name}</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("#### ℹ️ Thông tin cơ bản")
            st.info(f"**Run Name:** {selected_run_name}")
            st.info(f"**Run ID:** `{selected_run_id}`")
            st.info(f"**Trạng thái:** {selected_run.info.status}")
            start_time_ms = selected_run.info.start_time
            if start_time_ms:
                start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
            else:
                start_time = "Không có thông tin"
            st.info(f"**Thời gian chạy:** {start_time}")

        with col2:
            params = selected_run.data.params
            if params:
                st.write("#### ⚙️ Parameters")
                with st.container(height=200):
                    st.json(params)

            metrics = selected_run.data.metrics
            if metrics:
                st.write("#### 📊 Metrics")
                with st.container(height=200):
                    st.json(metrics)

    # Hiển thị chi tiết run vừa chạy từ tab Giảm Chiều (nếu có)
    st.markdown("---")
    if "last_run_name" not in st.session_state or not st.session_state["last_run_name"]:
        st.warning("⚠ Chưa có run nào được thực hiện gần đây. Vui lòng chạy giảm chiều trong tab 'Giảm Chiều' để xem chi tiết!")
    else:
        last_run_name = st.session_state["last_run_name"]
        st.subheader(f"📌 Chi tiết Run Gần Đây: {last_run_name}")

        # Tìm run với run_name vừa chạy
        selected_last_run = None
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            run_data = mlflow.get_run(run_id)
            if run_data.info.run_name == last_run_name:
                selected_last_run = run_data
                selected_last_run_id = run_id
                break

        if selected_last_run:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("#### ℹ️ Thông tin cơ bản")
                st.info(f"**Run Name:** {last_run_name}")
                st.info(f"**Run ID:** `{selected_last_run_id}`")
                st.info(f"**Trạng thái:** {selected_last_run.info.status}")
                start_time_ms = selected_last_run.info.start_time
                if start_time_ms:
                    start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    start_time = "Không có thông tin"
                st.info(f"**Thời gian chạy:** {start_time}")

            with col2:
                params = selected_last_run.data.params
                if params:
                    st.write("#### ⚙️ Parameters")
                    with st.container(height=200):
                        st.json(params)

                metrics = selected_last_run.data.metrics
                if metrics:
                    st.write("#### 📊 Metrics")
                    with st.container(height=200):
                        st.json(metrics)
        else:
            st.warning(f"⚠ Không tìm thấy run với tên '{last_run_name}'. Vui lòng kiểm tra lại hoặc chạy lại trong tab 'Giảm Chiều'!")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Powered by Streamlit & MLflow</p>", unsafe_allow_html=True)

# Giao diện chính
def main():
    st.title("🚀 MNIST Dimensionality Reduction with PCA & t-SNE")
    tabs = st.tabs(["📘 Dữ Liệu", "📘 PCA", "📘 t-SNE", "📉 Giảm Chiều", "📊 MLflow"])

    with tabs[0]:
        data()
    with tabs[1]:
        explain_pca()
    with tabs[2]:
        explain_tsne()
    with tabs[3]:
        dimensionality_reduction()
    with tabs[4]:
        show_experiment_selector()

if __name__ == "__main__":
    main()