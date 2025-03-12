import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import mlflow
from mlflow.tracking import MlflowClient
import random
from datetime import datetime

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/TonThatTruongVu/MNIST-ClusteringAlgorithms.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "TonThatTruongVu"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "519c4a864e131de52197f54d170c130beb15ffd5"
    mlflow.set_experiment("MNIST_Clustering")
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI

# Tải dữ liệu MNIST từ OpenML
@st.cache_data
def load_mnist_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32) / 255.0  # Chuẩn hóa và chuyển sang float32 ngay từ đầu
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

# Tab lý thuyết K-means
def ly_thuyet_K_means():
    st.header("📌 Lý thuyết K-Means")
    st.markdown("""
    - **K-Means** là một thuật toán phân cụm **không giám sát** (unsupervised learning) nhằm chia dữ liệu thành **K cụm** (clusters) dựa trên sự tương đồng giữa các điểm dữ liệu. Thuật toán sử dụng **khoảng cách Euclidean** để đo lường sự gần gũi giữa các điểm và tâm cụm (centroids).
    """)

    st.subheader("🔍 Cách hoạt động chi tiết")
    st.markdown("""
    Thuật toán K-Means hoạt động qua các bước lặp đi lặp lại như sau:
    """)


    with st.expander("1. Bước 1: Khởi tạo tâm cụm (Initialization)"):
        st.markdown("""
        - Chọn ngẫu nhiên **K điểm** từ tập dữ liệu làm **tâm cụm ban đầu** (centroids).  
        - **Ví dụ**: Với \( K = 3 \), chọn 3 điểm ngẫu nhiên từ tập MNIST làm các tâm cụm khởi đầu.
        """)

    with st.expander("2. Bước 2: Gán nhãn cụm (Assignment Step)"):
        st.markdown("""
        - Với mỗi điểm dữ liệu trong tập, tính **khoảng cách Euclidean** đến tất cả các tâm cụm.  
        - Gán điểm đó vào cụm có tâm gần nhất.  
        - **Công thức khoảng cách Euclidean**:  
        """)
        st.latex(r"d(x, c) = \sqrt{\sum_{i=1}^{n} (x_i - c_i)^2}")
        st.markdown("""
        Trong đó:  
        - \( x \): Điểm dữ liệu.  
        - \( c \): Tâm cụm.  
        - \( n \): Số chiều của dữ liệu (với MNIST là 784).
        """)

    with st.expander("3. Bước 3: Cập nhật tâm cụm (Update Step)"):
        st.markdown("""
        - Sau khi gán tất cả điểm vào các cụm, tính lại **tâm cụm mới** bằng cách lấy **trung bình tọa độ** của mọi điểm trong cụm đó.  
        - **Công thức**:  
        """)
        st.latex(r"c_j = \frac{1}{N_j} \sum_{x \in C_j} x")
        st.markdown("""
        Trong đó:  
        - $c_j$: Tâm cụm thứ $j$  
        - $N_j$: Số điểm trong cụm $j$  
        - $C_j$: Tập hợp các điểm thuộc cụm $j$  
        """)

    with st.expander("4. Bước 4: Lặp lại (Iteration)"):
        st.markdown("""
        - Quay lại bước 2, lặp lại quá trình gán nhãn và cập nhật tâm cụm cho đến khi:  
          - Các tâm cụm không còn thay đổi đáng kể (hội tụ).  
          - Hoặc đạt số lần lặp tối đa (max iterations).
        """)

    st.subheader("💡 Ví dụ với MNIST")
    st.markdown("""
    - Nếu \( K = 10 \) (số chữ số từ 0-9), K-Means sẽ cố gắng nhóm các ảnh chữ số thành 10 cụm.  
    - Ban đầu, chọn 10 ảnh ngẫu nhiên làm tâm. Sau vài lần lặp, các tâm cụm dần đại diện cho các nhóm chữ số (ví dụ: cụm 0 chứa hầu hết ảnh số 0).
    """)

# Tab lý thuyết DBSCAN
def ly_thuyet_DBSCAN():
    st.header("📌 Lý thuyết DBSCAN")
    st.write("""
    - **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) là một thuật toán phân cụm **không giám sát** dựa trên **mật độ** của các điểm dữ liệu. 
    - Khác với K-Means, DBSCAN không yêu cầu xác định trước số cụm, mà tự động tìm các cụm dựa trên phân bố dữ liệu và có khả năng phát hiện **nhiễu** (noise).
    """)

    st.subheader("🔍 Cách hoạt động chi tiết")
    st.markdown("""
    DBSCAN phân cụm dựaLIM trên hai tham số chính:  
    - **eps**: Bán kính lân cận (khoảng cách tối đa giữa hai điểm để coi là "gần nhau").  
    - **min_samples**: Số điểm tối thiểu trong vùng lân cận để hình thành một cụm.  
    Các bước cụ thể:
    """)


    with st.expander("1. Bước 1: Xác định các loại điểm (Point Classification)"):
        st.markdown("""
        - **Core Point (Điểm lõi)**: Một điểm có ít nhất **min_samples** láng giềng (bao gồm chính nó) trong bán kính **eps**.  
        - **Border Point (Điểm ranh giới)**: Không phải điểm lõi, nhưng nằm trong bán kính **eps** của ít nhất một điểm lõi.  
        - **Noise Point (Điểm nhiễu)**: Không phải điểm lõi, không nằm trong bán kính **eps** của bất kỳ điểm lõi nào.  
        - **Ví dụ**: Với MNIST, một điểm lõi có thể là trung tâm của vùng chữ số "0", các điểm ranh giới là viền, và nhiễu là các nét lỗi.
        """)

    with st.expander("2. Bước 2: Khởi tạo cụm (Cluster Initialization)"):
        st.markdown("""
        - Chọn một **điểm lõi chưa thăm** (unvisited core point) làm hạt giống (seed).  
        - Tạo cụm mới từ điểm này để bắt đầu quá trình phân cụm.
        """)

    with st.expander("3. Bước 3: Mở rộng cụm (Cluster Expansion)"):
        st.markdown("""
        - Thêm tất cả các điểm trong bán kính **eps** của điểm lõi vào cụm.  
        - Nếu một điểm được thêm là điểm lõi, tiếp tục mở rộng cụm từ điểm đó (đệ quy).  
        - **Công thức khoảng cách Euclidean**:  
        """)
        st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}")
        st.markdown("""
        Trong đó:  
        - \( x, y \): Hai điểm dữ liệu.  
        - \( n \): Số chiều (784 với MNIST).
        """)

    with st.expander("4. Bước 4: Đánh dấu nhiễu và lặp lại"):
        st.markdown("""
        - Các điểm không thuộc bất kỳ cụm nào được đánh dấu là **nhiễu**.  
        - Chọn điểm lõi chưa thăm tiếp theo, lặp lại quá trình cho đến khi tất cả điểm được xử lý.
        """)

    st.subheader("💡 Ví dụ với MNIST")
    st.markdown("""
    - Nếu **eps = 0.5** và **min_samples = 5**, DBSCAN có thể:  
      - Tìm các cụm dày đặc (như vùng chữ số giống nhau, ví dụ: các ảnh "1" thẳng đứng).  
      - Loại bỏ các nét vẽ bất thường hoặc các ảnh khác biệt lớn (như "1" nghiêng quá xa) làm nhiễu.  
    - Kết quả: Số cụm không cố định, phụ thuộc vào mật độ dữ liệu.
    """)

# Tab phân cụm
def clustering():
    st.header("⚙️ Phân cụm dữ liệu MNIST")
    
    if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
        st.warning("⚠️ Vui lòng tải dữ liệu từ tab 'Data' trước khi thực hiện phân cụm!")
        return

    X, y = st.session_state.X, st.session_state.y
    total_samples = X.shape[0]

    # Phần chia dữ liệu
    st.subheader("📌 Chia dữ liệu")
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False

    num_samples = st.slider("📌 Chọn số lượng ảnh để huấn luyện:", 1000, total_samples, 5000)
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần còn lại)", 0, 50, 15)
    train_size = remaining_size - val_size
    st.write(f"📌 **Tỷ lệ phân chia:** Train={train_size}%, Validation={val_size}%, Test={test_size}%")

    if st.button("✅ Xác nhận & Chia dữ liệu") and not st.session_state.data_split_done:
        st.session_state.data_split_done = True
        
        # Xử lý trường hợp chọn toàn bộ dữ liệu
        if num_samples == total_samples:
            X_selected = X
            y_selected = y
        else:
            X_selected, _, y_selected, _ = train_test_split(
                X, y, train_size=num_samples/total_samples, stratify=y, random_state=42
            )

        X_temp, X_test, y_temp, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size/100, stratify=y_selected, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(100 - test_size), stratify=y_temp, random_state=42
        )

        st.session_state.X_train = X_train
        st.session_state.X_val = X_val
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_val = y_val
        st.session_state.y_test = y_test
        st.session_state.train_size = X_train.shape[0]
        st.session_state.val_size = X_val.shape[0]
        st.session_state.test_size = X_test.shape[0]

        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, không cần chạy lại.")

    if not st.session_state.data_split_done:
        return

    X_train = st.session_state.X_train

    clustering_method = st.selectbox("Chọn kỹ thuật phân cụm:", ["K-means", "DBSCAN"])
    
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")
    st.session_state["run_name"] = run_name if run_name else "Default_Run"

    if "clustering_results" not in st.session_state:
        st.session_state.clustering_results = None
    if "models" not in st.session_state:
        st.session_state.models = []

    # Thêm cảnh báo khi chọn full data
    if num_samples == total_samples:
        st.warning("⚠️ Bạn đang chọn toàn bộ dữ liệu (70,000 mẫu). Điều này có thể gây lỗi do bộ nhớ hoặc thời gian tính toán quá lâu!")

    if clustering_method == "K-means":
        st.markdown("""
        - **K-means**: Phân cụm dựa trên số lượng cụm (clusters) được chỉ định.
        - **Tham số:**
          - **n_clusters**: Số lượng cụm mong muốn.
        """)
        n_clusters = st.slider("Số lượng cụm (n_clusters):", 2, 20, 10)

        if st.button("Phân cụm với K-means"):
            with mlflow.start_run(run_name=f"Kmeans_{st.session_state['run_name']}"):
                st.write("⏳ Đang chạy K-means...")
                progress_bar = st.progress(0)

                try:
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    
                    # Bổ sung thanh trạng thái
                    progress_bar.progress(0.1)  # Bắt đầu
                    model.fit(X_train)
                    progress_bar.progress(0.7)  # Sau khi fit
                    labels = model.labels_
                    silhouette_avg = silhouette_score(X_train, labels)
                    progress_bar.progress(1.0)  # Hoàn thành

                    mlflow.log_param("method", "K-means")
                    mlflow.log_param("n_clusters", n_clusters)
                    mlflow.log_param("num_samples", X_train.shape[0])
                    mlflow.log_metric("silhouette_score", silhouette_avg)
                    mlflow.sklearn.log_model(model, "kmeans_model")

                    st.session_state.clustering_results = {
                        "method": "K-means",
                        "labels": labels,
                        "silhouette_score": silhouette_avg,
                        "run_name": f"Kmeans_{st.session_state['run_name']}",
                        "status": "success"
                    }
                    st.session_state.models.append({
                        "name": "kmeans",
                        "run_name": f"Kmeans_{st.session_state['run_name']}",
                        "model": model
                    })

                except Exception as e:
                    error_message = str(e)
                    st.error(f"❌ Lỗi khi chạy K-means: {error_message}")
                    if "memory" in error_message.lower():
                        st.error("⚠️ Lỗi này có thể do chọn toàn bộ dữ liệu (70,000 mẫu). Hãy giảm số lượng mẫu!")
                    progress_bar.progress(0)  # Reset thanh trạng thái
                    mlflow.log_param("method", "K-means")
                    mlflow.log_param("n_clusters", n_clusters)
                    mlflow.log_param("num_samples", X_train.shape[0])
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error_message", error_message)
                    st.session_state.clustering_results = {
                        "method": "K-means",
                        "error_message": error_message,
                        "run_name": f"Kmeans_{st.session_state['run_name']}",
                        "status": "failed"
                    }

    elif clustering_method == "DBSCAN":
        st.markdown("""
        - **DBSCAN**: Phân cụm dựa trên mật độ, không cần chỉ định số cụm trước.
        - **Tham số:**
          - **eps**: Khoảng cách tối đa giữa hai điểm để coi là cùng cụm.
          - **min_samples**: Số lượng điểm tối thiểu để tạo thành một cụm.
        """)
        eps = st.slider("eps (khoảng cách tối đa):", 0.1, 10.0, 1.0)
        min_samples = st.slider("min_samples (số mẫu tối thiểu):", 1, 20, 5)

        if st.button("Phân cụm với DBSCAN"):
            with mlflow.start_run(run_name=f"DBSCAN_{st.session_state['run_name']}"):
                st.write("⏳ Đang chạy DBSCAN...")
                progress_bar = st.progress(0)

                try:
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    
                    # Bổ sung thanh trạng thái
                    progress_bar.progress(0.1)  # Bắt đầu
                    model.fit(X_train)
                    progress_bar.progress(0.7)  # Sau khi fit
                    labels = model.labels_
                    if len(np.unique(labels)) > 1 and not np.all(labels == -1):
                        silhouette_avg = silhouette_score(X_train, labels)
                    else:
                        silhouette_avg = None
                    progress_bar.progress(1.0)  # Hoàn thành

                    mlflow.log_param("method", "DBSCAN")
                    mlflow.log_param("eps", eps)
                    mlflow.log_param("min_samples", min_samples)
                    mlflow.log_param("num_samples", X_train.shape[0])
                    if silhouette_avg is not None:
                        mlflow.log_metric("silhouette_score", silhouette_avg)
                    mlflow.sklearn.log_model(model, "dbscan_model")

                    st.session_state.clustering_results = {
                        "method": "DBSCAN",
                        "labels": labels,
                        "silhouette_score": silhouette_avg,
                        "run_name": f"DBSCAN_{st.session_state['run_name']}",
                        "status": "success"
                    }
                    st.session_state.models.append({
                        "name": "dbscan",
                        "run_name": f"DBSCAN_{st.session_state['run_name']}",
                        "model": model,
                        "X_train": X_train,
                        "eps": eps,
                        "min_samples": min_samples,
                        "labels": labels
                    })

                except Exception as e:
                    error_message = str(e)
                    st.error(f"❌ Lỗi khi chạy DBSCAN: {error_message}")
                    if "memory" in error_message.lower():
                        st.error("⚠️ Lỗi này có thể do chọn toàn bộ dữ liệu (70,000 mẫu). Hãy giảm số lượng mẫu!")
                    progress_bar.progress(0)  # Reset thanh trạng thái
                    mlflow.log_param("method", "DBSCAN")
                    mlflow.log_param("eps", eps)
                    mlflow.log_param("min_samples", min_samples)
                    mlflow.log_param("num_samples", X_train.shape[0])
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error_message", error_message)
                    st.session_state.clustering_results = {
                        "method": "DBSCAN",
                        "error_message": error_message,
                        "run_name": f"DBSCAN_{st.session_state['run_name']}",
                        "status": "failed"
                    }

    # Hiển thị kết quả
    if st.session_state.clustering_results:
        results = st.session_state.clustering_results
        if results["status"] == "success":
            st.success(f"✅ Kết quả phân cụm với {results['method']}:")
            st.write(f"Số lượng cụm: {len(np.unique(results['labels']))}")
            if results['silhouette_score'] is not None:
                st.write(f"Silhouette Score: {results['silhouette_score']:.4f}")
            else:
                st.write("Silhouette Score: Không tính được (quá ít cụm hoặc tất cả là nhiễu)")
            st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")

            st.subheader("Hình ảnh mẫu từ các cụm")
            unique_labels = np.unique(results['labels'])
            max_clusters_to_display = st.slider("Số lượng cụm muốn hiển thị:", 1, len(unique_labels), min(5, len(unique_labels)))
            for label in unique_labels[:max_clusters_to_display]:
                if label != -1:
                    st.write(f"Cụm {label}:")
                    cluster_samples = X_train[results['labels'] == label][:5]
                    fig, axes = plt.subplots(1, min(5, len(cluster_samples)), figsize=(10, 2))
                    if len(cluster_samples) == 1:
                        axes = [axes]
                    for ax, sample in zip(axes, cluster_samples):
                        ax.imshow(sample.reshape(28, 28), cmap='gray')
                        ax.axis("off")
                    st.pyplot(fig)
        else:
            st.error(f"❌ Phân cụm thất bại: {results['error_message']}")

        st.write("📊 Hiển thị thông tin MLflow Experiments:")
        show_experiment_selector(context="predict")

    # Hiển thị kết quả

# Tab dự đoán
def predict():
    st.header("✍️ Dự đoán cụm")
    
    if "models" not in st.session_state or not st.session_state.models:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng huấn luyện mô hình trong tab 'Huấn Luyện' trước!")
        return

    # Hiển thị danh sách mô hình đã huấn luyện
    model_display = [f"{model['run_name']} ({model['name']})" for model in st.session_state.models]
    model_option = st.selectbox("🔍 Chọn mô hình:", model_display)
    selected_model_info = next(model for model in st.session_state.models if f"{model['run_name']} ({model['name']})" == model_option)
    model = selected_model_info["model"]
    model_name = selected_model_info["name"]
    st.success(f"✅ Đã chọn mô hình: {model_option}")

    # Hiển thị kết quả phân cụm của mô hình đã chọn
    st.subheader("📊 Kết quả phân cụm của mô hình đã chọn")
    if model_name == "kmeans":
        labels = model.labels_
        X_train = st.session_state.X_train  # Dữ liệu train từ phiên hiện tại
        silhouette_avg = silhouette_score(X_train, labels)
        st.write(f"**Phương pháp:** K-means")
        st.write(f"**Số lượng cụm:** {len(np.unique(labels))}")
        st.write(f"**Silhouette Score:** {silhouette_avg:.4f}")

        # Hiển thị hình ảnh mẫu từ các cụm
        st.subheader("Hình ảnh mẫu từ các cụm")
        unique_labels = np.unique(labels)
        max_clusters_to_display = st.slider("Số lượng cụm muốn hiển thị:", 1, len(unique_labels), min(5, len(unique_labels)), key="kmeans_clusters_display")
        for label in unique_labels[:max_clusters_to_display]:
            st.write(f"Cụm {label}:")
            cluster_samples = X_train[labels == label][:5]
            fig, axes = plt.subplots(1, min(5, len(cluster_samples)), figsize=(10, 2))
            if len(cluster_samples) == 1:
                axes = [axes]
            for ax, sample in zip(axes, cluster_samples):
                ax.imshow(sample.reshape(28, 28), cmap='gray')
                ax.axis("off")
            st.pyplot(fig)

    elif model_name == "dbscan":
        labels = selected_model_info["labels"]
        X_train = selected_model_info["X_train"]
        if len(np.unique(labels)) > 1 and not np.all(labels == -1):
            silhouette_avg = silhouette_score(X_train, labels)
        else:
            silhouette_avg = None
        st.write(f"**Phương pháp:** DBSCAN")
        st.write(f"**Số lượng cụm:** {len(np.unique(labels)) - (1 if -1 in labels else 0)} (không tính nhiễu)")
        st.write(f"**Số điểm nhiễu:** {np.sum(labels == -1)}")
        if silhouette_avg is not None:
            st.write(f"**Silhouette Score:** {silhouette_avg:.4f}")
        else:
            st.write("**Silhouette Score:** Không tính được (quá ít cụm hoặc tất cả là nhiễu)")

        # Hiển thị hình ảnh mẫu từ các cụm
        st.subheader("Hình ảnh mẫu từ các cụm")
        unique_labels = np.unique(labels)
        max_clusters_to_display = st.slider("Số lượng cụm muốn hiển thị:", 1, len(unique_labels), min(5, len(unique_labels)), key="dbscan_clusters_display")
        for label in unique_labels[:max_clusters_to_display]:
            if label != -1:  # Không hiển thị nhiễu
                st.write(f"Cụm {label}:")
                cluster_samples = X_train[labels == label][:5]
                fig, axes = plt.subplots(1, min(5, len(cluster_samples)), figsize=(10, 2))
                if len(cluster_samples) == 1:
                    axes = [axes]
                for ax, sample in zip(axes, cluster_samples):
                    ax.imshow(sample.reshape(28, 28), cmap='gray')
                    ax.axis("off")
                st.pyplot(fig)
            else:
                st.write("Nhiễu (-1):")
                noise_samples = X_train[labels == -1][:5]
                fig, axes = plt.subplots(1, min(5, len(noise_samples)), figsize=(10, 2))
                if len(noise_samples) == 1:
                    axes = [axes]
                for ax, sample in zip(axes, noise_samples):
                    ax.imshow(sample.reshape(28, 28), cmap='gray')
                    ax.axis("off")
                st.pyplot(fig)
    


# Tab MLflow
def show_experiment_selector(context="mlflow"):
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'> MLflow Experiments </h1>", unsafe_allow_html=True)
    if 'mlflow_url' in st.session_state:
        st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
    else:
        st.warning("⚠️ URL MLflow chưa được khởi tạo!")

    with st.sidebar:
        st.subheader("🔍 Tổng quan Experiment")
        experiment_name = "MNIST_Clustering"
        
        experiments = mlflow.search_experiments()
        selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

        if not selected_experiment:
            st.error(f"❌ Không tìm thấy Experiment '{experiment_name}'!", icon="🚫")
            return

        st.markdown(f"**Tên Experiment:** `{experiment_name}`")
        st.markdown(f"**ID:** `{selected_experiment.experiment_id}`")
        st.markdown(f"**Trạng thái:** {'🟢 Active' if selected_experiment.lifecycle_stage == 'active' else '🔴 Deleted'}")
        st.markdown(f"**Artifact Location:** `{selected_experiment.artifact_location}`")

    st.markdown("---")
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này!", icon="🚨")
        return

    with st.expander("🏃‍♂️ Danh sách Runs", expanded=True):
        st.write("Chọn một Run để xem chi tiết:")
        run_info = []
        used_names = set()

        for _, run in runs.iterrows():
            run_id = run["run_id"]
            run_data = mlflow.get_run(run_id)
            run_name = run_data.info.run_name if run_data.info.run_name else f"Run_{run_id[:8]}"
            display_name = run_name
            
            run_name_base = display_name
            counter = 1
            while display_name in used_names:
                display_name = f"{run_name_base}_{counter}"
                counter += 1
            used_names.add(display_name)
            run_info.append((display_name, run_id))

        run_name_to_id = dict(run_info)
        run_names = list(run_name_to_id.keys())

        selectbox_key = f"run_selector_{context}"
        selected_run_name = st.selectbox("🔍 Chọn Run:", run_names, key=selectbox_key, help="Chọn để xem thông tin chi tiết")

    selected_run_id = run_name_to_id[selected_run_name]
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.markdown(f"<h3 style='color: #28B463;'>📌 Chi tiết Run: {selected_run_name}</h3>", unsafe_allow_html=True)

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

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Powered by Streamlit & MLflow</p>", unsafe_allow_html=True)

# Hàm chính
def main():
   
    if "mlflow_initialized" not in st.session_state:
        mlflow_input()
        st.session_state.mlflow_initialized = True
        
    st.title("🖍️ MNIST Clustering App (OpenML)")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📘 Data", "📚 K-means", "📚 DBSCAN", "⚙️ Huấn Luyện", "✍️ Demo"])
    
    with tab1:
        data()
        
    with tab2:
        ly_thuyet_K_means()
        
    with tab3:
        ly_thuyet_DBSCAN()
        
    with tab4:
        clustering()
        
    with tab5:
        predict()

if __name__ == "__main__":
    main()