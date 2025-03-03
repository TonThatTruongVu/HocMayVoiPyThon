import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import mlflow
import os
import random
from datetime import datetime

# Tải dữ liệu MNIST từ OpenML
@st.cache_data
def load_mnist_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0  # Chuẩn hóa ngay khi tải
    return X, y.astype(int)

# Tab hiển thị dữ liệu
def data_processing():
    st.header("📘 Dữ Liệu MNIST")
    X, y = load_mnist_data()
    
    st.write("""
        **Thông tin tập dữ liệu MNIST:**
        - Tổng số mẫu: {}
        - Kích thước mỗi ảnh: 28 × 28 pixels (784 đặc trưng)
        - Số lớp: 10 (chữ số từ 0-9)
    """.format(X.shape[0]))

    st.subheader("Một số hình ảnh mẫu")
    n_samples = 5
    fig, axes = plt.subplots(1, n_samples, figsize=(10, 3))
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    for i, idx in enumerate(indices):
        axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Label: {y[idx]}")
        axes[i].axis("off")
    st.pyplot(fig)

# Tab chia dữ liệu
def split_data():
    st.title("📌 Chia dữ liệu Train/Test")
    X, y = load_mnist_data()
    total_samples = X.shape[0]

    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False

    num_samples = st.slider("Chọn số lượng ảnh để train:", 1000, total_samples, 10000)
    test_size_percent = st.slider("Chọn tỷ lệ test (%):", 10, 50, 20)  # Đổi sang phần trăm
    test_size = test_size_percent / 100  # Chuyển đổi sang dạng thập phân để sử dụng trong train_test_split

    if st.button("✅ Xác nhận & Lưu") and not st.session_state.data_split_done:
        st.session_state.data_split_done = True
        X_selected, _, y_selected, _ = train_test_split(X, y, train_size=num_samples, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=test_size, random_state=42)

        st.session_state.total_samples = num_samples
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.train_size = X_train.shape[0]
        st.session_state.test_size = X_test.shape[0]

        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_test.shape[0]]
        })
        st.success(f"🔹 Dữ liệu đã được chia: Train ({len(X_train)}), Test ({len(X_test)})")
        st.table(summary_df)

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, không cần chạy lại.")
# Tab huấn luyện và phân cụm

import streamlit as st
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

def train_evaluate():
    st.header("⚙️ Chọn mô hình & Huấn luyện")
    
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    model_choice = st.selectbox("Chọn mô hình:", ["K-Means", "DBSCAN"])
    
    if model_choice == "K-Means":
        st.markdown("""
        - **K-Means**: Thuật toán phân cụm chia dữ liệu thành K nhóm dựa trên khoảng cách Euclidean.
        - **Tham số cần chọn:**  
            - **Số cụm (K)**: Số nhóm mong muốn.
        """)
        n_clusters = st.slider("Số cụm (K):", 2, 20, 10)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        params = {"n_clusters": n_clusters}
    else:  # DBSCAN
        st.markdown("""
        - **DBSCAN**: Thuật toán phân cụm dựa trên mật độ, không cần xác định số cụm trước.
        - **Tham số cần chọn:**
            - **eps**: Bán kính lân cận để xác định điểm láng giềng.
            - **min_samples**: Số điểm tối thiểu để tạo thành cụm.
        """)
        eps = st.slider("Bán kính lân cận (eps):", 0.1, 10.0, 0.5, step=0.1)
        min_samples = st.slider("Số điểm tối thiểu:", 2, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        params = {"eps": eps, "min_samples": min_samples}

    run_name = st.text_input("🔹 Nhập tên Run:", f"{model_choice}_Run")
    st.session_state["run_name"] = run_name if run_name else "default_run"
    
    if st.button("🚀 Huấn luyện mô hình"):
        st.write(f"⏳ Đang huấn luyện mô hình '{model_choice}'...")
        with st.spinner("Đang xử lý dữ liệu và huấn luyện..."):
            # 🎯 **Tích hợp MLflow**
            try:
                # Kiểm tra và tạo experiment nếu cần
                experiment_name = "Clustering"
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    try:
                        experiment_id = mlflow.create_experiment(experiment_name)
                        st.info(f"✅ Đã tạo mới experiment '{experiment_name}' với ID: {experiment_id}")
                    except Exception as e:
                        st.warning(f"⚠️ Không thể tạo experiment '{experiment_name}': {str(e)}. Sử dụng experiment mặc định (ID=0).")
                        experiment_id = "0"  # Fallback về experiment mặc định từ link DagsHub
                else:
                    experiment_id = experiment.experiment_id

                with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}", experiment_id=experiment_id) as run:
                    run_id = run.info.run_id

                    # Log các tham số
                    mlflow.log_params({"model": model_choice, **params})
                    mlflow.log_param("train_size", X_train.shape[0])
                    mlflow.log_param("test_size", X_test.shape[0])
                    mlflow.log_param("total_samples", st.session_state.total_samples)

                    # Lưu dữ liệu tạm thời và log artifact
                    os.makedirs("mlflow_artifacts", exist_ok=True)
                    np.save("mlflow_artifacts/X_train.npy", X_train)
                    np.save("mlflow_artifacts/X_test.npy", X_test)
                    np.save("mlflow_artifacts/y_train.npy", y_train)
                    np.save("mlflow_artifacts/y_test.npy", y_test)
                    mlflow.log_artifacts("mlflow_artifacts")

                    # Huấn luyện mô hình trên dữ liệu gốc 784 chiều
                    model.fit(X_train)
                    labels_train = model.labels_ if model_choice == "K-Means" else model.fit_predict(X_train)
                    
                    # Đánh giá trên tập train (chỉ cho K-Means)
                    if model_choice == "K-Means":
                        label_mapping = {}
                        for i in range(n_clusters):
                            mask = labels_train == i
                            if np.sum(mask) > 0:
                                most_common = np.bincount(y_train[mask].astype(int)).argmax()
                                label_mapping[i] = most_common
                        predicted_labels = np.array([label_mapping.get(label, 0) for label in labels_train])
                        train_accuracy = accuracy_score(y_train.astype(int), predicted_labels)
                        st.success(f"✅ Độ chính xác trên tập train: {train_accuracy:.4f}")
                        mlflow.log_metric("train_accuracy", train_accuracy)

                    # Đánh giá trên tập test
                    labels_test = model.predict(X_test) if model_choice == "K-Means" else model.fit_predict(X_test)
                    if model_choice == "K-Means":
                        test_predicted_labels = np.array([label_mapping.get(label, 0) for label in labels_test])
                        test_accuracy = accuracy_score(y_test.astype(int), test_predicted_labels)
                        st.success(f"✅ Độ chính xác trên tập test: {test_accuracy:.4f}")
                        mlflow.log_metric("test_accuracy", test_accuracy)

                    # Log mô hình
                    mlflow.sklearn.log_model(model, model_choice.lower())

            except Exception as e:
                st.error(f"⚠️ Lỗi khi log vào MLflow: {str(e)}. Huấn luyện cục bộ hoàn tất.")
                # Huấn luyện cục bộ nếu MLflow lỗi
                model.fit(X_train)
                labels_train = model.labels_ if model_choice == "K-Means" else model.fit_predict(X_train)
                if model_choice == "K-Means":
                    label_mapping = {}
                    for i in range(n_clusters):
                        mask = labels_train == i
                        if np.sum(mask) > 0:
                            most_common = np.bincount(y_train[mask].astype(int)).argmax()
                            label_mapping[i] = most_common
                    predicted_labels = np.array([label_mapping.get(label, 0) for label in labels_train])
                    train_accuracy = accuracy_score(y_train.astype(int), predicted_labels)
                    st.success(f"✅ Độ chính xác trên tập train: {train_accuracy:.4f}")
                run_id = None

            # Lưu mô hình vào session_state
            if "models" not in st.session_state:
                st.session_state["models"] = []
            
            model_name = model_choice.lower().replace(" ", "_")
            if model_choice == "K-Means":
                model_name += f"_{n_clusters}"
            else:
                model_name += f"_eps{eps}_min{min_samples}"
            
            existing_model = next((item for item in st.session_state["models"] if item["name"] == model_name), None)
            if existing_model:
                count = 1
                new_model_name = f"{model_name}_{count}"
                while any(item["name"] == new_model_name for item in st.session_state["models"]):
                    count += 1
                    new_model_name = f"{model_name}_{count}"
                model_name = new_model_name
                st.warning(f"⚠️ Mô hình được lưu với tên mới: {model_name}")
            
            st.session_state["models"].append({"name": model_name, "model": model})
            st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
            st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")
            st.write("📋 Danh sách các mô hình đã lưu:", ", ".join([m["name"] for m in st.session_state["models"]]))
            
            if run_id:  # Chỉ hiển thị link nếu log thành công
                mlflow_tracking_uri = "https://dagshub.com/TonThatTruongVu/MNIST-ClusteringAlgorithms.mlflow"
                experiment_id = experiment_id if 'experiment_id' in locals() else "0"  # Dùng ID=0 nếu không có experiment_id
                mlflow_link = f"{mlflow_tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
                st.success(f"✅ Đã log dữ liệu cho 'Train_{st.session_state['run_name']}'!")
                st.markdown(f"🔗 [Truy cập MLflow UI]({mlflow_link})")
            else:
                st.info("📝 Huấn luyện hoàn tất nhưng không log MLflow do lỗi kết nối.")

from PIL import Image
import numpy as np

def preprocess_canvas_image(canvas_result):
    """Xử lý hình ảnh từ canvas thành định dạng phù hợp với MNIST (784 chiều)."""
    if canvas_result.image_data is not None:
        try:
            # Chuyển dữ liệu canvas thành ảnh PIL
            img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
            # Chuyển thành grayscale
            img_gray = img.convert("L")
            # Resize về 28x28
            img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
            # Chuyển thành mảng NumPy và chuẩn hóa về [0, 1]
            img_normalized = np.array(img_resized) / 255.0
            return img_normalized.reshape(1, -1)  # Reshape thành (1, 784)
        except Exception as e:
            st.error(f"⚠️ Lỗi khi xử lý ảnh từ canvas: {str(e)}")
            return None
    return None

from PIL import Image
import numpy as np

def preprocess_uploaded_image(uploaded_file):
    """Xử lý ảnh tải lên thành định dạng phù hợp với MNIST (784 chiều)."""
    if uploaded_file is not None:
        try:
            # Đọc ảnh từ file tải lên
            img = Image.open(uploaded_file).convert("L")  # Chuyển sang grayscale
            # Resize về 28x28
            img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
            # Chuyển thành mảng NumPy và chuẩn hóa về [0, 1]
            img_normalized = np.array(img_resized) / 255.0
            return img_normalized.reshape(1, -1)  # Reshape thành (1, 784)
        except Exception as e:
            st.error(f"⚠️ Lỗi khi xử lý ảnh tải lên: {str(e)}")
            return None
    return None
def demo():
    st.header("✍️ Vẽ số hoặc tải ảnh để dự đoán cụm")

    # Kiểm tra xem có mô hình nào đã huấn luyện chưa
    if "models" not in st.session_state or not st.session_state["models"]:
        st.error("⚠️ Mô hình chưa được huấn luyện! Vui lòng huấn luyện mô hình trong tab 'Huấn luyện & Đánh giá' trước.")
        return

    # Dropdown chọn mô hình đã huấn luyện từ st.session_state["models"]
    st.subheader("🔍 Chọn mô hình đã huấn luyện")
    model_names = [model["name"] for model in st.session_state["models"]]
    model_option = st.selectbox("Chọn mô hình:", model_names)
    model = next(model["model"] for model in st.session_state["models"] if model["name"] == model_option)

    # Chọn phương thức nhập liệu
    input_method = st.selectbox("📌 Chọn phương thức nhập:", ["Vẽ số", "Tải ảnh"])

    if input_method == "Vẽ số":
        st.subheader("Vẽ số")
        if "key_value" not in st.session_state:
            st.session_state.key_value = str(random.randint(0, 1000000))
        if st.button("🔄 Tải lại nếu không thấy canvas"):
            st.session_state.key_value = str(random.randint(0, 1000000))
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key=st.session_state.key_value,
            update_streamlit=True
        )
        input_data = preprocess_canvas_image(canvas_result)
        source = "vùng vẽ"
    else:  # Tải ảnh
        st.subheader("Tải ảnh")
        uploaded_file = st.file_uploader("Chọn ảnh số (jpg, png)...", type=["jpg", "png"])
        input_data = preprocess_uploaded_image(uploaded_file)
        source = "ảnh tải lên"

    if st.button("Dự đoán cụm"):
        if input_data is not None:
            # Hiển thị ảnh đã xử lý
            st.image(
                Image.fromarray((input_data.reshape(28, 28) * 255).astype(np.uint8)),
                caption=f"Ảnh xử lý từ {source}",
                width=100
            )

            # Dự đoán cụm trên dữ liệu gốc 784 chiều
            if isinstance(model, KMeans):
                cluster = model.predict(input_data)[0]
                st.subheader(f"🔢 Cụm dự đoán: {cluster}")
            elif isinstance(model, DBSCAN):
                cluster = model.fit_predict(input_data)[0]
                st.subheader(f"🔢 Cụm dự đoán: {cluster if cluster != -1 else 'Nhiễu (không thuộc cụm)'}")
            show_experiment_selector()

        else:
            st.error(f"⚠️ Hãy {'vẽ một số' if input_method == 'Vẽ số' else 'tải ảnh'} trước khi dự đoán!")

def show_experiment_selector():
    st.title(" MLflow Experiments ")
    experiment_name = "Clustering"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")

    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])
    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    st.write("### 🏃‍♂️ Các Runs gần đây:")
    run_info = [(run["run_id"], run["params.run_name"] if "params.run_name" in run else f"Run {run['run_id'][:8]}") for _, run in runs.iterrows()]
    run_name_to_id = {name: rid for rid, name in run_info}
    selected_run_name = st.selectbox("🔍 Chọn một run:", list(run_name_to_id.keys()))
    selected_run_id = run_name_to_id[selected_run_name]

    selected_run = mlflow.get_run(selected_run_id)
    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        start_time = datetime.fromtimestamp(selected_run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M:%S") if selected_run.info.start_time else "Không có thông tin"
        st.write(f"**Thời gian chạy:** {start_time}")

        if selected_run.data.params:
            st.write("### ⚙️ Parameters:")
            st.json(selected_run.data.params)
        if selected_run.data.metrics:
            st.write("### 📊 Metrics:")
            st.json(selected_run.data.metrics)

def ly_thuyet_K_means():
    st.header("📌 Lý thuyết K-Means")
    st.write("""
    - **K-Means** là một thuật toán phân cụm **không giám sát** (unsupervised learning) nhằm chia dữ liệu thành **K cụm** (clusters) dựa trên sự tương đồng giữa các điểm dữ liệu. Thuật toán sử dụng **khoảng cách Euclidean** để đo lường sự gần gũi giữa các điểm và tâm cụm (centroids).
    """)

    st.subheader("🔍 Cách hoạt động chi tiết")
    st.markdown("""
    Thuật toán K-Means hoạt động qua các bước lặp đi lặp lại như sau:
    """)

    # Sử dụng expander để hiển thị từng bước chi tiết
    with st.expander("1. Khởi tạo tâm cụm (Initialization)"):
        st.markdown("""
        - Chọn ngẫu nhiên **K điểm** từ tập dữ liệu làm **tâm cụm ban đầu** (centroids).  
        - **Ví dụ**: Với K = 3, chọn 3 điểm ngẫu nhiên từ tập MNIST làm các tâm cụm khởi đầu.
        """)

    with st.expander("2. Gán nhãn cụm (Assignment Step)"):
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

    with st.expander("3. Cập nhật tâm cụm (Update Step)"):
        st.markdown("""
        - Sau khi gán tất cả điểm vào các cụm, tính lại **tâm cụm mới** bằng cách lấy **trung bình tọa độ** của mọi điểm trong cụm đó.  
        - **Công thức**:  
        """)
        st.latex(r"c_j = \frac{1}{N_j} \sum_{x \in C_j} x")
        st.markdown("""
        Trong đó:  
        - \( c_j \): Tâm cụm thứ \( j \).  
        - \( N_j \): Số điểm trong cụm \( j \).  
        - \( C_j \): Tập hợp các điểm thuộc cụm \( j \).
        """)

    with st.expander("4. Lặp lại (Iteration)"):
        st.markdown("""
        - Quay lại bước 2, lặp lại quá trình gán nhãn và cập nhật tâm cụm cho đến khi:  
          - Các tâm cụm không còn thay đổi đáng kể (hội tụ).  
          - Hoặc đạt số lần lặp tối đa (max iterations).
        """)

    st.subheader("💡 Ví dụ với MNIST")
    st.markdown("""
    - Nếu K = 10 (số chữ số từ 0-9), K-Means sẽ cố gắng nhóm các ảnh chữ số thành 10 cụm.  
    - Ban đầu, chọn 10 ảnh ngẫu nhiên làm tâm. Sau vài lần lặp, các tâm cụm dần đại diện cho các nhóm chữ số (ví dụ: cụm 0 chứa hầu hết ảnh số 0).
    """)


def ly_thuyet_DBSCAN():
    st.header("📌 Lý thuyết DBSCAN")
    st.write("""
    - **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) là một thuật toán phân cụm **không giám sát** dựa trên **mật độ** của các điểm dữ liệu. 
    - Khác với K-Means, DBSCAN không yêu cầu xác định trước số cụm, mà tự động tìm các cụm dựa trên phân bố dữ liệu và có khả năng phát hiện **nhiễu** (noise).
    """)

    st.subheader("🔍 Cách hoạt động chi tiết")
    st.markdown("""
    DBSCAN phân cụm dựa trên hai tham số chính:  
    - **eps**: Bán kính lân cận (khoảng cách tối đa giữa hai điểm để coi là "gần nhau").  
    - **min_samples**: Số điểm tối thiểu trong vùng lân cận để hình thành một cụm.  
    Các bước cụ thể:
    """)

    # Sử dụng expander để hiển thị từng bước
    with st.expander("1. Xác định các loại điểm (Point Classification)"):
        st.markdown("""
        - **Core Point (Điểm lõi)**: Một điểm có ít nhất **min_samples** láng giềng (bao gồm chính nó) trong bán kính **eps**.  
        - **Border Point (Điểm ranh giới)**: Không phải điểm lõi, nhưng nằm trong bán kính **eps** của ít nhất một điểm lõi.  
        - **Noise Point (Điểm nhiễu)**: Không phải điểm lõi, không nằm trong bán kính **eps** của bất kỳ điểm lõi nào.  
        - **Ví dụ**: Với MNIST, một điểm lõi có thể là trung tâm của vùng chữ số "0", các điểm ranh giới là viền, và nhiễu là các nét lỗi.
        """)

    with st.expander("2. Khởi tạo cụm (Cluster Initialization)"):
        st.markdown("""
        - Chọn một **điểm lõi chưa thăm** (unvisited core point) làm hạt giống (seed).  
        - Tạo cụm mới từ điểm này để bắt đầu quá trình phân cụm.
        """)

    with st.expander("3. Mở rộng cụm (Cluster Expansion)"):
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

    with st.expander("4. Đánh dấu nhiễu và lặp lại"):
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

def main():
    st.title("🖊️ MNIST Clustering with Streamlit & MLflow")
    
    if "mlflow_initialized" not in st.session_state:
        mlflow.set_tracking_uri("https://dagshub.com/TonThatTruongVu/MNIST-ClusteringAlgorithms.mlflow")
        os.environ["MLFLOW_TRACKING_USERNAME"] = "TonThatTruongVu"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "519c4a864e131de52197f54d170c130beb15ffd5"
        mlflow.set_experiment("Clustering")
        st.session_state.mlflow_url = "https://dagshub.com/TonThatTruongVu/MNIST-ClusteringAlgorithms.mlflow"
        st.session_state.mlflow_initialized = True

    tabs = st.tabs(["Lý thuyết K-Means", "Lý thuyết DBSCAN", "Data", "Huấn luyện", "Dự đoán"])
    
    with tabs[0]:
        ly_thuyet_K_means()
    with tabs[1]:
        ly_thuyet_DBSCAN()
    with tabs[2]:
        data_processing()
    with tabs[3]:
        split_data()
        train_evaluate()
    with tabs[4]:
        demo()

if __name__ == "__main__":
    main()