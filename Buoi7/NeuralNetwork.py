import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import os
import mlflow
import mlflow.keras
import random
from datetime import datetime
import matplotlib.pyplot as plt
import traceback
import time
import requests
from mlflow.exceptions import MlflowException
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Tắt GPU để tránh lỗi cuDNN/cuBLAS (tạm thời) và tắt oneDNN
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Chỉ dùng CPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Tắt thông báo oneDNN

# Hàm khởi tạo MLflow
def mlflow_input():
    try:
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/Phuocbinh2003/Hoc_may_python.mlflow"
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
        os.environ["MLFLOW_TRACKING_USERNAME"] = "Phuocbinh2003"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1495823c8f9156923b06f15899e989db7e62052"  # Cập nhật token nếu cần
        mlflow.set_experiment("Neural_Network")
        st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
        st.success("✅ MLflow được khởi tạo thành công!")
    except Exception as e:
        st.error(f"❌ Lỗi khi khởi tạo MLflow: {str(e)}")
        traceback.print_exc()

# Hàm kiểm tra kết nối MLflow
def check_mlflow_connection():
    try:
        response = requests.get(st.session_state['mlflow_url'], timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False

# Hàm tải dữ liệu từ OpenML hoặc file cục bộ
@st.cache_data
def load_mnist_data():
    try:
        # Thử tải từ file cục bộ trước
        if os.path.exists("buoi4/X.npy") and os.path.exists("buoi4/y.npy"):
            X = np.load("buoi4/X.npy")
            y = np.load("buoi4/y.npy")
            st.success("✅ Đã tải dữ liệu từ file cục bộ!")
        else:
            # Nếu không có file cục bộ, tải từ OpenML
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
            np.save("buoi4/X.npy", X)
            np.save("buoi4/y.npy", y)
            st.success("✅ Đã tải dữ liệu từ OpenML và lưu cục bộ!")
        X = X.astype(np.float32) / 255.0
        y = y.astype(np.uint8)
        return X, y
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu MNIST: {str(e)}")
        return None, None

# Tab lý thuyết Neural Network
def explain_nn():
    st.markdown("""
    ## 🧠 Neural Network Cơ Bản

    **Neural Network (Mạng nơ-ron nhân tạo - ANN)** là một mô hình tính toán lấy cảm hứng từ cấu trúc và hoạt động của não bộ con người. Mạng bao gồm nhiều nơ-ron nhân tạo kết nối với nhau thành các lớp (layers), giúp mô hình học và nhận diện các mẫu trong dữ liệu.

    ### 🔰 Kiến trúc cơ bản:
    ### 📌 Cấu trúc của một mạng nơ-ron nhân tạo gồm ba loại lớp chính:
    1. **Input Layer**: Lớp tiếp nhận dữ liệu đầu vào.
    2. **Hidden Layers**: Xử lý thông tin thông qua các trọng số (weights) và hàm kích hoạt.
    3. **Output Layer**: Lớp đưa ra kết quả dự đoán.
    """)
    
    st.image("buoi7/img3.png", caption="Cấu trúc mạng nơ-ron (medium.com)", use_container_width=True)
    
    st.markdown("""
    ## 📌 Công thức toán học trong Neural Network:
    Mỗi nơ-ron trong một lớp nhận tín hiệu từ các nơ-ron lớp trước, nhân với trọng số (**weights**), cộng với **bias**, rồi đưa vào một **hàm kích hoạt** để quyết định tín hiệu truyền đi.
    """)

    st.markdown("### 🎯 Công thức tính giá trị đầu ra của một nơ-ron:")
    st.latex(r" z = \sum_{i=1}^{n} w_i x_i + b ")

    st.markdown(r"""
    Trong đó:
    - $$ x_i $$ là đầu vào (**input features**).
    - $$ w_i $$ là **trọng số** (**weights**) kết nối với nơ-ron đó.
    - $$ b $$ là **bias** (hệ số dịch chuyển).
    - $$ z $$ là tổng có trọng số (**weighted sum**).

    Sau khi tính toán $$ z $$, nó sẽ đi qua một **hàm kích hoạt** $$ f(z) $$ để tạo ra giá trị đầu ra.
    """)

    st.markdown("""
    ### 🎯 Hàm Kích Hoạt (Activation Functions)
    Hàm kích hoạt giúp mạng học được các tính phi tuyến tính, nhờ đó có thể mô hình hóa các mối quan hệ phức tạp.
    """)
    
    st.image("buoi7/img2.png", caption="Một số hàm kích hoạt cơ bản", use_container_width=True)
    
    st.markdown("- **Sigmoid:** Chuyển đổi giá trị đầu vào thành khoảng từ 0 đến 1, phù hợp cho bài toán phân loại nhị phân.")
    st.latex(r"f(z) = \sigma(z) = \frac{1}{1 + e^{-z}}")

    st.markdown("- **Tanh (Hyperbolic Tangent):** Đầu ra nằm trong khoảng từ -1 đến 1, giúp xử lý dữ liệu có cả giá trị dương và âm.")
    st.latex(r"f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}")

    st.markdown("- **ReLU (Rectified Linear Unit):** Nếu đầu vào âm thì bằng 0, còn nếu dương thì giữ nguyên giá trị.")
    st.latex(r"f(z) = ReLU(z) = \max(0, z)")

    st.markdown("### 🔄 Quá trình huấn luyện Neural Network")
    st.markdown("Mạng nơ-ron học bằng cách cập nhật các trọng số thông qua hai giai đoạn chính:")

    st.markdown("#### 1️⃣ Lan truyền thuận (Forward Propagation)")
    st.markdown("- Input đi qua từng lớp nơ-ron, tính toán đầu ra:")
    st.latex(r"f^{(l)} = \sigma(W^{(l)} f^{(l-1)} + b^{(l)})")

    st.markdown("#### 2️⃣ Tính toán loss")
    st.markdown("- Hàm mất mát đo lường sai số giữa dự đoán và thực tế.")
    st.latex(r"L = - \sum y_{true} \log(y_{pred})")  # Cross-Entropy Loss

    st.markdown("#### 3️⃣ Lan truyền ngược (Backpropagation)")
    st.markdown("- Tính đạo hàm của hàm mất mát theo trọng số và cập nhật trọng số.")
    st.latex(r"W^{(l)} = W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}")

    st.markdown("#### 4️⃣ Tối ưu hóa")
    st.markdown("- **Adam:** Một trong những thuật toán tối ưu phổ biến cho Neural Network.")

# Tab dữ liệu MNIST
def data():
    st.header("MNIST Dataset")
    st.write("""
      **MNIST** là một trong những bộ dữ liệu nổi tiếng và phổ biến nhất trong cộng đồng học máy, 
      đặc biệt là trong các nghiên cứu về nhận diện mẫu và phân loại hình ảnh.
  
      - Bộ dữ liệu bao gồm tổng cộng **70.000 ảnh chữ số viết tay** từ **0** đến **9**, 
        mỗi ảnh có kích thước **28 x 28 pixel**.
      - Chia thành:
        - **Training set**: 60.000 ảnh để huấn luyện.
        - **Test set**: 10.000 ảnh để kiểm tra.
      - Mỗi hình ảnh là một chữ số viết tay, được chuẩn hóa và chuyển thành dạng grayscale (đen trắng).
    """)

    st.subheader("Một số hình ảnh từ MNIST Dataset")
    st.image("buoi4/img3.png", caption="Một số hình ảnh từ MNIST Dataset", use_container_width=True)

    st.subheader("📊 Minh họa dữ liệu MNIST")
    st.image("buoi7/g1.gif", caption="Hình ảnh minh họa dữ liệu MNIST", use_container_width=True)

# Tab chia dữ liệu
def split_data():
    st.title("📌 Chia dữ liệu Train/Test")
    
    X, y = load_mnist_data()
    if X is None or y is None:
        return
    
    total_samples = X.shape[0]
    num_samples = st.slider("📌 Chọn số lượng ảnh để huấn luyện:", 1000, total_samples, 10000)
    num_samples = num_samples - 10
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    train_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong Train)", 0, 50, 15)
    
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={train_size - val_size}%")
    
    if st.button("✅ Xác nhận & Lưu"):
        X_selected, _, y_selected, _ = train_test_split(X, y, train_size=num_samples, stratify=y, random_state=42)
        X_train_full, X_test, y_train_full, y_test = train_test_split(X_selected, y_selected, test_size=test_size/100, stratify=y_selected, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size / (100 - test_size), stratify=y_train_full, random_state=42)
        
        st.session_state.update({
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test
        })
        
        summary_df = pd.DataFrame({"Tập dữ liệu": ["Train", "Validation", "Test"], "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]})
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)

# Tab huấn luyện
def thi_nghiem():
    st.header("🧠 Huấn luyện Neural Network trên MNIST")
    
    num = 0
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return
    
    X_train, X_val, X_test = [st.session_state[k].reshape(-1, 28 * 28) / 255.0 for k in ["X_train", "X_val", "X_test"]]
    y_train, y_val, y_test = [st.session_state[k] for k in ["y_train", "y_val", "y_test"]]
    
    k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
    num_layers = st.slider("Số lớp ẩn:", 1, 5, 2)
    num_neurons = st.slider("Số neuron mỗi lớp:", 32, 512, 128, 32)
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"])
    optimizer = st.selectbox("Optimizer:", ["adam", "sgd", "rmsprop"])
    epochs = st.slider("🕰 Số epochs:", min_value=1, max_value=50, value=20, step=1)
    learning_rate = st.slider("⚡ Tốc độ học (Learning Rate):", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")

    loss_fn = "sparse_categorical_crossentropy"
    run_name = st.text_input("🔹 Nhập tên Run:", f"NN_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    st.session_state['run_name'] = run_name
    
    if st.button("🚀 Huấn luyện mô hình"):
        with st.spinner("Đang huấn luyện..."):
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "num_layers": num_layers,
                "num_neurons": num_neurons,
                "activation": activation,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "k_folds": k_folds,
                "epochs": epochs
            })

            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            accuracies, losses = [], []

            training_progress = st.progress(0)
            training_status = st.empty()

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                X_k_train, X_k_val = X_train[train_idx], X_train[val_idx]
                y_k_train, y_k_val = y_train[train_idx], y_train[val_idx]

                model = Sequential([
                    Input(shape=(X_k_train.shape[1],))
                ] + [
                    Dense(num_neurons, activation=activation) for _ in range(num_layers)
                ] + [
                    Dense(10, activation="softmax")
                ])

                if optimizer == "adam":
                    opt = Adam(learning_rate=learning_rate)
                elif optimizer == "sgd":
                    opt = SGD(learning_rate=learning_rate)
                else:
                    opt = RMSprop(learning_rate=learning_rate)

                model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

                start_time = time.time()
                history = model.fit(X_k_train, y_k_train, epochs=epochs, validation_data=(X_k_val, y_k_val), verbose=0)
                elapsed_time = time.time() - start_time

                accuracies.append(history.history["val_accuracy"][-1])
                losses.append(history.history["val_loss"][-1])

                num += 1
                progress_percent = min(int((num / k_folds) * 100), 100)
                training_progress.progress(progress_percent)
                training_status.text(f"⏳ Đang huấn luyện... Fold {num}/{k_folds} ({progress_percent}%)")

            avg_val_accuracy = np.mean(accuracies)
            avg_val_loss = np.mean(losses)

            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            mlflow.log_metrics({
                "avg_val_accuracy": avg_val_accuracy,
                "avg_val_loss": avg_val_loss,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "elapsed_time": elapsed_time
            })

            mlflow.keras.log_model(model, "neural_network")
            mlflow.end_run()
            st.session_state["trained_model"] = model

            training_progress.progress(100)
            training_status.text("✅ Huấn luyện hoàn tất!")

            st.success(f"✅ Huấn luyện hoàn tất!")
            st.write(f"📊 **Độ chính xác trung bình trên tập validation:** {avg_val_accuracy:.4f}")
            st.write(f"📊 **Độ chính xác trên tập test:** {test_accuracy:.4f}")
            st.success(f"✅ Đã log dữ liệu cho Experiments Neural_Network với Name: **{st.session_state['run_name']}**!")
            st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")

# Tab dự đoán
def du_doan():
    st.header("✍️ Vẽ số để dự đoán")

    if "trained_model" not in st.session_state:
        st.error("⚠️ Chưa có mô hình! Hãy huấn luyện trước.")
        return

    model = st.session_state["trained_model"]
    st.success("✅ Đã sử dụng mô hình vừa huấn luyện!")

    if "key_value" not in st.session_state:
        st.session_state.key_value = str(random.randint(0, 1000000))

    if st.button("🔄 Tải lại nếu không thấy canvas"):
        st.session_state.key_value = str(random.randint(0, 1000000))

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key=st.session_state.key_value,
        update_streamlit=True
    )

    if st.button("Dự đoán số"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data[:, :, :3]).convert("L").resize((28, 28))
            img = np.array(img, dtype=np.float32) / 255.0
            img = img.reshape(1, -1)

            st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)

            prediction = model.predict(img)
            predicted_number = np.argmax(prediction, axis=1)[0]
            max_confidence = np.max(prediction)

            st.subheader(f"🔢 Dự đoán: {predicted_number}")
            st.write(f"📊 Mức độ tin cậy: {max_confidence:.2%}")

            prob_df = pd.DataFrame(prediction.reshape(1, -1), columns=[str(i) for i in range(10)]).T
            prob_df.columns = ["Mức độ tin cậy"]
            st.bar_chart(prob_df)
        else:
            st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")

# Tab MLflow
def show_experiment_selector():
    st.title("📊 MLflow")
    
    if 'mlflow_url' not in st.session_state:
        st.warning("⚠️ MLflow chưa được khởi tạo!")
        mlflow_input()

    experiment_name = "Neural_Network"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])
    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    st.write("### 🏃‍♂️ Các Runs gần đây:")
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_tags = mlflow.get_run(run_id).data.tags
        run_name = run_tags.get("mlflow.runName", f"Run {run_id[:8]}")
        run_info.append((run_name, run_id))
    
    run_name_to_id = dict(run_info)
    run_names = list(run_name_to_id.keys())
    
    selected_run_name = st.selectbox("🔍 Chọn một run:", run_names)
    selected_run_id = run_name_to_id[selected_run_name]
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        
        start_time_ms = selected_run.info.start_time
        start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"
        st.write(f"**Thời gian chạy:** {start_time}")

        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

# Giao diện chính
def Neural_Network():
    if "mlflow_initialized" not in st.session_state:
        mlflow_input()
        st.session_state.mlflow_initialized = True

    st.title("🚀 Neural Network Classification App")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📘 Lý thuyết NEURAL NETWORK",
        "📊 Mẫu dữ liệu",
        "🧠 Huấn luyện",
        "🖥️ DEMO",
        "🔥 MLflow"
    ])

    with tab1:
        explain_nn()
    with tab2:
        data()
    with tab3:
        split_data()
        thi_nghiem()
    with tab4:
        du_doan()
    with tab5:
        show_experiment_selector()

if __name__ == "__main__":
    Neural_Network()