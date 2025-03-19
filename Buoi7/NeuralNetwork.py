import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image
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

# Hàm khởi tạo MLflow
def mlflow_input():
    try:
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/TonThatTruongVu/MNIST-NeuralNetwork.mlflow"
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
        os.environ["MLFLOW_TRACKING_USERNAME"] = "TonThatTruongVu"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "519c4a864e131de52197f54d170c130beb15ffd5"
        mlflow.set_experiment("MNIST_NeuralNetwork")
        st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
        st.success("✅ MLflow được khởi tạo thành công!")
    except Exception as e:
        st.error(f"❌ Lỗi khi khởi tạo MLflow: {str(e)}")
        traceback.print_exc()

# Hàm tải dữ liệu từ OpenML
@st.cache_data
def load_mnist_data():
    try:
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float32) / 255.0
        y = y.astype(np.uint8)
        return X, y
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu MNIST từ OpenML: {str(e)}")
        return None, None

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
            if X is not None and y is not None:
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.data_loaded = True
                st.success("✅ Dữ liệu đã được tải thành công!")
            else:
                st.error("❌ Không thể tải dữ liệu!")

    if st.session_state.data_loaded:
        X, y = st.session_state.X, st.session_state.y
        st.write(f"""
            **Thông tin tập dữ liệu MNIST:**
            - Tổng số mẫu: {X.shape[0]}
            - Kích thước mỗi ảnh: 28 × 28 pixels (784 đặc trưng)
            - Số lớp: 10 (chữ số từ 0-9)
        """)

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
# Tab chia dữ liệu
def split_data():
    st.header("📌 Chia dữ liệu Train/Validation/Test")
    if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
        st.warning("⚠ Vui lòng tải dữ liệu từ tab 'Dữ Liệu' trước khi tiếp tục!")
        return

    X, y = st.session_state.X, st.session_state.y
    total_samples = X.shape[0]
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False

    num_samples = st.slider("📌 Chọn số lượng ảnh để train:", 1000, total_samples, 10000)
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    if st.button("✅ Xác nhận & Lưu") and not st.session_state.data_split_done:
        try:
            indices = np.random.choice(total_samples, num_samples, replace=False)
            X_selected = X[indices]
            y_selected = y[indices]

            stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
            )

            stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_size/(100 - test_size),
                stratify=stratify_option, random_state=42
            )

            st.session_state.total_samples = num_samples
            st.session_state.X_train = X_train
            st.session_state.X_val = X_val
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_val = y_val
            st.session_state.y_test = y_test
            st.session_state.test_size = X_test.shape[0]
            st.session_state.val_size = X_val.shape[0]
            st.session_state.train_size = X_train.shape[0]
            st.session_state.data_split_done = True

            summary_df = pd.DataFrame({
                "Tập dữ liệu": ["Train", "Validation", "Test"],
                "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
            })
            st.success("✅ Dữ liệu đã được chia thành công!")
            st.table(summary_df)
        except Exception as e:
            st.error(f"❌ Lỗi khi chia dữ liệu: {str(e)}")
            traceback.print_exc()

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, không cần chạy lại.")

# Hàm huấn luyện với Cross-Validation
def thi_nghiem():
    st.header("⚙️ Huấn luyện Neural Network với Cross-Validation")
    
    num = 0
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return
    
    X_train, X_val, X_test = [st.session_state[k].reshape(-1, 28 * 28) / 255.0 for k in ["X_train", "X_val", "X_test"]]
    y_train, y_val, y_test = [st.session_state[k] for k in ["y_train", "y_val", "y_test"]]
    
    k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
    num_layers = st.slider("Số lớp ẩn:", 1, 5, 2)
    num_neurons = st.slider("Số neuron mỗi lớp:", 32, 512, 128, 32)
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh", "softmax"])
    optimizer = st.selectbox("Optimizer:", ["adam", "sgd", "rmsprop"])
    epochs = st.slider("🕰 Số epochs:", min_value=1, max_value=50, value=20, step=1)
    learning_rate = st.slider("⚡ Tốc độ học (Learning Rate):", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")

    loss_fn = "sparse_categorical_crossentropy"
    run_name = st.text_input("🔹 Nhập tên Run:", f"TF_NN_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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

                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(X_k_train.shape[1],))
                ] + [
                    tf.keras.layers.Dense(num_neurons, activation=activation) for _ in range(num_layers)
                ] + [
                    tf.keras.layers.Dense(10, activation="softmax")
                ])

                if optimizer == "adam":
                    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                elif optimizer == "sgd":
                    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
                else:
                    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

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

# Hàm hiển thị thông tin MLflow Experiments
def show_experiment_selector():
    if 'mlflow_url' not in st.session_state:
        st.warning("⚠️ URL MLflow chưa được khởi tạo!")
        mlflow_input()

    st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
    experiment_name = "MNIST_NeuralNetwork"
    
    try:
        experiments = mlflow.search_experiments()
        selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

        if not selected_experiment:
            st.error(f"❌ Không tìm thấy Experiment '{experiment_name}'!")
            return

        st.subheader(f"📌 Experiment: {experiment_name}")
        st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
        st.write(f"**Trạng thái:** {'🟢 Active' if selected_experiment.lifecycle_stage == 'active' else '🔴 Deleted'}")
        st.write(f"**Artifact Location:** `{selected_experiment.artifact_location}`")

        runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])
        if runs.empty:
            st.warning("⚠ Không có runs nào trong experiment này!")
            return

        st.subheader("🏃‍♂️ Danh sách Runs")
        run_info = []
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            run_data = mlflow.get_run(run_id)
            run_name = run_data.info.run_name if run_data.info.run_name else f"Run_{run_id[:8]}"
            run_info.append((run_name, run_id))

        run_name_to_id = dict(run_info)
        run_names = list(run_name_to_id.keys())

        selected_run_name = st.selectbox("🔍 Chọn Run để xem chi tiết:", run_names, key="run_selector_du_doan")
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
                start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"
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
    except Exception as e:
        st.error(f"❌ Lỗi khi truy cập MLflow: {str(e)}")
        traceback.print_exc()

# Tab dự đoán
def du_doalkan():
    st.header("✍️ Dự đoán số viết tay")

    if 'mlflow_url' not in st.session_state:
        st.warning("⚠️ MLflow chưa được khởi tạo. Đang khởi tạo...")
        mlflow_input()

    model = None
    try:
        experiment_name = "MNIST_NeuralNetwork"
        experiments = mlflow.search_experiments()
        experiment = next((exp for exp in experiments if exp.name == experiment_name), None)
        if not experiment:
            st.error("❌ Không tìm thấy experiment 'MNIST_NeuralNetwork'!")
        else:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            successful_runs = runs[runs["status"] == "FINISHED"]
            if successful_runs.empty:
                st.error("⚠️ Chưa có mô hình nào được huấn luyện thành công!")
            else:
                run_options = {f"{row['tags.mlflow.runName']} (Run ID: {row['run_id'][:8]})": row["run_id"] 
                            for _, row in successful_runs.iterrows()}
                selected_run_name = st.selectbox("📌 Chọn mô hình đã huấn luyện:", list(run_options.keys()))
                selected_run_id = run_options[selected_run_name]

                model_uri = f"runs:/{selected_run_id}/neural_network"
                for attempt in range(3):
                    try:
                        model = mlflow.keras.load_model(model_uri)
                        st.success(f"✅ Đã chọn mô hình: {selected_run_name}")
                        break
                    except MlflowException as e:
                        st.warning(f"⚠️ Lỗi tải mô hình (thử {attempt+1}/3): {str(e)}")
                        if attempt == 2:
                            st.error("❌ Không thể tải mô hình sau 3 lần thử!")
                            model = None
                        time.sleep(2)

    except Exception as e:
        st.error(f"❌ Lỗi khi truy cập MLflow: {str(e)}")
        traceback.print_exc()

    if model is None and 'trained_model' in st.session_state:
        st.warning("⚠️ Không tải được mô hình từ MLflow, dùng mô hình cục bộ đã huấn luyện.")
        model = st.session_state['trained_model']

    if model is None:
        st.error("❌ Không có mô hình nào để dự đoán!")
        return

    input_method = st.radio("📥 Chọn phương thức nhập liệu:", ("Vẽ tay", "Tải ảnh lên"))
    img = None
    if input_method == "Vẽ tay":
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
        if st.button("Dự đoán số từ bản vẽ"):
            if canvas_result.image_data is not None:
                img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
                img = img.resize((28, 28)).convert("L")
                img = np.array(img, dtype=np.float32) / 255.0
                img = img.reshape(1, -1)
            else:
                st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")

    else:
        uploaded_file = st.file_uploader("📤 Tải ảnh lên (định dạng PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Ảnh đã tải lên", width=150)
            if st.button("Dự đoán số từ ảnh"):
                img = Image.open(uploaded_file).convert("L")
                img = img.resize((28, 28))
                img = np.array(img, dtype=np.float32) / 255.0
                img = img.reshape(1, -1)

    if img is not None:
        st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)
        prediction = np.argmax(model.predict(img), axis=1)[0]
        st.subheader(f"🔢 Dự đoán: {prediction}")

        confidence_scores = model.predict(img)[0]
        predicted_class_confidence = confidence_scores[prediction]
        st.write(f"📈 **Độ tin cậy:** {predicted_class_confidence:.4f} ({predicted_class_confidence * 100:.2f}%)")

        st.write("**Xác suất cho từng lớp (0-9):**")
        confidence_df = pd.DataFrame({"Nhãn": range(10), "Xác suất": confidence_scores})
        st.bar_chart(confidence_df.set_index("Nhãn"))

        # Hiển thị thông tin MLflow sau khi dự đoán
        st.subheader("📊 Thông tin chi tiết từ MLflow")
        show_experiment_selector()

# Giao diện chính
def main():
    if "mlflow_initialized" not in st.session_state:
        mlflow_input()
        st.session_state.mlflow_initialized = True

    st.title("🖊️ MNIST Neural Network Classification App")
    tabs = st.tabs(["📘 Dữ Liệu", "📌 Chia Dữ Liệu", "⚙️ Huấn Luyện", "🔢 Dự Đoán"])

    with tabs[0]:
        data()
    with tabs[1]:
        split_data()
    with tabs[2]:
        thi_nghiem()
    with tabs[3]:
        du_doalkan()

if __name__ == "__main__":
    main()