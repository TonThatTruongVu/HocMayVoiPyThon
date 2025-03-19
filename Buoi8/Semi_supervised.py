import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
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

# Hàm khởi tạo MLflow
def mlflow_input():
    try:
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/TonThatTruongVu/MNIST-NeuralNetwork.mlflow"
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
        os.environ["MLFLOW_TRACKING_USERNAME"] = "TonThatTruongVu"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "519c4a864e131de52197f54d170c130beb15ffd5"  # Cập nhật token nếu cần
        mlflow.set_experiment("Neural_Network_Pseudo_Labelling")
        st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
        st.success("✅ MLflow được khởi tạo thành công!")
    except Exception as e:
        st.error(f"❌ Lỗi khi khởi tạo MLflow: {str(e)}")
        traceback.print_exc()
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
import streamlit as st

def explain_pseudo_labeling():
    st.header("🧠 Lý thuyết về Pseudo Labelling")
    
    # Giới thiệu tổng quan
    st.subheader("Pseudo Labelling là gì?")
    st.write("""
    **Pseudo Labelling** là một phương pháp **semi-supervised learning** (học bán giám sát) giúp kết hợp dữ liệu có nhãn và không nhãn để cải thiện độ chính xác của mô hình. Phương pháp này đặc biệt hữu ích trong các bài toán phân loại khi dữ liệu có nhãn bị hạn chế.
    """)
    
    # Cách hoạt động
    st.subheader("Pseudo Labelling hoạt động như thế nào?")
    st.write("""
    Pseudo Labeling hiểu đơn giản là bạn sử dụng một mô hình sau khi huấn luyện với dữ liệu có nhãn để dự đoán **“nhãn giả”** cho các dữ liệu không nhãn. Sau đó, dữ liệu có nhãn ban đầu được kết hợp với dữ liệu có nhãn giả vừa tạo để huấn luyện lại mô hình. 

    Để đảm bảo chất lượng nhãn giả, ta thường lọc ra những dự đoán có **độ tin cậy cao** (ví dụ: xác suất dự đoán vượt qua một ngưỡng - **threshold**, chẳng hạn 0.95). Điều này giúp giảm thiểu nhiễu từ các nhãn giả không chính xác.
    """)
    
    # Minh họa bằng ảnh
    st.image("https://images.viblo.asia/6bfb0385-865f-415f-a472-b2d1ca94b79b.png", 
             caption="Quy trình Pseudo Labelling (Nguồn: [viblo.asia](https://viblo.asia/p/doi-dong-ve-pseudo-labeling-trong-machine-learning-1VgZvQmrKAw))")

    # Lợi ích
    st.subheader("Lợi ích của Pseudo Labelling")
    st.write("""
    Pseudo Labeling là một phương pháp hiệu quả giúp:
    - **Cải thiện độ chính xác** của bài toán phân loại.
    - Tận dụng tối đa dữ liệu không nhãn, đặc biệt trong trường hợp dữ liệu có nhãn bị hạn chế.
    """)
    
    # Lưu ý khi sử dụng
    st.subheader("Một số lưu ý khi dùng Pseudo Labelling")
    st.write("""
    Khi áp dụng Pseudo Labelling, bạn nên cân nhắc những điểm sau:

    1. **Không nên trộn lẫn dữ liệu có nhãn và nhãn giả một cách đơn giản**:  
       - Nên tách biệt dữ liệu nhãn thật và nhãn giả để sử dụng **hai hàm loss riêng biệt**.  
       - Hàm loss cho dữ liệu nhãn giả nên có **trọng số thấp hơn** (weight) nhằm giảm ảnh hưởng của nhãn giả không chính xác.  

    2. **Thử nghiệm trộn lẫn dữ liệu**:  
        - Bạn cũng có thể trộn lẫn dữ liệu có nhãn và nhãn giả để xem kết quả thế nào. Tuy nhiên, sau cùng, mô hình cần được đánh giá trên **tập test** để đảm bảo tính khách quan.

    Những lưu ý này giúp tối ưu hóa hiệu quả của Pseudo Labelling và giảm thiểu rủi ro từ nhãn giả sai lệch.
    """)
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

    num_samples = st.slider("📌 Chọn số lượng ảnh tổng cộng:", 1000, total_samples, 6500, key="split_num_samples")
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20, key="split_test_size")
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần còn lại)", 0, 50, 15, key="split_val_size")
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    if st.button("✅ Xác nhận & Chia dữ liệu", key="split_button") and not st.session_state.data_split_done:
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

    if st.session_state.data_split_done:
        st.subheader("📌 Chọn % dữ liệu labeled ban đầu")
        labeled_percent = st.slider(
            "Chọn % số lượng ảnh cho tập labeled ban đầu:", 
            1, 100, 1, 
            key="labeled_percent",
            help="Phần trăm này sẽ được áp dụng trên toàn bộ tập Train."
        )

        if st.button("✅ Xác nhận % dữ liệu labeled", key="labeled_button"):
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            total_train_samples = len(X_train)
            num_labeled = int(total_train_samples * (labeled_percent / 100))  # Tính số mẫu dựa trên tổng Train

            # Lấy mẫu ngẫu nhiên từ toàn bộ tập Train, sau đó phân bổ đều cho các class
            labeled_indices = []
            for digit in range(10):
                digit_indices = np.where(y_train == digit)[0]
                num_samples_per_class = max(1, int(num_labeled / 10))  # Phân bổ đều cho 10 class
                if len(digit_indices) < num_samples_per_class:
                    num_samples_per_class = len(digit_indices)  # Nếu class có ít mẫu hơn
                selected_indices = digit_indices[:num_samples_per_class]
                labeled_indices.extend(selected_indices)

            # Nếu số mẫu chưa đủ num_labeled, lấy thêm ngẫu nhiên từ các class còn lại
            if len(labeled_indices) < num_labeled:
                remaining_indices = [i for i in range(len(X_train)) if i not in labeled_indices]
                additional_indices = np.random.choice(remaining_indices, num_labeled - len(labeled_indices), replace=False)
                labeled_indices.extend(additional_indices)

            X_labeled = X_train[labeled_indices]
            y_labeled = y_train[labeled_indices]
            unlabeled_indices = [i for i in range(len(X_train)) if i not in labeled_indices]
            X_unlabeled = X_train[unlabeled_indices]
            y_unlabeled = y_train[unlabeled_indices]

            # Lưu vào session_state
            st.session_state.X_labeled = X_labeled
            st.session_state.y_labeled = y_labeled
            st.session_state.X_unlabeled = X_unlabeled
            st.session_state.y_unlabeled = y_unlabeled

            st.write(f"✅ Tập labeled ban đầu: {len(X_labeled)} mẫu")
            st.write(f"✅ Tập unlabeled còn lại: {len(X_unlabeled)} mẫu")
        #    st.write("""
        #    **Cách lấy dữ liệu labeled:**
        #    - Tính tổng số mẫu labeled dựa trên % của toàn bộ tập Train.
        #    - Phân bổ đều số mẫu cho mỗi class (0-9), lấy các mẫu đầu tiên.
        #    - Nếu chưa đủ số mẫu mong muốn, lấy thêm ngẫu nhiên từ các mẫu còn lại.
        #    """)

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, bạn có thể chọn % dữ liệu labeled bên dưới.")
import matplotlib.pyplot as plt
import tensorflow as tf
from mlflow.models.signature import infer_signature
import os  # Thêm import này để kiểm tra thư mục




import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt

def train():
    st.header("⚙️ Huấn luyện Neural Network với Pseudo Labelling")
    if "X_labeled" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu labeled! Hãy chọn % dữ liệu labeled trong tab 'Chia dữ liệu' trước.")
        return

    # Lấy dữ liệu từ session_state
    X_labeled = st.session_state.X_labeled
    y_labeled = st.session_state.y_labeled
    X_unlabeled = st.session_state.X_unlabeled
    y_unlabeled = st.session_state.y_unlabeled
    X_val = st.session_state.X_val
    X_test = st.session_state.X_test
    y_val = st.session_state.y_val
    y_test = st.session_state.y_test

    # Các tham số huấn luyện
    num_layers = st.slider("Số lớp ẩn:", 1, 10, 2, key="train_num_layers")
    num_neurons = st.slider("Số neuron mỗi lớp:", 32, 512, 128, 32, key="train_num_neurons")
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"], key="train_activation")
    optimizer = st.selectbox("Optimizer:", ["adam", "sgd", "rmsprop"], key="train_optimizer")
    epochs = st.slider("🕰 Số epochs:", min_value=1, max_value=50, value=20, step=1, key="train_epochs")
    learning_rate = st.number_input(
        "⚡ Tốc độ học (Learning Rate):", 
        min_value=1e-5, 
        max_value=1e-1, 
        value=1e-3, 
        step=1e-5, 
        format="%.5f", 
        key="train_learning_rate"
    )
    threshold = st.slider("Ngưỡng quyết định (Threshold):", 0.5, 1.0, 0.95, 0.01, key="pseudo_threshold")
    
    # Tùy chọn điều kiện dừng
    stop_condition = st.selectbox(
        "Điều kiện dừng quá trình lặp:",
        ["Lặp theo số bước cố định", "Lặp cho đến khi gán hết nhãn"],
        key="stop_condition"
    )
    if stop_condition == "Lặp theo số bước cố định":
        max_iterations = st.slider("Số vòng lặp tối đa:", 1, 20, 5, key="pseudo_iterations")
    else:
        max_iterations = float('inf')
    
    run_name = st.text_input("🔹 Nhập tên Run:", "", key="train_run_name")
    st.session_state["run_name"] = run_name if run_name else "Default_NN_Pseudo_Run"

    # Khởi tạo biến lưu kết quả và thông tin trong session_state
    if "training_results" not in st.session_state:
        st.session_state.training_results = None
    if "test_accuracy" not in st.session_state:
        st.session_state.test_accuracy = None
    if "pseudo_data" not in st.session_state:
        st.session_state.pseudo_data = []  # Lưu thông tin: ảnh, số lượng, val_accuracy theo vòng
    if "test_images" not in st.session_state:
        st.session_state.test_images = None

    if st.button("Huấn luyện mô hình", key="train_button"):
        if not run_name:
            st.error("⚠️ Vui lòng nhập tên Run trước khi huấn luyện!")
            return
        
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}") as run:
            st.write(f"Debug: Run Name trong MLflow: {run.info.run_name}")

            # Log các tham số
            mlflow.log_param("total_samples", st.session_state.total_samples)
            mlflow.log_param("test_size", st.session_state.test_size)
            mlflow.log_param("validation_size", st.session_state.val_size)
            mlflow.log_param("train_size", st.session_state.train_size)
            mlflow.log_param("num_layers", num_layers)
            mlflow.log_param("num_neurons", num_neurons)
            mlflow.log_param("activation", activation)
            mlflow.log_param("optimizer", optimizer)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("threshold", threshold)
            mlflow.log_param("stop_condition", stop_condition)
            if stop_condition == "Lặp theo số bước cố định":
                mlflow.log_param("max_iterations", max_iterations)

            st.write("⏳ Đang huấn luyện mô hình với Pseudo Labelling...")
            progress_bar = st.progress(0)

            try:
                vong_lap = 0
                st.session_state.pseudo_data = []  # Reset danh sách thông tin vòng lặp
                while vong_lap < max_iterations and len(X_unlabeled) > 0:
                    st.write(f"🔄 Vòng lặp {vong_lap + 1}")

                    # Bước 2: Huấn luyện mô hình trên tập labeled
                    model = Sequential()
                    model.add(Input(shape=(X_labeled.shape[1],)))
                    for _ in range(num_layers):
                        model.add(Dense(num_neurons, activation=activation))
                    model.add(Dense(10, activation="softmax"))

                    if optimizer == "adam":
                        opt = Adam(learning_rate=learning_rate)
                    elif optimizer == "sgd":
                        opt = SGD(learning_rate=learning_rate)
                    else:
                        opt = RMSprop(learning_rate=learning_rate)

                    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                    model.fit(X_labeled, y_labeled, epochs=epochs, batch_size=32, verbose=0)

                    # Đánh giá trên tập validation
                    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                    st.write(f"📌 Độ chính xác Validation tại vòng lặp {vong_lap + 1}: {val_accuracy:.4f}")

                    # Bước 3: Dự đoán nhãn cho tập unlabeled
                    probabilities = model.predict(X_unlabeled, verbose=0)
                    pseudo_labels = np.argmax(probabilities, axis=1)
                    confidence_scores = np.max(probabilities, axis=1)

                    # Bước 4: Gán Pseudo Label dựa trên ngưỡng
                    high_confidence_mask = confidence_scores >= threshold
                    X_pseudo = X_unlabeled[high_confidence_mask]
                    y_pseudo = pseudo_labels[high_confidence_mask]

                    if len(X_pseudo) > 0:
                        st.write(f"✅ Gán nhãn giả cho {len(X_pseudo)} mẫu với độ tin cậy >= {threshold}")
                        X_labeled = np.vstack((X_labeled, X_pseudo))
                        y_labeled = np.hstack((y_labeled, y_pseudo))
                        X_unlabeled = X_unlabeled[~high_confidence_mask]
                        y_unlabeled = y_unlabeled[~high_confidence_mask]
                        mlflow.log_param(f"new_labeled_samples_vong_lap_{vong_lap + 1}", len(X_pseudo))

                        # Lưu thông tin số lượng
                        labeled_count = len(X_labeled)
                        unlabeled_count = len(X_unlabeled)
                        st.write(f"📊 Số ảnh đã gán nhãn: {labeled_count}")
                        st.write(f"📊 Số ảnh chưa gán nhãn: {unlabeled_count}")

                        # Hiển thị 10 ảnh ví dụ trong quá trình huấn luyện
                        st.subheader(f"Ví dụ 10 ảnh vừa được gán nhãn giả (Vòng {vong_lap + 1}):")
                        num_examples = min(10, len(X_pseudo))
                        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                        example_indices = np.random.choice(len(X_pseudo), num_examples, replace=False)
                        for i, idx in enumerate(example_indices):
                            row, col = divmod(i, 5)
                            axes[row, col].imshow(X_pseudo[idx].reshape(28, 28), cmap='gray')
                            axes[row, col].set_title(f"Nhãn giả: {y_pseudo[idx]}")
                            axes[row, col].axis('off')
                        plt.tight_layout()
                        st.pyplot(fig)

                        # Lưu thông tin vòng lặp (bao gồm dữ liệu để hiển thị lại 5 ảnh sau)
                        st.session_state.pseudo_data.append({
                            "vong_lap": vong_lap + 1,
                            "X_pseudo": X_pseudo,  # Lưu dữ liệu ảnh để tái tạo
                            "y_pseudo": y_pseudo,  # Lưu nhãn giả để tái tạo
                            "labeled_count": labeled_count,
                            "unlabeled_count": unlabeled_count,
                            "val_accuracy": val_accuracy
                        })

                    else:
                        st.write("⚠ Không có mẫu nào vượt ngưỡng, dừng lại.")
                        break

                    vong_lap += 1
                    if stop_condition == "Lặp theo số bước cố định":
                        progress_bar.progress(vong_lap / max_iterations)
                    else:
                        progress_bar.progress(len(X_labeled) / st.session_state.train_size)

                # Đánh giá cuối cùng trên tập Test và hiển thị kết quả
                st.subheader("📊 Kết quả cuối cùng trên tập Test")
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                st.session_state.test_accuracy = test_accuracy
                st.write(f"✅ Độ chính xác trên tập Test: {test_accuracy:.4f}")

                # Hiển thị 10 ảnh ví dụ từ tập Test
                st.subheader("Ví dụ 10 ảnh dự đoán trên tập Test:")
                test_predictions = model.predict(X_test, verbose=0)
                test_predicted_labels = np.argmax(test_predictions, axis=1)
                num_examples = 10
                example_indices = np.random.choice(len(X_test), num_examples, replace=False)
                fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                for i, idx in enumerate(example_indices):
                    row, col = divmod(i, 5)
                    axes[row, col].imshow(X_test[idx].reshape(28, 28), cmap='gray')
                    axes[row, col].set_title(f"Thực tế: {y_test[idx]}\nDự đoán: {test_predicted_labels[idx]}")
                    axes[row, col].axis('off')
                plt.tight_layout()
                st.pyplot(fig)
                st.session_state.test_images = fig

                # Log metrics vào MLflow
                mlflow.log_metric("final_val_accuracy", val_accuracy)
                mlflow.log_metric("final_test_accuracy", test_accuracy)
                mlflow.log_metric("final_test_loss", test_loss)

                # Ghi thẳng vào MLflow với signature
                input_example = X_test[:1]
                output_example = model.predict(input_example)
                signature = infer_signature(input_example, output_example)
                mlflow.keras.log_model(model, "Neural_Network_Pseudo_Labelling", signature=signature)

                st.session_state.training_results = {
                    "final_val_accuracy": val_accuracy,
                    "test_accuracy": test_accuracy,
                    "test_loss": test_loss,
                    "run_name": f"Train_{st.session_state['run_name']}",
                    "status": "success"
                }

                # Lưu mô hình vào danh sách models
                if "models" not in st.session_state:
                    st.session_state["models"] = []
                model_name = "Neural_Network_Pseudo_Labelling"
                full_run_name = f"Train_{st.session_state['run_name']}"
                existing_model = next((item for item in st.session_state["models"] if item["name"] == model_name), None)
                if existing_model:
                    count = 1
                    new_model_name = f"{model_name}_{count}"
                    while any(item["name"] == new_model_name for item in st.session_state["models"]):
                        count += 1
                        new_model_name = f"{model_name}_{count}"
                    model_name = new_model_name
                    st.warning(f"⚠️ Mô hình được lưu với tên: {model_name}")
                st.session_state["models"].append({
                    "name": model_name,
                    "run_name": full_run_name,
                    "model": model
                })

            except Exception as e:
                error_message = str(e)
                mlflow.log_param("status", "failed")
                mlflow.log_metric("test_accuracy", -1)
                mlflow.log_metric("test_loss", -1)
                mlflow.log_param("error_message", error_message)
                st.session_state.training_results = {
                    "error_message": error_message,
                    "run_name": f"Train_{st.session_state['run_name']}",
                    "status": "failed"
                }

    # Hiển thị lại kết quả và thông tin từ session_state
    if st.session_state.training_results:
        st.subheader("📊 Kết quả huấn luyện")
        if st.session_state.training_results["status"] == "success":
            st.write(f"✅ Độ chính xác Validation cuối cùng: {st.session_state.training_results['final_val_accuracy']:.4f}")
            st.write(f"✅ Độ chính xác Test cuối cùng: {st.session_state.test_accuracy:.4f}")
            st.success(f"✅ Đã log dữ liệu cho **{st.session_state.training_results['run_name']}**!")

            # Hiển thị lại thông tin và 5 ảnh ví dụ của từng vòng lặp
            if st.session_state.pseudo_data:
                st.subheader("📸 Thông tin và ảnh nhãn giả từ các vòng lặp")
                for data in st.session_state.pseudo_data:
                    st.write(f"**Vòng lặp {data['vong_lap']}**")
                    st.write(f"📌 Độ chính xác Validation: {data['val_accuracy']:.4f}")
                    st.write(f"📊 Số ảnh đã gán nhãn: {data['labeled_count']}")
                    st.write(f"📊 Số ảnh chưa gán nhãn: {data['unlabeled_count']}")

                    # Hiển thị lại 5 ảnh ví dụ
                    X_pseudo = data['X_pseudo']
                    y_pseudo = data['y_pseudo']
                    num_examples = min(5, len(X_pseudo))  # Chỉ lấy 5 ảnh
                    fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # 1 hàng, 5 cột
                    example_indices = np.random.choice(len(X_pseudo), num_examples, replace=False)
                    for i, idx in enumerate(example_indices):
                        axes[i].imshow(X_pseudo[idx].reshape(28, 28), cmap='gray')
                        axes[i].set_title(f"Nhãn giả: {y_pseudo[idx]}")
                        axes[i].axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)

            # Hiển thị lại ảnh tập test
            if st.session_state.test_images:
                st.subheader("📸 Ví dụ 10 ảnh dự đoán trên tập Test")
                st.pyplot(st.session_state.test_images)

        else:
            st.error(f"❌ Lỗi khi huấn luyện mô hình: {st.session_state.training_results['error_message']}")

    # Hiển thị danh sách mô hình đã lưu
    if "models" in st.session_state and st.session_state["models"]:
        st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")
        st.write("📋 Danh sách các mô hình đã lưu:")
        model_display = [f"{model['run_name']} ({model['name']})" for model in st.session_state["models"]]
        st.write(", ".join(model_display))
def predict():
    st.header("🔍 Dự đoán với Neural Network")
    if "models" not in st.session_state or not st.session_state["models"]:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện! Hãy huấn luyện mô hình trước trong tab 'Huấn luyện'.")
        return

    # Chọn mô hình từ danh sách đã huấn luyện
    model_options = [f"{model['run_name']} ({model['name']})" for model in st.session_state["models"]]
    selected_model_name = st.selectbox("Chọn mô hình để dự đoán:", model_options, key="predict_model_select")
    selected_model = next(model for model in st.session_state["models"] if f"{model['run_name']} ({model['name']})" == selected_model_name)
    model = selected_model["model"]

    # Chọn phương thức dự đoán (xóa "Dự đoán trên tập Test")
    prediction_mode = st.radio(
        "Chọn phương thức dự đoán:",
        ["Vẽ tay", "Tải ảnh lên"],
        key="predict_mode"
    )

    if prediction_mode == "Vẽ tay":
        st.write("Vẽ số từ 0-9 trên canvas bên dưới (kích thước 28x28):")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("Dự đoán số vẽ tay", key="predict_draw_button"):
            if canvas_result.image_data is not None:
                # Chuyển ảnh vẽ tay thành định dạng phù hợp (28x28, grayscale)
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(img) / 255.0  # Chuẩn hóa về [0, 1]
                img_array = img_array.reshape(1, 28 * 28)  # Reshape thành (1, 784)

                # Dự đoán
                prediction = model.predict(img_array, verbose=0)
                predicted_label = np.argmax(prediction)
                confidence = np.max(prediction)

                st.write(f"✅ Dự đoán: **{predicted_label}** (Độ tin cậy: {confidence:.4f})")
                st.image(img, caption="Ảnh vẽ tay đã xử lý (28x28)", width=100)
            else:
                st.warning("⚠️ Vui lòng vẽ một số trước khi dự đoán!")

    elif prediction_mode == "Tải ảnh lên":
        uploaded_file = st.file_uploader("Tải lên ảnh số (28x28, grayscale):", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # Xử lý ảnh tải lên
            img = Image.open(uploaded_file).convert('L')  # Chuyển thành grayscale
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0  # Chuẩn hóa về [0, 1]
            img_array = img_array.reshape(1, 28 * 28)  # Reshape thành (1, 784)

            if st.button("Dự đoán ảnh tải lên", key="predict_upload_button"):
                # Dự đoán
                prediction = model.predict(img_array, verbose=0)
                predicted_label = np.argmax(prediction)
                confidence = np.max(prediction)

                st.write(f"✅ Dự đoán: **{predicted_label}** (Độ tin cậy: {confidence:.4f})")
                st.image(img, caption="Ảnh tải lên đã xử lý (28x28)", width=100)

    # Gọi show_experiment_selector() sau khi dự đoán
    if st.session_state.get("training_results") or st.session_state.get("models"):
        show_experiment_selector()
from datetime import datetime

def show_experiment_selector(context="mlflow"):
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'> MLflow Experiments </h1>", unsafe_allow_html=True)
    if 'mlflow_url' in st.session_state:
        st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
    else:
        st.warning("⚠️ URL MLflow chưa được khởi tạo!")

    with st.sidebar:
        st.subheader("🔍 Tổng quan Experiment")
        experiment_name = "Neural_Network_Pseudo_Labelling"  # Tên experiment từ mlflow_input()
        
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
def main():
    st.title("Neural Network với Pseudo Labelling trên MNIST")
    mlflow_input()
    tabs = st.tabs(["Dữ Liệu", "Lý thuyết", "Chia dữ liệu", "Huấn luyện", "Dự đoán"])
    with tabs[0]:
        data()
    with tabs[1]:
        explain_pseudo_labeling()
    with tabs[2]:
        split_data()
    with tabs[3]:
        train()
    with tabs[4]:
        predict()

if __name__ == "__main__":
    main()