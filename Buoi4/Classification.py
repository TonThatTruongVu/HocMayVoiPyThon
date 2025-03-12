import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image

import os
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import cross_val_score
import random
from datetime import datetime
import matplotlib.pyplot as plt

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/TonThatTruongVu/MNIST-Classification.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "TonThatTruongVu"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "519c4a864e131de52197f54d170c130beb15ffd5"
    mlflow.set_experiment("MNIST_Classification")
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI


def load_mnist_data():
    X = np.load("Buoi4/X.npy")
    y = np.load("Buoi4/y.npy")
    return X, y


def data():
    st.header("📘 Dữ Liệu MNIST")
    try:
        X, y = load_mnist_data()
        
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
    except FileNotFoundError:
        st.error("⚠️ Không tìm thấy file dữ liệu `X.npy` hoặc `y.npy` trong thư mục `buoi4/`!")

def split_data():
    st.title("📌 Chia dữ liệu Train/Test")
    X, y = load_mnist_data() 
    total_samples = X.shape[0]

    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False  

    num_samples = st.slider("📌 Chọn số lượng ảnh để train:", 1000, total_samples, 10000)
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    if st.button("✅ Xác nhận & Lưu") and not st.session_state.data_split_done:
        st.session_state.data_split_done = True
        
        # Step 1: Select a subset of num_samples from the full dataset
        indices = np.random.choice(total_samples, num_samples, replace=False)
        X_selected = X[indices]
        y_selected = y[indices]

        # Step 2: Split the selected subset into train+val and test
        stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
        )

        # Step 3: Split the train_full into train and validation
        stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size / (100 - test_size),
            stratify=stratify_option, random_state=42
        )

        # Store the split data in session state
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
        st.session_state.df = pd.DataFrame(np.hstack((X_selected, y_selected.reshape(-1, 1))), 
                                          columns=[f"pixel_{i}" for i in range(X_selected.shape[1])] + ["label"])
        st.session_state.y = st.session_state.df["label"]

        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, không cần chạy lại.")

def train():
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    X_train = st.session_state.X_train 
    X_val = st.session_state.X_val
    X_test = st.session_state.X_test 
    y_train = st.session_state.y_train 
    y_val = st.session_state.y_val 
    y_test = st.session_state.y_test 

    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    st.header("⚙️ Chọn mô hình & Huấn luyện")
    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])

    if model_choice == "Decision Tree":
        st.markdown("""
        ### 🌳 Decision Tree (Cây Quyết Định)
        - **Decision Tree** giúp chia dữ liệu thành các nhóm bằng cách đặt câu hỏi nhị phân dựa trên đặc trưng.
        - **Tiêu chí chia nhánh**:
          - **Gini Index**: Đo xác suất chọn nhầm nhãn nếu lấy ngẫu nhiên một điểm trong nhóm.
            - Gini = 0: Nhóm chỉ chứa một loại nhãn.
            - Gini cao: Nhóm chứa nhiều nhãn khác nhau.
          - **Entropy**: 
            - **Entropy** đo mức độ hỗn loạn của nhóm, cao khi nhóm chứa nhiều nhãn khác nhau.
        - **Tham số quan trọng**:
          - `max_depth`: Giới hạn độ sâu tối đa của cây để tránh overfitting.
          - `criterion`: Chọn tiêu chí chia nhánh (Gini hoặc Entropy).
        """)
        
        max_depth = st.slider("max_depth (Độ sâu tối đa)", 1, 20, 5)
        criterion = st.selectbox("Criterion (Tiêu chí chia)", ["gini", "entropy"], 
                                 help="Chọn 'gini' để giảm phân loại sai, hoặc 'entropy' để nhóm dễ đoán hơn.")
        
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            random_state=42
        )

    elif model_choice == "SVM":
        st.markdown("""
        ### 🛠️ Support Vector Machine (SVM)
        - **SVM** tìm siêu phẳng tối ưu để phân tách dữ liệu theo cách tốt nhất.
        - **Tham số quan trọng**:
          - **C (Regularization)**: Điều chỉnh mức độ chấp nhận lỗi.
            - C nhỏ: Cho phép một số điểm bị phân loại sai → tránh overfitting.
            - C lớn: Giảm lỗi tối đa nhưng dễ bị overfitting.
          - **Kernel**: Cách ánh xạ dữ liệu để tìm ranh giới phân tách.
        """)

        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])

        st.markdown("""
        **🔍 Giải thích các loại Kernel:**
        - **Linear**: Phân tách dữ liệu bằng một đường thẳng (hoặc siêu phẳng trong không gian cao hơn). Tốt khi dữ liệu có thể phân tách tuyến tính.
        - **RBF (Radial Basis Function)**: Dùng hàm Gaussian để ánh xạ dữ liệu, phù hợp với dữ liệu phi tuyến tính phức tạp. Đây là lựa chọn mặc định phổ biến.
        - **Poly (Polynomial)**: Sử dụng hàm đa thức để ánh xạ dữ liệu, hữu ích khi quan hệ giữa các đặc trưng có dạng đa thức.
        - **Sigmoid**: Dựa trên hàm sigmoid, tương tự mạng nơ-ron, nhưng thường ít hiệu quả hơn RBF.
        """)

        if kernel == "poly":
            degree = st.slider("Degree (Bậc đa thức)", 2, 5, 3)
            model = SVC(C=C, kernel=kernel, degree=degree, random_state=42)
        else:
            model = SVC(C=C, kernel=kernel, random_state=42)

    n_folds = st.slider("Chọn số folds (KFold Cross-Validation):", min_value=2, max_value=10, value=5)
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")
    st.session_state["run_name"] = run_name if run_name else "Default_Run"

    if "training_results" not in st.session_state:
        st.session_state.training_results = None

    if st.button("Huấn luyện mô hình"):
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}") as run:
            st.write(f"Debug: Run Name trong MLflow: {run.info.run_name}")

            df = st.session_state.df
            mlflow.log_param("dataset_shape", df.shape)
            mlflow.log_param("target_column", st.session_state.y.name)
            mlflow.log_param("test_size", st.session_state.test_size)
            mlflow.log_param("validation_size", st.session_state.val_size)
            mlflow.log_param("train_size", st.session_state.train_size)

            dataset_path = "dataset.csv"
            df.to_csv(dataset_path, index=False)
            mlflow.log_artifact(dataset_path)

            mlflow.log_param("model", model_choice)
            if model_choice == "Decision Tree":
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("criterion", criterion)
            elif model_choice == "SVM":
                mlflow.log_param("C", C)
                mlflow.log_param("kernel", kernel)
                if kernel == "poly":
                    mlflow.log_param("degree", degree)
            mlflow.log_param("n_folds", n_folds)

            st.write("⏳ Đang đánh giá và huấn luyện mô hình...")
            progress_bar = st.progress(0)
            total_steps = n_folds + 1
            step_progress = 1.0 / total_steps

            try:
                # Đánh giá bằng Cross-Validation
                st.write("🔍 Đánh giá mô hình qua Cross-Validation...")
                cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds)
                for i in range(n_folds):
                    progress_bar.progress((i + 1) * step_progress)
                    st.write(f"📌 Fold {i + 1} - Accuracy: {cv_scores[i]:.4f}")
                    mlflow.log_metric(f"accuracy_fold_{i+1}", cv_scores[i])

                mean_cv_score = cv_scores.mean()
                std_cv_score = cv_scores.std()

                # Huấn luyện mô hình cuối cùng
                model.fit(X_train, y_train)  # Không cần in thông báo riêng
                progress_bar.progress(1.0)

                y_pred = model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)

   
   
                mlflow.log_metric("cv_accuracy_mean", mean_cv_score)
                mlflow.log_metric("cv_accuracy_std", std_cv_score)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.sklearn.log_model(model, model_choice.lower())

                st.session_state.training_results = {
                    "cv_accuracy_mean": mean_cv_score,
                    "cv_accuracy_std": std_cv_score,
                    "test_accuracy": test_accuracy,
                    "run_name": f"Train_{st.session_state['run_name']}",
                    "status": "success"
                }

            except Exception as e:
                error_message = str(e)
                mlflow.log_param("status", "failed")
                mlflow.log_metric("cv_accuracy_mean", -1)
                mlflow.log_metric("cv_accuracy_std", -1)
                mlflow.log_metric("test_accuracy", -1)
                mlflow.log_param("error_message", error_message)

                st.session_state.training_results = {
                    "error_message": error_message,
                    "run_name": f"Train_{st.session_state['run_name']}",
                    "status": "failed"
                }

            if "models" not in st.session_state:
                st.session_state["models"] = []

            model_name = model_choice.lower().replace(" ", "_")
            if model_choice == "SVM":
                model_name += f"_{kernel}"

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

    if st.session_state.training_results:
        if st.session_state.training_results["status"] == "success":
            st.success(f"📊 Cross-Validation Accuracy trung bình: {st.session_state.training_results['cv_accuracy_mean']:.4f} (±{st.session_state.training_results['cv_accuracy_std']:.4f})")
            st.success(f"✅ Độ chính xác trên test set: {st.session_state.training_results['test_accuracy']:.4f}")
            st.success(f"✅ Đã log dữ liệu cho **{st.session_state.training_results['run_name']}**!")
            st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
        else:
            st.error(f"❌ Lỗi khi huấn luyện mô hình: {st.session_state.training_results['error_message']}")
            st.warning(f"⚠️ Dữ liệu lỗi đã được log vào MLflow với run name: **{st.session_state.training_results['run_name']}**")
            st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")

   
    if "models" in st.session_state and st.session_state["models"]:
        st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")
        st.write("📋 Danh sách các mô hình đã lưu:")
        model_display = [f"{model['run_name']} ({model['name']})" for model in st.session_state["models"]]
        st.write(", ".join(model_display))

def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0
        return img.reshape(1, -1)
    return None


def preprocess_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        img = img.resize((28, 28))
        img = np.array(img, dtype=np.float32) / 255.0
        return img.reshape(1, -1)
    return None

def du_doan():
   
    st.header("✍️ Dự đoán số")

   
    if "models" not in st.session_state or not st.session_state["models"]:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng huấn luyện mô hình trước!")
        return

    model_display = [f"{model['run_name']} ({model['name']})" for model in st.session_state["models"]]
    model_option = st.selectbox("🔍 Chọn mô hình:", model_display)
    selected_model_info = next(model for model in st.session_state["models"] if f"{model['run_name']} ({model['name']})" == model_option)
    model = selected_model_info["model"]
    st.success(f"✅ Đã chọn mô hình: {model_option}")

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
            img = preprocess_canvas_image(canvas_result)
            if img is None:
                st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")
    else:
        uploaded_file = st.file_uploader("📤 Tải ảnh lên (định dạng PNG/JPG, kích thước bất kỳ)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Ảnh đã tải lên", width=150)
            if st.button("Dự đoán số từ ảnh"):
                img = preprocess_uploaded_image(uploaded_file)
                if img is None:
                    st.error("⚠️ Lỗi khi xử lý ảnh tải lên!")

    if img is not None:
        st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)
        
        # Dự đoán nhãn
        prediction = model.predict(img)
        st.subheader(f"🔢 Dự đoán: {prediction[0]}")

        # Lấy độ tin cậy (confidence scores)
        confidence_scores = model.predict_proba(img)[0]  # Lấy xác suất cho tất cả các lớp
        predicted_class_confidence = confidence_scores[prediction[0]]  # Độ tin cậy của nhãn dự đoán
        
        # Hiển thị độ tin cậy của nhãn dự đoán
        st.write(f"📈 **Độ tin cậy:** {predicted_class_confidence:.4f} ({predicted_class_confidence * 100:.2f}%)")

        # Hiển thị tất cả độ tin cậy của các lớp (tùy chọn)
        st.write("**Xác suất cho từng lớp (0-9):**")
        confidence_df = pd.DataFrame({
            "Nhãn": range(10),
            "Xác suất": confidence_scores
        })
        st.bar_chart(confidence_df.set_index("Nhãn"))

        # Hiển thị thông tin MLflow Experiments
        st.write("📊 Hiển thị thông tin MLflow Experiments:")
        show_experiment_selector(context="predict")

def show_experiment_selector(context="mlflow"):
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'> MLflow Experiments </h1>", unsafe_allow_html=True)
    if 'mlflow_url' in st.session_state:
        st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
    else:
        st.warning("⚠️ URL MLflow chưa được khởi tạo!")

    with st.sidebar:
        st.subheader("🔍 Tổng quan Experiment")
        experiment_name = "MNIST_Classification"
        
        experiments = mlflow.search_experiments()
        selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

        if not selected_experiment:
            st.error(f"❌ Không tìm thấy Experiment '{experiment_name}'!", icon="🚫")
            return

        st.markdown(f"**Tên Experiment:** `{experiment_name}`")
        st.markdown(f"**ID:** `{selected_experiment.experiment_id}`")
        st.markdown(f"**Trạng thái:** {'🟢 Active' if selected_experiment.lifecycle_stage == 'active' else '🔴 Deleted'}")
        st.markdown(f"**Artifact Location:** `{selected_experiment.artifact_location}`")

        if "run_name" in st.session_state:
            st.markdown(f"**Run hiện tại:** `{st.session_state['run_name']}`")
        else:
            st.warning("⚠ Chưa có run_name nào được thiết lập.", icon="ℹ️")

    st.markdown("---")
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này!", icon="🚨")
        return

    with st.expander("🏃‍♂️ Danh sách Runs", expanded=True):
        st.write("Chọn một Run để xem chi tiết:")
        run_info = []
        used_names = set()

        model_dict = {model["run_name"]: model["name"] for model in st.session_state.get("models", [])}

        for _, run in runs.iterrows():
            run_id = run["run_id"]
            run_data = mlflow.get_run(run_id)
            run_name = run_data.info.run_name if run_data.info.run_name else f"Run_{run_id[:8]}"
            
            model_name = model_dict.get(run_name, "Unknown Model")
            display_name = f"{run_name} ({model_name})"
            
            run_name_base = display_name
            counter = 1
            while display_name in used_names:
                display_name = f"{run_name_base}_{counter}"
                counter += 1
            used_names.add(display_name)
            run_info.append((display_name, run_id))

        run_name_to_id = dict(run_info)
        run_names = list(run_name_to_id.keys())

        # Sử dụng key khác nhau dựa trên context
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

    else:
        st.warning("⚠ Không tìm thấy thông tin cho Run này!", icon="🚨")

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Powered by Streamlit & MLflow</p>", unsafe_allow_html=True)

def main():
    if "mlflow_initialized" not in st.session_state:
        mlflow_input()
        st.session_state.mlflow_initialized = True
        
    st.title("🖊️ MNIST Classification App")
    
    tab1, tab2, tab3 = st.tabs(["📘 Data", "⚙️ Huấn luyện", "🔢 Dự đoán"])
    
    with tab1:
        data()
        
    with tab2:
        split_data()
        train()
        
    with tab3:
        du_doan()
        

if __name__ == "__main__":
    main()