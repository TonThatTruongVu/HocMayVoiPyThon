import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import joblib
import pandas as pd
import os
import mlflow
from datetime import datetime


# Load dữ liệu MNIST
def load_mnist_data():
    X = np.load("Buoi4/X.npy")
    y = np.load("Buoi4/y.npy")
    return X, y


def data():
    st.header("📘 Dữ Liệu MNIST")
    X, y = load_mnist_data()
    
    st.write("""
        **Thông tin tập dữ liệu MNIST:**
        - Tổng số mẫu: {}
        - Kích thước mỗi ảnh: 28 × 28 pixels (784 đặc trưng)
        - Số lớp: 10 (chữ số từ 0-9)
    """.format(X.shape[0]))

    # Display sample images
    st.subheader("Một số hình ảnh mẫu")
    n_samples = 5
    fig, axes = plt.subplots(1, n_samples, figsize=(10, 3))
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    for i, idx in enumerate(indices):
        axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Label: {y[idx]}")
        axes[i].axis("off")
    st.pyplot(fig)


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
        
        X_selected, _, y_selected, _ = train_test_split(X, y, train_size=num_samples, stratify=y, random_state=42)
        stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42)
        stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size/(100-test_size), stratify=stratify_option, random_state=42)

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

        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, không cần chạy lại.")


def train_evaluate():
    st.header("⚙️ Chọn mô hình & Huấn luyện")
    
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    X_train = st.session_state.X_train.reshape(-1, 28 * 28) / 255.0
    X_test = st.session_state.X_test.reshape(-1, 28 * 28) / 255.0
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])
    
    if model_choice == "Decision Tree":
        st.markdown("""
        - **🌳 Decision Tree (Cây quyết định)** giúp chia dữ liệu thành các nhóm bằng cách đặt câu hỏi nhị phân dựa trên đặc trưng.
        - **Tham số cần chọn:**  
            - **max_depth**: Giới hạn độ sâu tối đa của cây.
            - **Chọn số folds (KFold Cross-Validation)**: Đây là số lần chia dữ liệu huấn luyện để đánh giá mô hình trước khi huấn luyện chính thức.  
        """)
        max_depth = st.slider("max_depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth)
        params = {"max_depth": max_depth}
    elif model_choice == "SVM":
        st.markdown("""
        - **🛠️ SVM (Support Vector Machine)** là mô hình phân loại mạnh mẽ, hoạt động bằng cách tìm một **siêu phẳng (hyperplane)** tối ưu trong không gian dữ liệu để phân tách các lớp (class) với **lề phân tách (margin)** lớn nhất có thể. Khi dữ liệu không thể phân tách tuyến tính trong không gian ban đầu, SVM sử dụng **hàm kernel** để biến đổi dữ liệu sang không gian chiều cao hơn, nơi các lớp trở nên phân tách tuyến tính.
        - **Tham số cần chọn:**
            - **C (Regularization)**: Điều chỉnh sự cân bằng giữa việc **tối ưu hóa lề phân tách lớn** và **giảm thiểu lỗi phân loại** trên tập huấn luyện.
                - Giá trị nhỏ (C thấp): Ưu tiên lề phân tách lớn, chấp nhận một số điểm dữ liệu bị phân loại sai (mô hình đơn giản hơn, ít overfitting).
                - Giá trị lớn (C cao): Ưu tiên phân loại chính xác tất cả các điểm dữ liệu, có thể dẫn đến lề nhỏ hơn và nguy cơ overfitting cao hơn.
            - **Kernel**: Hàm biến đổi dữ liệu sang không gian mới để dễ phân tách hơn. Các loại kernel bao gồm:
                - **Linear**: Tạo siêu phẳng tuyến tính trong không gian ban đầu, phù hợp khi dữ liệu có thể phân tách tuyến tính mà không cần biến đổi phức tạp. Hiệu quả với dữ liệu đơn giản, ít chiều.
                - **Poly (Polynomial)**: Biến đổi dữ liệu bằng hàm đa thức, phù hợp khi mối quan hệ giữa các đặc trưng có dạng phi tuyến phức tạp. Độ bậc (degree) càng cao, mô hình càng phức tạp.
                - **RBF (Radial Basis Function)**: Dựa trên khoảng cách Gaussian, đo lường sự giống nhau giữa các điểm dữ liệu. Đây là lựa chọn mặc định phổ biến, rất hiệu quả với dữ liệu phi tuyến phức tạp như MNIST, vì nó linh hoạt và không cần giả định dạng phân phối cụ thể.
                - **Sigmoid**: Dựa trên hàm sigmoid (giống mạng nơ-ron), biến đổi dữ liệu theo cách phi tuyến, ít được dùng hơn nhưng có thể hiệu quả trong một số trường hợp đặc biệt, đặc biệt khi dữ liệu có đặc điểm tương tự đầu ra của mạng nơ-ron.
        """)
        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        if kernel == "poly":
            degree = st.slider("Độ bậc đa thức (degree)", 2, 5, 3)
            model = SVC(C=C, kernel=kernel, degree=degree, probability=True)  # Thêm probability=True
            params = {"C": C, "kernel": kernel, "degree": degree}
        else:
            model = SVC(C=C, kernel=kernel, probability=True)  # Thêm probability=True
            params = {"C": C, "kernel": kernel}

    n_folds = st.slider("Chọn số folds (KFold Cross-Validation):", min_value=2, max_value=10, value=5)
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")
    st.session_state["run_name"] = run_name if run_name else "default_run"
    
    if st.button("Huấn luyện mô hình"):
        st.write(f"⏳ Mô hình '{model_choice}' đang được huấn luyện...")
        with st.spinner("Đang huấn luyện mô hình, vui lòng đợi..."):
            with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}") as run:
                run_id = run.info.run_id
                mlflow.log_params({"model": model_choice, **params})
                if "train_size" in st.session_state:
                    mlflow.log_param("train_size", st.session_state.train_size)
                if "val_size" in st.session_state:
                    mlflow.log_param("val_size", st.session_state.val_size)
                if "test_size" in st.session_state:
                    mlflow.log_param("test_size", st.session_state.test_size)
                if "total_samples" in st.session_state:
                    mlflow.log_param("num_samples", st.session_state.total_samples)

                os.makedirs("mlflow_artifacts", exist_ok=True)
                np.save("mlflow_artifacts/X_train.npy", X_train)
                np.save("mlflow_artifacts/X_test.npy", X_test)
                np.save("mlflow_artifacts/y_train.npy", y_train)
                np.save("mlflow_artifacts/y_test.npy", y_test)
                mlflow.log_artifacts("mlflow_artifacts")

                st.write("⏳ Đang chạy Cross-Validation...")
                cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds)
                mean_cv_score = cv_scores.mean()
                std_cv_score = cv_scores.std()
                st.success(f"📊 **Cross-Validation Accuracy**: {mean_cv_score:.4f}")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ Độ chính xác trên test set: {acc:.4f}")

                mlflow.log_metric("test_accuracy", acc)
                mlflow.log_metric("cv_accuracy_mean", mean_cv_score)
                mlflow.log_metric("cv_accuracy_std", std_cv_score)
                mlflow.sklearn.log_model(model, model_choice.lower())

        if "models" not in st.session_state:
            st.session_state["models"] = []

        model_name = model_choice.lower().replace(" ", "_")
        if model_choice == "SVM":
            model_name += f"_{kernel}"
        existing_model = next((item for item in st.session_state["models"] if item["name"] == model_name), None)
        if existing_model:
            count = 1
            new_model_name = f"{model_name}_{count}"
            while any(item["name"] == new_model_name for item in st.session_state["models"]):
                count += 1
                new_model_name = f"{model_name}_{count}"
            model_name = new_model_name
            st.warning(f"⚠️ Mô hình được lưu với tên: {model_name}")

        st.session_state["models"].append({"name": model_name, "model": model})
        st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
        st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")

        st.write("📋 Danh sách các mô hình đã lưu:")
        model_names = [model["name"] for model in st.session_state["models"]]
        st.write(", ".join(model_names))

        st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}**!")
        mlflow_tracking_uri = "https://dagshub.com/TonThatTruongVu/MNIST-Classification.mlflow"
        experiment_id = mlflow.get_experiment_by_name("Classification").experiment_id
        mlflow_link = f"{mlflow_tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
        st.markdown(f"🔗 [Truy cập MLflow UI]({mlflow_link})")


def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"⚠️ Không tìm thấy mô hình tại `{path}`")
        st.stop()


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


import random
import random

def demo():
    st.header("✍️ Vẽ số hoặc tải ảnh để dự đoán")
    
    # Kiểm tra xem có mô hình nào đã huấn luyện không
    if "models" not in st.session_state or not st.session_state["models"]:
        st.error("⚠️ Mô hình chưa được huấn luyện! Vui lòng huấn luyện mô hình trong tab 'Huấn luyện' trước.")
        return

    # Lấy danh sách mô hình đã huấn luyện
    model_names = [model["name"] for model in st.session_state.get("models", [])]
    model_option = st.selectbox("🔍 Chọn mô hình:", model_names)
    model = next(model["model"] for model in st.session_state["models"] if model["name"] == model_option)

    # Lựa chọn giữa vẽ số và tải ảnh
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
            height=300,  # Tăng chiều cao từ 150 lên 300
            width=300,   # Tăng chiều rộng từ 150 lên 300
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

    if st.button("Dự đoán số"):
        if input_data is not None:
            st.image(Image.fromarray((input_data.reshape(28, 28) * 255).astype(np.uint8)), caption=f"Ảnh xử lý từ {source}", width=100)
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = probabilities[int(prediction)] * 100
            st.subheader(f"🔢 Dự đoán: {prediction}")
            st.write(f"📈 **Độ tin cậy**: {confidence:.2f}%")
            st.write("---")
            show_experiment_selector()
        else:
            st.error(f"⚠️ Hãy {'vẽ một số' if input_method == 'Vẽ số' else 'tải ảnh'} trước khi dự đoán!")

# Các hàm phụ trợ
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

# Các hàm phụ trợ (đảm bảo đã định nghĩa)
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


def show_experiment_selector():
    st.title("📊 MLflow Experiments - DAGsHub")
    experiment_name = "Classification"
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


def main():
    st.title("🖊️ MNIST Classification with Streamlit & MLflow")
    
    if "mlflow_initialized" not in st.session_state:
        mlflow.set_tracking_uri("https://dagshub.com/TonThatTruongVu/MNIST-Classification.mlflow")
        os.environ["MLFLOW_TRACKING_USERNAME"] = "TonThatTruongVu"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "519c4a864e131de52197f54d170c130beb15ffd5"
        mlflow.set_experiment("Classification")
        st.session_state.mlflow_url = "https://dagshub.com/TonThatTruongVu/MNIST-Classification.mlflow"
        st.session_state.mlflow_initialized = True

    tabs = st.tabs(["📘 Data", "📌 Chia dữ liệu", "⚙️ Huấn luyện", "🔢 Dự đoán"])
    
    with tabs[0]:
        data()
    with tabs[1]:
        split_data()
    with tabs[2]:
        train_evaluate()
    with tabs[3]:
        demo()
        


if __name__ == "__main__":
    main()