import os
import cv2
import joblib
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from upload_sheet2 import handle_result
import torch

# Cấu hình
MISCLASSIFIED_FOLDER = "misclassified"
THRESHOLD = 0.7  # ngưỡng cosine similarity
os.makedirs(MISCLASSIFIED_FOLDER, exist_ok=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "model")

# Chọn device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available())  # True/False
print("Current device:", torch.cuda.current_device())  # index GPU hiện tại
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Load mô hình MTCNN và FaceNet (PyTorch)
face_detector = MTCNN(keep_all=False, device=device)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load dữ liệu embeddings và labels
known_embeddings = joblib.load(os.path.join(MODEL_DIR, "known_embeddings.joblib"))
known_labels = joblib.load(os.path.join(MODEL_DIR, "known_labels.joblib"))

# Hàm nhận diện ảnh
def identify(img_array, filename="image.jpg"):
    # Chuyển BGR -> RGB
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Detect face (MTCNN trả về tensor C,H,W hoặc None)
    face_tensor = face_detector(img_rgb)
    
    # Nếu không detect được face, return luôn
    if face_tensor is None:
        return {"status": "fail", "message": "Không phát hiện khuôn mặt"}
    
    # Nếu MTCNN trả về nhiều face (list), lấy đầu tiên
    if isinstance(face_tensor, list):
        if len(face_tensor) == 0:
            return {"status": "fail", "message": "Không phát hiện khuôn mặt"}
        face_tensor = face_tensor[0]

    # Resize face về 160x160
    if face_tensor.shape[-2:] != (160, 160):
        face_tensor = torch.nn.functional.interpolate(
            face_tensor.unsqueeze(0), size=(160, 160), mode='bilinear'
        ).squeeze(0)

    # Tính embedding
    with torch.no_grad():
        face_tensor = face_tensor.to(device)
        embedding = embedder(face_tensor.unsqueeze(0)).cpu().numpy()[0]

        # Ép sang float và loại bỏ NaN trước khi wrap json
        embedding = np.nan_to_num(np.array(embedding, dtype=float))

    # Cosine similarity
    sims = cosine_similarity([embedding], known_embeddings)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    predicted_label = known_labels[best_idx] if best_score >= THRESHOLD else "Unknown"

    # Lưu ảnh Unknown
    if predicted_label == "Unknown":
        save_path = os.path.join(MISCLASSIFIED_FOLDER, f"Unknown_{filename}")
        cv2.imwrite(save_path, img_array)

    return {"status": "ok", "filename": filename, "predicted_label": predicted_label, "score": float(best_score)}

# Wrapper xử lý ảnh
def process_face(img_array, filename="image.jpg"):
    result = identify(img_array, filename)
    if result["status"] == "ok":
        label = result["predicted_label"]
        score = result["score"]
        print(f"Nhận diện: {label} (score={score:.2f})")
    return result

# Ghi kết quả lên sheet
def sheet_result(result, mode, location="D6 41"):
    if result.get("status") == "ok" and result.get("predicted_label") != "Unknown":
        handle_result(result["predicted_label"], status="2", location=location, mode=mode)
    return result
