import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

DATASET_PATH = "RawDataset"

# ====== Khởi tạo các model ======
detector = MTCNN()
embedder = FaceNet()
classifier = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', max_iter=500, early_stopping=True, random_state=42)

# ====== Hàm xử lý ảnh để sinh embedding ======
def extract_face(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return None
    h_img, w_img, _ = img.shape
    x, y, w, h = faces[0]['box']
    x, y = max(x, 0), max(y, 0)
    x2, y2 = min(x+w, w_img), min(y+h, h_img)
    face = img[y:y2, x:x2]
    face = cv2.resize(face, (160, 160))
    return face

# ====== Load dữ liệu ======
def load_dataset(dataset_path):
    X, y = [], []
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            face = extract_face(img_path)
            if face is not None:
                X.append(face)
                y.append(person_name)
    return np.array(X), np.array(y)

# ====== Kiểm tra đã có embeddings chưa ======
if os.path.exists("embeddings.npy") and os.path.exists("labels.npy"):
    print("Loading existing embeddings...")
    embeddings = np.load("embeddings.npy")
    y = np.load("labels.npy")
else:
    print("Loading dataset and extracting faces...")
    X, y = load_dataset("DATASET_PATH")
    print("Embedding faces...")
    embeddings = embedder.embeddings(X)
    np.save("embeddings.npy", embeddings)
    np.save("labels.npy", y)

# ====== Encode label ======
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ====== Chia dữ liệu thành train, val, test ======
# Bước 1: train_val (80%) và test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    embeddings, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Bước 2: train (64%) và val (16%) từ train_val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# ====== Huấn luyện ======
print("Training classifier...")
classifier.fit(X_train, y_train)

# ====== Đánh giá mô hình =======
# ====== Đánh giá trên validation set ======
print("Validation performance:")
y_val_pred = classifier.predict(X_val)
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# ====== Đánh giá trên test set ======
print("Test performance:")
y_test_pred = classifier.predict(X_test)
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

#------------------------
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("Validation Accuracy:", val_acc)
print("Test Accuracy:", test_acc)

print("Final training loss:", classifier.loss_)

print("Validation F1 (macro):", f1_score(y_val, y_val_pred, average='macro'))
print("Validation F1 (weighted):", f1_score(y_val, y_val_pred, average='weighted'))
print("Test F1 (macro):", f1_score(y_test, y_test_pred, average='macro'))
print("Test F1 (weighted):", f1_score(y_test, y_test_pred, average='weighted'))

# ====== Lưu mô hình và encoder ======
joblib.dump(classifier, "face_classifier.joblib")
joblib.dump(le, "label_encoder.joblib")
