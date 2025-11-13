from flask import Flask, request, jsonify
import cv2, numpy as np, threading, time, traceback
from inference5 import identify, sheet_result
#import os

#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
#os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

app = Flask(__name__)

# Giới hạn số task xử lý
MAX_ACTIVE_TASKS = 5
active_tasks = 0
lock = threading.Lock() # sử dụng lock

# API ca làm thêm
@app.route('/api/infer_main', methods=['POST'])
def infer_main():
    return handle_request("main")

# API ca làm phụ
@app.route('/api/infer_extra', methods=['POST'])
def infer_extra():
    return handle_request("extra")

def handle_request(mode):
    """Xử lý ảnh đồng bộ, trả kết quả ngay, nhưng vẫn có giới hạn"""
    global active_tasks
    start_time = time.time()
    try:
        with lock:
            if active_tasks >= MAX_ACTIVE_TASKS:
                print(f"[{mode}]  Tu choi task moi (server busy)")
                return jsonify({"status": "busy", "message": "Server busy, please restart WEB"}), 503
            active_tasks += 1
            print(f"[{mode}]  Nhan task moi. Active: {active_tasks}")

        if 'image' not in request.files:
            raise ValueError("Khong có file 'image' trong request")

        file = request.files['image']
        filename = file.filename
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Khong đoc đuoc anh, đinh dang khong hop le")

        # Gọi AI
        print(f"[{mode}]  Bat đau xu ly anh: {filename}")
        result = identify(img, filename)
        end_time = time.time()
        print(f"[{mode}]  AI hoan tat trong {end_time - start_time:.2f} giay")
        
        try:
            from upload_sheet import NAME_MAP
            predicted = result.get("predicted_label")
            if predicted:
                result["mapped_name"] = NAME_MAP.get(predicted, predicted)
            else:
                result["mapped_name"] = "Unknow"
        except Exception as e:
            print(f"Lỗi mapped_name: {e}")
            reuslt["maooed_name"] = "Unkown"

        # Ghi Google Sheet ở background 
        try:
            threading.Thread(target=sheet_result, args=(result, mode), daemon=True).start()
            print(f"[{mode}]  Đã khởi tạo thread ghi Google Sheet cho {filename}")
        except Exception as e:
            print(f"[{mode}]  Lỗi khi khởi tạo thread sheet_result: {e}")

        # Trả kết quả đầy đủ cho client 
        return jsonify({
            "mode": mode,
            "status": "success",
            "filename": filename,
            "processing_time": round(end_time - start_time, 2),
            "data": result
        })

    except Exception as e:
        traceback.print_exc()
        with lock:
            active_tasks = max(active_tasks - 1, 0)
        return jsonify({"error": str(e)}), 500

    finally:
        with lock:
            active_tasks -= 1
        print(f"[{mode}]  Giải phóng task. Active: {active_tasks}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
