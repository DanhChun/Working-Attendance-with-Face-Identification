import requests
from datetime import datetime
import json
import numpy as np


# Map nhãn danh sách → họ tên nhân viên
NAME_MAP = {
    "anh_Danh": "Chu Cao Danh",
}
PERSON_TO_FILE = {
    "anh_Danh": {
        "file_id": "1348RG_UwjiaeR-cx6fcuz2Xm3acM8i7A4EDfRizU0Hs",
        "file_name": "DanhCC",
        "position": "",
        "department": "",
        "apps_script_url": "https://script.google.com/macros/s/AKfycbywPuqPO68Twhv-rpp8Gv4J4XxKOQEnZdnRC44SyL70Lxh5GydxHJMwbX-wTsOnQ6xJ/exec"
    }
}

def handle_result(name, status, location, mode):
    routed_name = None
    for key, val in NAME_MAP.items():
        if val == name:
            routed_name = key
            break

    if not routed_name or routed_name not in PERSON_TO_FILE:
        print(f"⚠️ Không map được {name}")
        return

    sheet_info = PERSON_TO_FILE[routed_name] 
    data = {
        "name": name,
        "position": sheet_info.get("position", "UNKNOWN"),
        "status": status,
        "location": location,
        "file_id": sheet_info["file_id"],
        "file_name": sheet_info.get("file_name", ""),
        "department": sheet_info.get("department"),
        "mode": mode
    }

    def convert_data(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    clean_data = {k: convert_data(v) for k, v in data.items()}

    # lấy url theo từng người
    apps_url = sheet_info.get("apps_script_url")
    if not apps_url:
        print(f"Không có apps script của {name}")
        return
        
    try:
        r = requests.post(apps_url,
                         json=clean_data,
                         headers={"Content-Type": "application/json"},
                         timeout=20)
        print(f"Gửi Apps Script: {r.status_code}, Trả về: {r.text}")

    except requests.RequestException as e:
        print(f"lỗi network khi gọi GAS: {e}")
        raise
    # except Exception as e:
    #     print(f" Lỗi xử lý: {e}")
    #     raise
