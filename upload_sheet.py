import requests
from datetime import datetime
import json
import numpy as np

# Map nhãn danh sách → họ tên nhân viên
NAME_MAP = {
    "<file_name_as_well_as_person_name>": "<Full_name>", #  can change
}

PERSON_TO_SHEET = {
    "": {
        "sheet_name": "<Sheet_name>", #  can change
        "telegram_token": "<put your token here>", #  can change
        "telegram_chat_id": "put your chat id here", #  can change
        "position": "...", #  can change
        "department": "..." #  can change
    }
}

APPS_SCRIPT_URL = "<put your url web app of apps script here>"  #  can change

def handle_result(name, status, location, mode):
    routed_name = name if name in PERSON_TO_SHEET else None
    if not routed_name:
        print(f"⚠️ Không map được {name}")
        return

    sheet_info = PERSON_TO_SHEET[routed_name]
    data = {
        "name": NAME_MAP.get(name, name),
        "position": sheet_info.get("position", "UNKNOWN"),
        "status": status,
        "location": location,
        "sheet": sheet_info["sheet_name"],
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

    try:
        r = requests.post(APPS_SCRIPT_URL,
                         json=clean_data,
                         headers={"Content-Type": "application/json"},
                         timeout=10)
        print(f"Gửi Apps Script: {r.status_code}, Trả về: {r.text}")
    except Exception as e:
        print(f"Lỗi gửi Apps Script: {e}")
