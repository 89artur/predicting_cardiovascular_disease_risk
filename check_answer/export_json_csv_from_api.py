import requests
import pandas as pd
import json
from pathlib import Path
import os

def save_api_response_to_files():
    # 1. Настройки пути
    output_dir = Path(r"C:\Users\msmk8\REPO\predicting_cardiovascular_disease_risk\check_answer")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Делаем запрос к API 
    api_url = "http://127.0.0.1:8000/predict"  # URL вашего API
    try:
        response = requests.post(api_url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Ошибка при запросе к API: {e}")
        return

    # 3. Сохраняем оригинальный JSON
    json_path = output_dir / "api_response.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 4. Преобразуем в CSV
    try:
        # Извлекаем массив predictions из JSON
        predictions = data.get('predictions', [])
        
        # Создаем DataFrame
        df = pd.DataFrame(predictions)
        
        # Сохраняем CSV
        csv_path = output_dir / "predictions_from_api_from_json.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Файлы успешно сохранены:\n"
              f"- JSON: {json_path}\n"
              f"- CSV: {csv_path}")
    except Exception as e:
        print(f"Ошибка при конвертации в CSV: {e}")

if __name__ == "__main__":
    save_api_response_to_files()