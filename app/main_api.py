from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import joblib


app = FastAPI()

#создадим init класс загрузки csv файла, определяющим путь к файлу и загрузку файла тестовой выборки heart_test.csv 
class CSVDownloaderPreprocessor:
    def __init__(self, csv_dir: str = "../datasets"): 
        """
        указываем что файл тестовой выборки heart_test.csv,  по умолчанию в папке datasets, фиксируем оператором Path
        """
        # Получаем абсолютный путь к корню проекта
        project_root = Path(__file__).parent.parent  # predicting_cardiovascular_disease_risk/
        
        #Формируем путь к данным
        self.csv_dir = project_root / "datasets" if csv_dir is None else Path(csv_dir)
        self.csv_dir = self.csv_dir.resolve()  # Нормализуем путь
        # Проверим существование директории при инициализации
        if not self.csv_dir.exists():
            available = [p.name for p in project_root.parent.iterdir() if p.is_dir()]
            raise FileNotFoundError(
                f"Директория с данными не найдена: {self.csv_dir}\n"
                f"Проверьте:\n"
                f"1. Существует ли папка 'datasets' в {project_root}\n"
                f"2. Доступные папки в {project_root.parent}: {available}"
            )
    
    def load_csv(self, filename: str = "heart_train.csv") -> pd.DataFrame:
        """
        Загружаем csv файл тестовой выборки, по умолчанию heart_train.csv и преобразуем в датафрейм
        """
        try:
            file_path = self.csv_dir / filename
            # проверяем наличие файла по пути
            if not file_path.exists():
                raise FileNotFoundError(f"Файл {file_path} не найден")
                
            df = pd.read_csv(file_path)
            # проверяем заполнен ли файл
            if df.empty:
                raise ValueError("Файл пуст")
            return df
        except Exception as e:
            raise e
        
    def translate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Перевод названий столбцов, признаков"""
        trans_columns={'Unnamed: 0': 'порядковый номер', 
                        'Age': 'Возраст', 
                        'Cholesterol': 'Холестерин', 
                        'Heart rate': 'Частота сердечных сокращений', 
                        'Diabetes': 'Диабет',
                        'Family History': 'Семейный анамнез', 
                        'Smoking': 'Курение', 
                        'Obesity': 'Ожирение', 
                        'Alcohol Consumption': 'Употребление алкоголя',
                        'Exercise Hours Per Week': 'Часы упражнений в неделю', 
                        'Diet': 'Диета', 
                        'Previous Heart Problems': 'Предыдущие проблемы с сердцем',
                        'Medication Use': 'Прием лекарств', 
                        'Stress Level': 'Уровень стресса', 
                        'Sedentary Hours Per Day': 'Часы сидячего образа жизни в день', 
                        'Income': 'Доход',
                        'BMI': 'ИМТ', 
                        'Triglycerides': 'Триглицериды', 
                        'Physical Activity Days Per Week': 'Дни физической активности в неделю',
                        'Sleep Hours Per Day': 'Часы сна в день', 
                        'Blood sugar': 'Сахар в крови', 
                        'CK-MB': 'CK-MB (показатель поврежд. миокарда)', 
                        'Troponin': 'Тропонин (поврежден. серд. мышцы)', 
                        'Gender': 'Пол',
                        'Systolic blood pressure': 'Систолическое артериальное давление', 
                        'Diastolic blood pressure': 'Диастолическое артериальное давление',
                        'id': 'id'
                        }
        return df.rename(columns=trans_columns)

    def convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """преобразование типов"""
        df['Диета'] = df['Диета'].astype('category')
        df['Уровень стресса'] = df['Уровень стресса'].astype('category')
        return df
        
    def normalize_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """нормирование одного признака"""
        df['Дни физической активности в неделю'] =df['Дни физической активности в неделю']/7
        return df
    def correct_gender_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """корректировка признака Пол"""
        df['Пол'] = df['Пол'].apply(lambda x: 1 if x == 'male' else 0)
        df['Пол'] = df['Пол'].astype('bool')
        return df
    def drop_empty_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """удаление пропусков в данных"""
        df = df.dropna()
        return df
    def convert_bool_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """замена некоторых признаков на булевый"""
        df['Диабет'] = df['Диабет'].astype('bool')
        df['Семейный анамнез'] = df['Семейный анамнез'].astype('bool')
        df['Курение'] = df['Курение'].astype('bool')
        df['Ожирение'] = df['Ожирение'].astype('bool')
        df['Употребление алкоголя'] = df['Употребление алкоголя'].astype('bool')
        df['Предыдущие проблемы с сердцем'] = df['Предыдущие проблемы с сердцем'].astype('bool')
        df['Прием лекарств'] = df['Прием лекарств'].astype('bool')
        return df
    def change_zero_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """замена не физичных по смыслу 0 в тестотовой выборке на среднее из тренирвочной выборки"""
        df['Холестерин'] = df['Холестерин'].apply(lambda x: 0.5029544643918678 if x == 0 else x)
        df['Триглицериды'] = df['Триглицериды'].apply(lambda x: 0.5064384885317601 if x == 0 else x)
        df['Часы сна в день'] = df['Часы сна в день'].apply(lambda x: 0.5833904970960809 if x == 0 else x)
        return df
    def del_gender_income_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """удаление признаков пол и доход"""
        df = df.drop(columns='Доход')
        df = df.drop(columns='Пол')
        return df
    def remove_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """удаление признака порядковый номер и запись столбца id в индекс"""
        df = df.drop(columns='порядковый номер')
        df = df.set_index('id')
        return df

# сделаем класс, который использует лучшую модель и порог классификации для предсказания 
class UseModelThreshold:
    def __init__(self):
        """Определение модели МО и порога классификации то как они созранены в файле"""
        self.model_data = self._load_model()
        self.model = self.model_data['model']
        self.threshold = self.model_data['threshold']
        
    def _load_model(self):
        """Загрузка модели  МО и порога классификации из файла """
        model_path = Path(__file__).parent.parent / "model" / "best_model.joblib"    
        if not model_path.exists():
            raise FileNotFoundError("Файл модели не найден")
        return joblib.load(model_path)
    def GetPredict(self, df: pd.DataFrame) -> List[Dict]:
        """"Используя модель и порог классификации делаем педсказания и выгружаем в json"""
        # получаем вероятности из обработанной тестовой выборки heart_test
        proba = self.model.predict_proba(df)[:, 1]
        #применяем порог классификации
        pred = (proba >= self.threshold).astype(int)
        # собираем JSON
        results = []
        for row_id, pred in zip(df.index, pred):
            results.append({
            "id": int(row_id),
            "prediction": int(pred),
            })
        return results


@app.post("/predict")
async def predict():
    """
    Основной эндпоинт для получения предсказаний
    Загружает тестовые данные, обрабатывает их и возвращает предсказания
    """
    try:
        # Инициализация загрузчика данных
        preprocessor = CSVDownloaderPreprocessor()
        
        # Загрузка и предобработка данных
        df = preprocessor.load_csv("heart_test.csv")
        processed_df = preprocessor.translate_columns(df)
        processed_df = preprocessor.convert_dtypes(processed_df)
        processed_df = preprocessor.normalize_feature(processed_df)
        processed_df = preprocessor.correct_gender_feature(processed_df)
        processed_df = preprocessor.drop_empty_feature(processed_df)
        processed_df = preprocessor.convert_bool_feature(processed_df)
        processed_df = preprocessor.change_zero_feature(processed_df)
        processed_df = preprocessor.del_gender_income_feature(processed_df)
        processed_df = preprocessor.remove_index(processed_df)
        
        # Получение предсказаний
        predictor = UseModelThreshold()
        predictions = predictor.GetPredict(processed_df)
        
        return JSONResponse(content={"predictions" : predictions})
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)