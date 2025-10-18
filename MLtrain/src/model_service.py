import os
import joblib
import pandas as pd

# === Пути ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_xgb_enhanced.pkl")

class TaxiBidModel:
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Инициализация сервиса модели.
        Загружает обученную модель XGBoost из models/model_xgb_enhanced.pkl
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ Модель не найдена: {model_path}\n"
                f"Сначала обучите её с помощью train_model.py"
            )
        self.model = joblib.load(model_path)
        print(f"✅ Модель загружена из {model_path}")

    def predict_proba(self, features: dict) -> float:
        """
        Рассчитывает вероятность того, что клиент примет предложение.

        :param features: словарь признаков для одного заказа
        :return: вероятность принятия (float от 0 до 1)
        """

        required_fields = [
            "distance_in_meters", "duration_in_seconds", "driver_rating", "platform",
            "price_start_local", "price_bid_local", "pickup_in_meters", "pickup_in_seconds",
            "day_of_week", "time_of_day", "price_diff", "price_ratio", "is_peak_hour",
            "is_weekend", "driver_account_age_days", "pickup_distance_ratio",
            "pickup_time_ratio", "price_per_km"
        ]

        missing = [f for f in required_fields if f not in features]
        if missing:
            raise ValueError(f"❌ Отсутствуют признаки: {missing}")

        sample = pd.DataFrame([features])
        probability = self.model.predict_proba(sample)[0, 1]
        return float(probability)

    def predict_label(self, features: dict, threshold: float = 0.5) -> int:
        """
        Возвращает бинарное решение (0/1) в зависимости от порога.
        :param threshold: порог вероятности (по умолчанию 0.5)
        """
        proba = self.predict_proba(features)
        return int(proba >= threshold)

# === Пример использования ===
if __name__ == "__main__":
    service = TaxiBidModel()

    example = {
        "distance_in_meters": 1500,
        "duration_in_seconds": 420,
        "driver_rating": 4.7,
        "platform": "android",
        "price_start_local": 200,
        "price_bid_local": 280,
        "pickup_in_meters": 600,
        "pickup_in_seconds": 180,
        "day_of_week": 5,
        "time_of_day": "evening",
        "price_diff": 80,
        "price_ratio": 1.4,
        "is_peak_hour": 1,
        "is_weekend": 1,
        "driver_account_age_days": 250,
        "pickup_distance_ratio": 0.4,
        "pickup_time_ratio": 0.43,
        "price_per_km": 187
    }

    proba = service.predict_proba(example)
    label = service.predict_label(example, threshold=0.5)

    print(f"🔮 Вероятность принятия: {proba:.3f}")
    print(f"📦 Решение модели (1=примет, 0=откажет): {label}")
