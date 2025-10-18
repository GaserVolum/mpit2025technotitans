import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from model_service import TaxiBidModel

# === Пути ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

# === 1. Загружаем данные ===
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 18"], errors="ignore")
df["is_done"] = df["is_done"].replace({"done": 1, "cancel": 0}).astype(int)

# === 2. Обработка временных признаков ===
df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
df["driver_reg_date"] = pd.to_datetime(df["driver_reg_date"], errors="coerce")

df["hour"] = df["order_timestamp"].dt.hour.fillna(12).astype(int)
df["day_of_week"] = df["order_timestamp"].dt.dayofweek.fillna(0).astype(int)

def get_time_of_day(h):
    if 6 <= h < 12:
        return "morning"
    elif 12 <= h < 18:
        return "day"
    elif 18 <= h < 24:
        return "evening"
    else:
        return "night"

df["time_of_day"] = df["hour"].apply(get_time_of_day)
df["is_peak_hour"] = df["hour"].apply(lambda x: int(x in range(7, 10) or x in range(17, 21)))
df["is_weekend"] = df["day_of_week"].apply(lambda x: int(x in [4, 5, 6]))
df["driver_account_age_days"] = (
    (df["order_timestamp"] - df["driver_reg_date"]).dt.days.fillna(0).clip(lower=0)
)

# === 3. Дополнительные признаки ===
df["price_diff"] = df["price_bid_local"] - df["price_start_local"]
df["price_ratio"] = df["price_bid_local"] / (df["price_start_local"] + 1e-6)
df["pickup_distance_ratio"] = df["pickup_in_meters"] / (df["distance_in_meters"] + 1e-6)
df["pickup_time_ratio"] = df["pickup_in_seconds"] / (df["duration_in_seconds"] + 1e-6)
df["price_per_km"] = df["price_bid_local"] / (df["distance_in_meters"] / 1000 + 1e-6)

# === 4. Отбираем нужные поля ===
features_list = [
    "distance_in_meters", "duration_in_seconds", "driver_rating", "platform",
    "price_start_local", "price_bid_local", "pickup_in_meters", "pickup_in_seconds",
    "day_of_week", "time_of_day", "price_diff", "price_ratio", "is_peak_hour",
    "is_weekend", "driver_account_age_days", "pickup_distance_ratio",
    "pickup_time_ratio", "price_per_km"
]

X = df[features_list]
y_true = df["is_done"]

# === 5. Загружаем модель ===
model_service = TaxiBidModel()

# === 6. Прогон через модель ===
preds, probas = [], []
print("🚀 Прогоняем все строки через модель...")
for i in tqdm(range(len(X))):
    row = X.iloc[i].to_dict()
    proba = model_service.predict_proba(row)
    label = int(proba >= 0.5)
    probas.append(proba)
    preds.append(label)

df["model_proba"] = probas
df["model_pred"] = preds

# === 7. Метрики качества ===
cm = confusion_matrix(y_true, preds)
roc = roc_auc_score(y_true, probas)
precision = precision_score(y_true, preds)
recall = recall_score(y_true, preds)
f1 = f1_score(y_true, preds)
accuracy = (y_true == preds).mean()

mismatches = len(df) - (y_true == preds).sum()

print("\n РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ:")
print(f" Совпадений: {(y_true == preds).sum()} из {len(df)}")
print(f" Несовпадений: {mismatches}")
print(f" Accuracy: {accuracy:.4f}")
print(f" ROC-AUC: {roc:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f"️ F1-score: {f1:.4f}")

print("\n🧾 Confusion Matrix (истина / предсказание):")
print(pd.DataFrame(cm, index=["Actual 0 (cancel)", "Actual 1 (done)"],
                   columns=["Pred 0 (cancel)", "Pred 1 (done)"]))

#  Сохраняем результат для анализа
output_path = os.path.join(BASE_DIR, "data", "model_predictions.csv")
df.to_csv(output_path, index=False)
print(f"\n💾 Подробный файл сохранён: {output_path}")
