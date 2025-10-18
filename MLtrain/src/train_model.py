import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib
from datetime import datetime

#  Пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_xgb_enhanced.pkl")

# 1. Загрузка данных
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 18"], errors="ignore")

#  2. Целевая переменная
df["is_done"] = df["is_done"].map({"done": 1, "cancel": 0})
df = df.dropna(subset=["is_done"])

# 3. Фильтрация выбросов
df = df[
    (df["price_start_local"] > 0) &
    (df["price_bid_local"] > 0) &
    (df["distance_in_meters"] > 0) &
    (df["duration_in_seconds"] > 0) &
    (df["price_bid_local"] < df["price_start_local"] * 10)
]

#  4. Временные признаки
df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
df["hour"] = df["order_timestamp"].dt.hour
df["day_of_week"] = df["order_timestamp"].dt.dayofweek

def get_time_of_day(h):
    if pd.isna(h):
        return "unknown"
    if 6 <= h < 12:
        return "morning"
    elif 12 <= h < 18:
        return "day"
    elif 18 <= h < 24:
        return "evening"
    else:
        return "night"

df["time_of_day"] = df["hour"].apply(get_time_of_day)

#  5. Новый фичи-инжиниринг
# 1) Разница и соотношение цен
df["price_diff"] = df["price_bid_local"] - df["price_start_local"]
df["price_ratio"] = df["price_bid_local"] / (df["price_start_local"] + 1e-6)

# 2) Пиковые часы и выходные
df["is_peak_hour"] = ((df["hour"].between(7, 9)) | (df["hour"].between(17, 20))).astype(int)
df["is_weekend"] = df["day_of_week"].isin([4, 5, 6]).astype(int)

# 3) Возраст водителя в системе
df["driver_reg_date"] = pd.to_datetime(df["driver_reg_date"], errors="coerce")
df["driver_account_age_days"] = (
    (df["order_timestamp"] - df["driver_reg_date"]).dt.days.fillna(0).clip(lower=0)
)

# 4) Соотношение pickup/дистанция и времени
df["pickup_distance_ratio"] = df["pickup_in_meters"] / (df["distance_in_meters"] + 1e-6)
df["pickup_time_ratio"] = df["pickup_in_seconds"] / (df["duration_in_seconds"] + 1e-6)

# 5) Цена за километр
df["price_per_km"] = df["price_bid_local"] / (df["distance_in_meters"] / 1000 + 1e-6)

#  6. Отбор признаков
features = [
    "distance_in_meters", "duration_in_seconds", "driver_rating", "platform",
    "price_start_local", "price_bid_local", "pickup_in_meters", "pickup_in_seconds",
    "day_of_week", "time_of_day", "price_diff", "price_ratio",
    "is_peak_hour", "is_weekend", "driver_account_age_days",
    "pickup_distance_ratio", "pickup_time_ratio", "price_per_km"
]

features = [f for f in features if f in df.columns]
X = df[features]
y = df["is_done"].astype(int)

# 7. Препроцессинг
cat_features = ["platform", "time_of_day"]
num_features = [c for c in features if c not in cat_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# === 8. Базовая модель ===
xgb = XGBClassifier(
    random_state=42,
    eval_metric="auc",
    scale_pos_weight=y.value_counts()[0] / y.value_counts()[1]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", xgb)
])

#  9. RandomizedSearch для ускорения
param_dist = {
    "clf__n_estimators": [300, 400, 500],
    "clf__max_depth": [4, 6, 8, 10],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0],
    "clf__min_child_weight": [1, 3, 5]
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    scoring="roc_auc",
    cv=3,
    n_iter=15,
    n_jobs=-1,
    random_state=42,
    verbose=2
)

print("🔍 Поиск оптимальных параметров XGBoost...")
search.fit(X_train, y_train)

print(f"\n✅ Лучшие параметры: {search.best_params_}")
print(f"✅ Лучшая средняя AUC (CV): {search.best_score_:.4f}")


# === 10. Оценка на тесте ===
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)

print("\n🚀 Финальный ROC-AUC на тесте:", round(roc_auc, 4))
print(classification_report(y_test, y_pred))

# === 11. Сохранение ===
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
print(f"💾 Модель сохранена в {MODEL_PATH}")
