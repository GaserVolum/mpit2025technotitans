import os
import pandas as pd
import numpy as np
from datetime import datetime
from model_service import TaxiBidModel

# === Пути ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

# === Загружаем данные ===
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 18"], errors="ignore")
df["is_done"] = df["is_done"].replace({"done": 1, "cancel": 0}).astype(int)

# === Выбираем конкретный заказ ===
order_index = 2  # можно заменить на любой индекс
order = df.iloc[order_index]
order_id = order["order_id"]

#Вывод полной информации о заказе
print("\n АНАЛИЗИРУЕМ ЗАКАЗ")
print("=" * 50)
print(f" order_id: {order_id}")
print(f" Время заказа: {order['order_timestamp']}")
print(f" Расстояние: {order['distance_in_meters']} м")
print(f"️ Длительность: {order['duration_in_seconds']} сек")
print(f" Водитель: {order['driver_id']} (рейтинг: {order['driver_rating']})")
print(f" Машина: {order['carmodel']} — {order['carname']}")
print(f" Платформа: {order['platform']}")
print(f" Расстояние подачи: {order['pickup_in_meters']} м")
print(f" Время подачи: {order['pickup_in_seconds']} сек")
print(f" Стартовая цена: {order['price_start_local']}")
print(f" Бид из базы: {order['price_bid_local']}")
print(f" Фактический результат: {'✅ принят' if order['is_done'] else '❌ отменён'}")
print("=" * 50)

# Функции для генерации фич
def get_time_of_day(hour):
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "day"
    elif 18 <= hour < 24:
        return "evening"
    else:
        return "night"

def prepare_features(order_row):
    row = order_row.copy()
    # преобразуем дату
    row["order_timestamp"] = pd.to_datetime(row["order_timestamp"], errors="coerce")
    row["driver_reg_date"] = pd.to_datetime(row["driver_reg_date"], errors="coerce")

    # временные признаки
    row["hour"] = row["order_timestamp"].hour if pd.notnull(row["order_timestamp"]) else 12
    row["day_of_week"] = row["order_timestamp"].dayofweek if pd.notnull(row["order_timestamp"]) else 0
    row["time_of_day"] = get_time_of_day(row["hour"])
    row["is_peak_hour"] = int(row["hour"] in range(7,10) or row["hour"] in range(17,21))
    row["is_weekend"] = int(row["day_of_week"] in [4,5,6])

    # числовые признаки
    row["driver_account_age_days"] = (
        (row["order_timestamp"] - row["driver_reg_date"]).days
        if pd.notnull(row["driver_reg_date"]) and pd.notnull(row["order_timestamp"]) else 0
    )

    row["price_diff"] = row["price_bid_local"] - row["price_start_local"]
    row["price_ratio"] = row["price_bid_local"] / (row["price_start_local"] + 1e-6)
    row["pickup_distance_ratio"] = row["pickup_in_meters"] / (row["distance_in_meters"] + 1e-6)
    row["pickup_time_ratio"] = row["pickup_in_seconds"] / (row["duration_in_seconds"] + 1e-6)
    row["price_per_km"] = row["price_bid_local"] / (row["distance_in_meters"] / 1000 + 1e-6)

    return row

# Загружаем модель
model = TaxiBidModel()

#Генерируем диапазон возможных бидов
start_price = order["price_start_local"]
bids = np.arange(start_price, start_price * 2, 5)

results = []
for bid in bids:
    row = order.copy()
    row["price_bid_local"] = bid
    prepared = prepare_features(row)
    p_accept = model.predict_proba(prepared)
    expected_revenue = bid * p_accept
    results.append((bid, p_accept, expected_revenue))

df_result = pd.DataFrame(results, columns=["bid", "p_accept", "expected_revenue"])

#  Находим лучшую цену
best = df_result.loc[df_result["expected_revenue"].idxmax()]

#  Выводим результат
print("\n Таблица кандидатов (часть):")
print(df_result.head(12).to_string(index=False))

print("\n Топ по ожидаемой прибыли:")
print(df_result.sort_values("expected_revenue", ascending=False).head(10).to_string(index=False))

print("\n РЕЗУЛЬТАТ АНАЛИЗА")
print(f"Оптимальная цена для заказа {order_id}: {best.bid:.0f} руб.")
print(f" Вероятность принятия: {best.p_accept:.2f}")
print(f" Ожидаемая прибыль: {best.expected_revenue:.2f}")

#  Сохраняем результаты
out_path = os.path.join(BASE_DIR, "data", f"optimal_bid_order_{order_id}.csv")
df_result.to_csv(out_path, index=False)
print(f"\n Результаты сохранены: {out_path}")
