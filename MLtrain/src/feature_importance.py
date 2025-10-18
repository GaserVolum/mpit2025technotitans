import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Пути ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_xgb_enhanced.pkl")

# === 1. Загрузка модели ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Модель не найдена по пути: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print(f"✅ Модель загружена из {MODEL_PATH}")

# === 2. Извлекаем пайплайн и модель XGB ===
clf = model.named_steps["clf"]
preprocessor = model.named_steps["preprocessor"]

# === 3. Получаем имена признаков после OneHotEncoder ===
num_features = preprocessor.transformers_[0][2]
cat_encoder = preprocessor.transformers_[1][1]
cat_features = preprocessor.transformers_[1][2]

encoded_cat_features = list(cat_encoder.get_feature_names_out(cat_features))
all_features = num_features + encoded_cat_features

# === 4. Извлекаем важности признаков ===
importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values("importance", ascending=False)

# === 5. Выводим топ-15 ===
top_features = feature_importance_df.head(15)
print("\n🏆 Топ-15 наиболее значимых признаков:")
print(top_features)

# === 6. Визуализация ===
plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"][::-1], top_features["importance"][::-1])
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

# Сохранение графика
plot_path = os.path.join(BASE_DIR, "models", "feature_importance.png")
plt.savefig(plot_path)
plt.show()

print(f"\n📊 График важности признаков сохранён: {plot_path}")
