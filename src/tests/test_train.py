import pandas as pd
import numpy as np

# =============== ИМПОРТ ТЕСТИРУЕМЫХ МОДУЛЕЙ ===============
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from models.autoencoder import create_default_autoencoder
# ======================================================

# from sklearn.model_selection import train_test_split

# all_data_dir = "D:\\yniver\\modeling_work_system\\data\\tarin\\final\\testing_scale.csv"
all_data_dir = "D:\\yniver\\modeling_work_system\\data\\train\\final\\final.csv"


model_1 = create_default_autoencoder()
df = pd.read_csv(
    filepath_or_buffer = all_data_dir, 
    sep = ','
)

normal_data = df[df['is_anom'] == False].copy()
anomal_data = df[df['is_anom'] == True].copy()

# 4. Удаляем целевую колонку
normal_data = normal_data.drop(columns = ['is_anom'])
anomal_data = anomal_data.drop(columns = ['is_anom'])

print(normal_data.columns)
print(anomal_data.columns)

# 5. Преобразуем в numpy и проверяем типы
normal_data = normal_data.values.astype(np.float32)
anomal_data = anomal_data.values.astype(np.float32)

# 6. Финальная проверка
print(f"Размер normal_data: {normal_data.shape}")
print(f"Размер anomal_data: {anomal_data.shape}")
print(f"Тип normal_data: {normal_data.dtype}")
print(f"Тип anomal_data: {anomal_data.dtype}")
print(f"Есть ли NaN в normal_data: {np.isnan(normal_data).any()}")
print(f"Есть ли NaN в anomal_data: {np.isnan(anomal_data).any()}")
print(f"Есть ли inf в normal_data: {np.isinf(normal_data).any()}")
print(f"Есть ли inf в anomal_data: {np.isinf(anomal_data).any()}")

# 7. Обучаем модель
history = model_1.fit(
    normal_data, normal_data,
    batch_size=64,
    epochs=10,
    validation_data=(anomal_data, anomal_data),
    verbose=1
)

print(history.history)