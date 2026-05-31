# Конфигурация и константы
import warnings
import torch

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# Веса для комбинирования сигналов
RAIN_WEIGHT = 0.35
WIND_WEIGHT = 0.20
TEMP_WEIGHT = 0.10
PRESSURE_WEIGHT = 0.0
HUMIDITY_WEIGHT = 0.05
PM10_WEIGHT = 0.15
PM25_WEIGHT = 0.15

# Параметры по умолчанию
DEFAULT_LAT = 51.17
DEFAULT_LON = 104.18
DEFAULT_HORIZON = 30
DEFAULT_WINDOW = 64
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32