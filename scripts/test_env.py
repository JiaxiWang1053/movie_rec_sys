import torch
import pandas as pd
import lightgbm as lgb

print("Torch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Pandas:", pd.__version__)
print("LightGBM OK")