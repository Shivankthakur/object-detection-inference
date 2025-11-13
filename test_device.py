import torch
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))

import psutil
print("physical_cores:", psutil.cpu_count(logical=False))