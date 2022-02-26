'''
from GPUtil import showUtilization as gpu_usage
import torch
from numba import cuda
gpu_usage()

torch.cuda.empty_cache()
gpu_usage()

cuda.select_device(0)
cuda.close()
cuda.select_device(0)
'''

import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()