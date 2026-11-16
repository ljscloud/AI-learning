"""打印当前解释器与 PyTorch、GPU 状态。"""
import platform
import sys

import torch
import torchvision


if __name__ == "__main__":
    print("Python executable:", sys.executable)
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:", torch.cuda.get_device_name(i))
    print("Tensor 测试:", torch.rand(2, 3))
