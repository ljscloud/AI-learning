"""训练和测试共用的路径、预处理、设备选择与评估函数。"""
import json
import random
from pathlib import Path

import torch
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 尽量减少同一环境中的随机差异；不保证跨设备逐位一致。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        name = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
    if name == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA 不可用，请检查驱动与 PyTorch 安装，或使用 --device cpu")
    if name == "mps" and not mps_ok:
        raise RuntimeError("MPS 不可用，请使用 --device cpu")
    return torch.device(name)


def load_mnist(data_dir, train):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return datasets.MNIST(
        root=str(data_dir), train=train, download=True, transform=transform
    )


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()  # 关闭 Dropout。
    loss_sum, correct, total = 0.0, 0, 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
