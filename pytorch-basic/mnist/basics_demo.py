"""不需要下载数据的 Tensor / autograd / Dataset / DataLoader 演示。"""
import torch
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.arange(10, dtype=torch.float32).reshape(-1, 1)
        self.y = 2 * self.x + 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def main():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print("Tensor:\n", tensor)
    print("shape:", tensor.shape, "dtype:", tensor.dtype, "device:", tensor.device)
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1
    y.backward()
    print("y = x² + 2x + 1；x=3 时，y =", y.item(), "dy/dx =", x.grad.item())
    dataset = ToyDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print("Dataset 长度:", len(dataset), "第 0 个样本:", dataset[0])
    for i, (features, targets) in enumerate(loader, start=1):
        print(f"batch {i}: x={features.flatten().tolist()}, y={targets.flatten().tolist()}")


if __name__ == "__main__":
    main()
