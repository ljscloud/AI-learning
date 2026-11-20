"""MNIST 卷积网络；输入 [N, 1, 28, 28]，输出 [N, 10]。"""
import torch
from torch import nn


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [N, 16, 14, 14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [N, 32, 7, 7]
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        # CrossEntropyLoss 接收原始 logits，因此这里不加 softmax。
        return self.layers(x)


if __name__ == "__main__":
    model = MNISTCNN()
    output = model(torch.randn(4, 1, 28, 28))
    print(model)
    print("输入形状: (4, 1, 28, 28)")
    print("输出形状:", tuple(output.shape))
