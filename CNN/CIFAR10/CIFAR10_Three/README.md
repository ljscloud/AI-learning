# CIFAR-10：LeNet、ResNet18 与无残差连接的 ResNet18

本项目提供三个网络的训练和一张四面板对比图。默认依次训练 LeNet、ResNet18、ResNet18（no skips），训练全部完成后自动保存图像。没有预填的准确率；所有性能曲线和准确率都来自实际生成的 CSV。

## 1. 无残差版本改了什么

标准 ResNet18 的 BasicBlock 是：

```text
输入 x ── Conv3×3 → BN → ReLU → Conv3×3 → BN ── (+) → ReLU
       └──────────── identity / Conv1×1 + BN ──────┘
```

无残差版本变为：

```text
输入 x ── Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU
```

`NoSkipBasicBlock` 删除所有 8 个块中的跳连相加，并删除 Layer2/3/4 首块中仅服务于捷径的 `1×1 Conv + BN` 投影分支。**主分支的卷积、BN、ReLU、通道数、卷积步长均保留**；分辨率降低仍由主分支的步长为 2 的卷积完成。

实现时先按相同过程初始化两个 ResNet18，再把无残差版本中的块替换为保留原主分支模块的 `NoSkipBasicBlock`。在相同初始化种子下，两者共有的 Conv/BN/FC 参数和缓冲区初始值相同，避免新建块引入额外初始化差异。

| 网络 | 块内跳连 | 可训练参数量 | 输出 |
|---|---|---:|---|
| LeNet 教学变体 | 无 | 62,006 | `[B, 10]` |
| CIFAR 适配版 ResNet18 | 有 | 11,173,962 | `[B, 10]` |
| ResNet18（no skips） | 全部删除 | 11,000,138 | `[B, 10]` |

无残差版本少了 173,824 个参数，来自三个捷径投影分支。不能把它描述成严格等参数量对照。两种深网络的各阶段输出尺寸相同：

```text
输入       [B,   3, 32, 32]
首层       [B,  64, 32, 32]
Layer1     [B,  64, 32, 32]
Layer2     [B, 128, 16, 16]
Layer3     [B, 256,  8,  8]
Layer4     [B, 512,  4,  4]
全局池化   [B, 512,  1,  1]
分类输出   [B, 10]
```

这里的 LeNet 使用 RGB 输入、ReLU、最大池化和 Dropout，是教学变体；两个 ResNet 使用 CIFAR 小图像首层配置：`3×3/stride=1/padding=1`，无开头的最大池化，无预训练权重。

## 2. 安装依赖

在项目目录、用于实际训练的 Python 环境内执行：

```bash
python -m pip install -r requirements.txt
```

若需要特定 CUDA 安装，请使用 PyTorch 官网针对目标设备给出的 torch/torchvision 配对命令。已经装好原实验环境时，新增依赖只有：

```bash
python -m pip install matplotlib
```

## 3. 可选：只检查模型

```bash
python train.py --inspect-only
```

该命令仅对全零输入做前向，输出形状和参数量，不下载数据、不更新权重，不需要 matplotlib。

## 4. 训练三个模型

```bash
python train.py \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.05 \
  --seed 42 \
  --split-seed 2026 \
  --workers 2 \
  --device cuda \
  --out ./outputs_three_models
```

若数据加载环境不支持多进程，可以使用 `--workers 0`；CPU 环境可设置 `--device cpu`。正式实验建议保留默认的 `--workers 2`：增强的随机数由 worker 使用，避免 LeNet 的 Dropout 与主进程数据增强竞争随机数流。

训练条件与原实验一致：

- CIFAR-10 官方训练集按类别分层划分，45,000 张训练、5,000 张验证，官方测试集 10,000 张。
- 三个模型复用同一组索引；训练集随机裁剪和翻转，验证/测试不使用随机增强；归一化均为 mean/std=0.5。
- SGD，初始学习率 0.05，momentum 0.9，weight decay 5e-4；余弦退火，100 epoch，batch size 128。
- 每个模型开始前使用相同种子；每轮重新设置随机种子并使用相同数据打乱/worker 种子。
- 每轮验证，按验证集准确率保存最佳权重；完成训练后载入最佳权重，测试集只评估一次。

三种模型依次执行，不是同时占用三份 GPU 模型内存。原来的 `outputs` 目录不作为新的默认输出位置。

## 5. 查看自动生成的图

完整运行后，目录结构如下：

```text
outputs_three_models/seed_42/
├── config.json
├── split.json
├── lenet_history.csv
├── resnet18_history.csv
├── resnet18_no_skip_history.csv
├── lenet_best.pt
├── resnet18_best.pt
├── resnet18_no_skip_best.pt
├── comparison.csv
├── three_models_comparison.png
└── three_models_comparison.svg
```

PNG 和 SVG 是同一张图的两种格式。图片使用英文标签，避免服务器缺少中文字体导致方框。

四个面板分别展示：

1. **Training loss**：三个网络的训练交叉熵损失曲线，用于观察优化过程。
2. **Validation accuracy**：验证集准确率曲线；圆点标出各自保存的最佳 checkpoint。
3. **Test accuracy**：加载各自最佳验证 checkpoint 后的测试准确率，柱顶标明百分比。
4. **Trainable parameters**：可训练参数量；纵轴为对数刻度，使 LeNet 和两个较大的网络均清晰可见。

图脚同时列出 ResNet18 减去无残差版本的测试准确率差，单位是**百分点**；若结果为负就按负值显示，不预设哪个模型更好。

## 6. 不重新训练，只重新绘图

三个模型训练完成后，可随时执行：

```bash
python plot_results.py --results-dir outputs_three_models/seed_42
```

绘图程序仅读取 `comparison.csv` 和三个 `*_history.csv`，不导入 torch、不下载数据、不训练。它会检查三个模型是否齐全、随机种子是否一致、轮数是否一致，以及选中的最佳验证指标是否与日志一致。缺少模型时会报错，避免生成不完整的三模型对比图。

## 7. 如何解释结果

重点比较 `ResNet18` 与 `ResNet18 (no skips)`，因为二者主分支深度、通道数、BN 和初始参数相同，主要变化是删除跳连及其投影分支。LeNet 用于观察浅层小模型与深层网络的差异。

应结合训练 Loss、验证曲线和最终测试准确率讨论。仅凭单次运行不能保证残差模型一定更好，也不能把参数量差异误写成完全等容量的实验。需要更稳定结论时，可保持 `--split-seed 2026`，分别使用 `--seed 42`、`43`、`44` 训练，并比较均值和标准差。

参考：[torchvision ResNet 官方实现](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html)。本项目的无跳连块在相同主分支计算后直接执行 ReLU，不再执行官方实现中的 `out += identity`。
