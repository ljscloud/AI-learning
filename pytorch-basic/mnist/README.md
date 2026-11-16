# 第 1 周：环境与 PyTorch 入门——MNIST 完整实践

本项目对应图片中的三个实践任务：编写 model.py/train.py/test.py；训练、保存、重新加载与独立测试；建立 AI-learning/pytorch-basic/mnist 目录并记录命令。可以在 PyCharm 中直接运行，也能从终端启动。

## 1. 项目文件

```text
AI-learning/
└── pytorch-basic/
    └── mnist/
        ├── model.py          # CNN 模型定义；可单独运行检查形状
        ├── train.py          # 训练、验证、保存最佳模型
        ├── test.py           # 重新加载模型，评估独立测试集
        ├── common.py         # 共用路径、预处理、设备与评估函数（必须保留）
        ├── check_env.py      # 检查解释器、PyTorch 和 GPU
        ├── basics_demo.py    # Tensor/autograd/Dataset/DataLoader 入门
        ├── requirements.txt  # 固定的依赖版本
        ├── README.md         # 本操作说明
        ├── VERIFICATION.md   # 本次实际运行记录
        ├── .gitignore
        ├── data/             # 首次运行自动下载 MNIST；压缩包不包含数据
        └── outputs/
            ├── best.pt           # 验证集上表现最好的模型
            ├── last.pt           # 最后一轮模型
            ├── history.csv       # 每轮训练和验证结果
            ├── train_config.json # 训练参数与环境
            ├── test_results.json # 测试准确率、损失、混淆矩阵
            └── predictions.csv   # 每张测试图片的真实和预测标签
```

除数据集外，压缩包附带本次实际训练产生的模型与结果。重新训练会覆盖同一 outputs 目录中的文件；要保留多次实验，使用不同的 --output-dir。

## 2. 任务一：准备环境，检查 Linux/SSH/conda/GPU

### 2.1 本地电脑或 Linux 服务器

有本地 Python 环境即可完成全部 MNIST 任务。只有使用远程服务器时才需要 SSH。在自己电脑的终端执行下面命令，并替换用户名、主机地址和端口：

```bash
ssh -p 22 username@server_ip
```

登录后执行：

```bash
pwd
python --version
conda --version
```

Linux/Windows 的 NVIDIA 机器再执行：

```bash
nvidia-smi
```

检查显卡型号、驱动版本、显存占用和是否有其他进程。nvidia-smi 中的 CUDA Version 表示驱动支持的 CUDA 版本上限，不能据此断定 Python 环境已经装好 CUDA 版 PyTorch；要继续检查 torch.cuda.is_available()。

Mac 不使用 nvidia-smi。Apple 芯片的 Mac 可尝试 MPS，其他情况直接使用 CPU。本项目会自动选择 CUDA → MPS → CPU。

### 2.2 创建独立 conda 环境

本教程建议新建 Python 3.11 环境，固定使用 PyTorch 2.6.0 + torchvision 0.21.0，便于重复操作。这里固定的是一个已实际验证的版本组合，不表示最新版本。

在 Anaconda Prompt、系统终端或已配置 conda 的 PyCharm Terminal 中执行：

```bash
conda create -n mnist-pytorch python=3.11 -y
conda activate mnist-pytorch
python --version
python -m pip install --upgrade pip
```

安装依赖时，以下方案按系统选择一个。

**Windows/Linux，先用 CPU 跑通（无需 NVIDIA 显卡）：**

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

**macOS：**

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0
```

**Windows/Linux，使用 NVIDIA GPU：**

根据显卡与驱动选择官方支持的构建。以下是 CUDA 12.4 构建的安装示例，只在硬件和驱动支持时使用：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

较新的 GPU 可能需要更新的 PyTorch/CUDA 构建，此时参考官方安装选择器，成对安装匹配的 torch 和 torchvision。无需安装 torchaudio。

固定版本对应的官方命令：https://docs.pytorch.org/get-started/previous-versions/

最新硬件与安装选择器：https://pytorch.org/get-started/locally/

### 2.3 将项目放到指定目录

解压项目，保留 AI-learning/pytorch-basic/mnist 层级。在 PyCharm 选择 File → Open，打开 mnist 文件夹。若自己从头创建，则先建 AI-learning，再建 pytorch-basic/mnist，并将所有 .py 文件放到 mnist 同一层。

不要将自己的脚本命名为 torch.py、torchvision.py 或 random.py，以免覆盖依赖包。

## 3. 任务二：在 PyCharm 中配置并验证

1. 打开 Settings/Preferences → Project → Python Interpreter。不同 PyCharm 版本的菜单文字可能略有差别，关键是找到项目解释器设置。
2. 添加已有的 conda 解释器，选择刚才创建的 mnist-pytorch 环境。
3. 找不到路径时，在已激活该环境的终端执行：

   ```bash
   python -c "import sys; print(sys.executable)"
   ```

4. 将输出的 python 可执行文件设为 PyCharm 项目解释器。Windows 通常以 python.exe 结尾，macOS/Linux 通常以 bin/python 结尾。
5. 打开 check_env.py，右键 → Run 'check_env'，检查 Python executable、PyTorch 和 torchvision 版本。
6. 没有 NVIDIA GPU 时，CUDA available: False 是正常情况；这不影响 CPU 训练。
7. 打开 model.py，右键运行，应输出输入 (4, 1, 28, 28)、输出 (4, 10)。

终端的激活环境与 PyCharm 的运行解释器要分别核对，二者不一定自动同步。

## 4. 任务三：学习 Tensor、autograd、Dataset、DataLoader

打开 basics_demo.py，右键运行，或执行：

```bash
python basics_demo.py
```

依次观察：

1. Tensor 是保存数值的多维数组，示例创建一个 2×2 的浮点张量，打印 shape、dtype 和 device。
2. requires_grad=True 告诉 PyTorch 记录对该张量的可微计算。
3. 示例定义 y=x²+2x+1，并执行 y.backward()。x=3 时 y=16，x.grad=8，因为导数是 2x+2。
4. Dataset 实现 __len__ 和 __getitem__，分别返回样本数量和指定索引的样本。
5. DataLoader 将 Dataset 按 batch_size=4 分批。10 个样本产生 4、4、2 三个批次。
6. 在训练中把 shuffle 设置为 True，可在每轮重新打乱样本顺序；验证和测试保持 False。

MNISTCNN 继承 nn.Module。它的 forward 对应前向传播；loss.backward() 通过 autograd 计算参数梯度；optimizer.step() 根据梯度更新参数。

## 5. 任务四：理解并运行 model.py

网络结构如下：

```text
灰度图片 [N,1,28,28]
→ Conv2d(1,16,3,padding=1) → ReLU → MaxPool2d(2)
→ [N,16,14,14]
→ Conv2d(16,32,3,padding=1) → ReLU → MaxPool2d(2)
→ [N,32,7,7]
→ Flatten → [N,1568]
→ Linear(1568,64) → ReLU → Dropout(0.2)
→ Linear(64,10)
→ 每个数字类别的 logits [N,10]
```

N 是批次大小。输入有 1 个灰度通道，每张图片为 28×28。最后 10 个输出对应数字 0～9。

标签是 0～9 的整数，不需要手动做 one-hot 编码。CrossEntropyLoss 接收原始 logits，模型最后不添加 softmax。预测时 logits.argmax(dim=1) 返回得分最高的类别。

## 6. 任务五：训练与保存模型

### 6.1 PyCharm 直接运行

1. 确保 common.py、model.py、train.py、test.py 位于同一文件夹。
2. 打开 train.py，右键 → Run 'train'。
3. 第一次运行需要联网下载数据集，存入当前项目的 data/MNIST/raw；后续检测到完整文件后会复用。
4. 默认训练 5 轮，batch_size=128，学习率 0.001，随机种子 42。
5. 等待打印“训练完成”，查看 outputs 中的模型和日志文件。

代码按脚本所在目录定位数据和输出，即使 PyCharm 的 Working directory 不同也能找到默认路径。相对的 --data-dir、--output-dir 和 --checkpoint 也以 mnist 脚本目录为基准。

### 6.2 命令行训练

在 PyCharm Terminal 中先进入 mnist 目录并激活环境：

```bash
conda activate mnist-pytorch
python train.py --epochs 5 --batch-size 128 --lr 0.001 --device auto
```

如果终端不在 mnist 中，先执行 cd，将目录替换成你解压后 mnist 的实际路径：

```bash
cd "/你的实际路径/AI-learning/pytorch-basic/mnist"
```

强制 CPU：

```bash
python train.py --epochs 5 --device cpu
```

强制 CUDA（需要 CUDA 可用）：

```bash
python train.py --epochs 5 --device cuda
```

Apple 芯片 Mac 尝试 MPS：

```bash
python train.py --epochs 5 --device mps
```

用独立目录保留一组新实验：

```bash
python train.py --epochs 10 --output-dir outputs/run2
```

### 6.3 每一步训练在做什么

1. 固定随机种子，选择计算设备。
2. 使用 torchvision.datasets.MNIST 获取官方 60,000 张训练图片。
3. ToTensor 将图片变成 [1,28,28] 浮点张量，像素从 0～255 映射到 0～1。
4. Normalize 使用固定均值 0.1307、标准差 0.3081 做标准化，训练、验证和测试使用相同预处理。
5. 用固定种子将训练数据分成 55,000 张训练样本和 5,000 张验证样本。官方 10,000 张测试集不参与这一步。
6. DataLoader 分批读取数据，默认 num_workers=0，便于 Windows 和 PyCharm 直接运行。
7. 创建模型、CrossEntropyLoss 和 Adam 优化器。
8. 每轮调用 model.train()；每个批次依次执行清除梯度、前向计算、计算损失、反向传播、更新参数。
9. 每轮结束调用 model.eval()，关闭 Dropout，并在不记录梯度的情况下计算验证指标。
10. 每轮保存 last.pt；当验证准确率刷新最高值时保存 best.pt。
11. 把各轮损失、准确率、耗时写入 history.csv，把配置写入 train_config.json。

best.pt 和 last.pt 保存的是模型参数及简单元数据，能用于重新加载和推理。这里没有保存优化器状态，不提供精确的断点续训。

### 6.4 阅读训练日志

日志格式示例（数值以本次实际执行为准）：

```text
Epoch 01/5 | train_loss=... train_acc=... | val_loss=... val_acc=... | ...s
```

- epoch：遍历整个训练集一次。
- train_loss：该轮训练损失，按样本数加权平均。
- train_acc：训练过程中各批次预测的累计正确率，计算时 Dropout 开启且参数持续更新。
- val_loss、val_acc：该轮结束后固定模型在验证集上的损失与准确率。
- seconds：该轮训练与验证的耗时。

训练准确率与验证准确率的计算状态不同，某轮验证准确率高于训练准确率不一定是错误。不能只凭训练集准确率判断模型效果。

## 7. 任务六：重新加载与独立测试

### 7.1 在 PyCharm 运行

1. train.py 结束后确认 outputs/best.pt 存在；压缩包也附带本次训练模型。
2. 打开 test.py，右键 → Run 'test'。若菜单提供 pytest，选择普通 Python 运行配置，而非 pytest。
3. test.py 是另一个进程，它新建模型并从磁盘加载参数，没有复用 train.py 内存中的模型。
4. 等待评估完成，查看测试准确率及 outputs/test_results.json。

### 7.2 在终端运行

```bash
python test.py --checkpoint outputs/best.pt --device auto
```

测试另一组实验：

```bash
python test.py --checkpoint outputs/run2/best.pt --output-dir outputs/run2
```

强制将保存的模型放到 CPU 测试：

```bash
python test.py --checkpoint outputs/best.pt --device cpu
```

### 7.3 测试流程

1. 确认模型路径存在。
2. torch.load(..., map_location="cpu", weights_only=True) 先把检查点加载到 CPU。
3. 新建 MNISTCNN，并通过 load_state_dict 恢复权重。
4. 把模型移动到目标设备，调用 model.eval()。
5. 加载 train=False 对应的官方 10,000 张测试图片，不打乱顺序。
6. 在 torch.inference_mode() 中前向计算，不反向传播、不更新参数。
7. 汇总测试损失、准确率、各数字的分类正确率和 10×10 混淆矩阵。
8. 保存 test_results.json 和 predictions.csv。

混淆矩阵的行是真实类别，列是预测类别；对角线为正确分类数量。JSON 中 test_accuracy 是 0～1 的比例，如 0.99 表示 99%。predictions.csv 中 correct=0 的行就是错分样本。

不要使用测试集挑选最佳 epoch 或反复据此调整超参数；模型选择使用验证集，最终效果由独立测试集报告。

## 8. 任务七：建立仓库并记录启动命令

在 AI-learning 根目录执行：

```bash
git init
git add pytorch-basic/mnist
git status
git commit -m "Add MNIST PyTorch training and independent evaluation"
```

这些是供你创建仓库时执行的命令，本交付未自动创建或提交 Git 仓库。若当前目录已经属于现有 Git 仓库，直接使用该仓库即可，无需再次 git init。

.gitignore 忽略 data、模型 .pt、Python 缓存及 IDE 目录，避免把较大的文件加入源码版本管理。提交后，模型可以作为单独交付文件附上。

README 中应保留的最小启动命令：

```bash
conda activate mnist-pytorch
python check_env.py
python basics_demo.py
python model.py
python train.py --epochs 5 --batch-size 128 --lr 0.001 --device auto
python test.py --checkpoint outputs/best.pt --device auto
```

需要记录当前完整依赖时执行（按各自平台保存）：

```bash
python -m pip freeze > environment-freeze.txt
```

## 9. 常见问题

**No module named torch/torchvision**：运行脚本的解释器未安装依赖。运行 check_env.py 或打印 sys.executable，核对 PyCharm 项目解释器与安装依赖时的解释器。

**找不到匹配版本的安装包**：本教程的固定版本建议配合 Python 3.11 使用。不要用系统中更新的 Python 3.15 安装这组较早的依赖。

**CUDA available 为 False**：没有 NVIDIA 显卡时直接用 CPU/MPS。有显卡时检查驱动、PyTorch 构建以及 PyCharm 使用的解释器。

**torchvision::nms does not exist 等版本错误**：通常是 torch、torchvision 版本或 CPU/CUDA 构建不匹配，在独立环境中按官方成对安装。

**数据下载失败**：检查网络、代理和 HTTPS 证书。可在能联网的电脑先运行一次下载，然后把整个 data/MNIST 目录复制到目标机器的 data 目录。也可从 https://ossci-datasets.s3.amazonaws.com/mnist/ 下载四个文件：train-images-idx3-ubyte.gz、train-labels-idx1-ubyte.gz、t10k-images-idx3-ubyte.gz、t10k-labels-idx1-ubyte.gz；将原始 .gz 文件和解压后的同名无 .gz 文件放到 data/MNIST/raw 后再运行。不要通过禁用 SSL 校验解决证书问题。

**找不到 best.pt**：先运行 train.py；如果训练指定了 --output-dir，测试时也应通过 --checkpoint 指向对应位置。

**显存不足**：将 --batch-size 改为 64 或 32，或使用 --device cpu。

**MPS 算子或运行错误**：先用 --device cpu 验证完整流程，再根据硬件升级适配的 torch/torchvision。

**运行 test.py 出现 collected 0 items**：PyCharm 使用了 pytest 配置。新建普通 Python 配置，把 Script path 设置为 test.py。

**conda activate 不可用**：Windows 使用 Anaconda Prompt；macOS/Linux 可先执行 conda init zsh 或 conda init bash，再重新打开终端。

## 10. 验收清单

- [ ] 能说明解释器路径、torch/torchvision 版本以及当前使用的设备。
- [ ] basics_demo.py 打印 y=16、梯度为 8，产生 3 个数据批次。
- [ ] model.py 输出 (4,10)。
- [ ] 能从 PyCharm 和命令行启动 train.py。
- [ ] 训练结束生成 best.pt、last.pt、history.csv、train_config.json。
- [ ] 重新启动 test.py，加载保存的模型并评估 10,000 张测试图片。
- [ ] 生成 test_results.json 和 predictions.csv，并能解释准确率。
- [ ] 保留源代码、模型、实际结果与 README。

图片未规定准确率门槛，因此以完整跑通训练、保存、加载和独立测试为验收重点。不同设备、版本和随机计算可能产生不同数值。

## 11. 官方参考

- PyTorch 安装：https://pytorch.org/get-started/locally/
- 固定版本与匹配构建：https://docs.pytorch.org/get-started/previous-versions/
- MNIST 参数：https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html
- ToTensor：https://docs.pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
- 保存和加载参数：https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

本周优先理解代码与 PyTorch 基础，对应图片中的安排，不需要另行精读论文。
