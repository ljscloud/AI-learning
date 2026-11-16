# 本次实际运行记录

验证日期：2026-11-15（Asia/Shanghai）。以下为真实运行结果，不是预设的示例数值。

## 环境

- Python：3.9.25（本机已有环境；README 建议新环境使用 Python 3.11）。
- PyTorch：2.6.0。
- torchvision：0.21.0。
- 设备：CPU；CUDA、MPS 均未启用。
- 解释器：/opt/anaconda3/envs/pytorch/bin/python。

本次实测验证了 CPU 训练和测试。CUDA、MPS 路径由代码支持，本次未在对应硬件实测。

## 执行与结果

```bash
python check_env.py
python basics_demo.py
python model.py
python train.py --epochs 5 --device cpu
python test.py --device cpu
```

- 环境检查通过。
- Tensor/autograd 示例：x=3 时 y=16，dy/dx=8。
- DataLoader 示例：10 个样本分为 4、4、2 三个批次。
- 模型形状检查：输入 (4,1,28,28)，输出 (4,10)。
- 训练：55,000 张；验证：5,000 张；测试：10,000 张。
- 最佳验证准确率：98.96%，出现在第 5 轮。
- 独立测试正确数量：9,900 / 10,000。
- 独立测试准确率：99.00%。
- 独立测试平均交叉熵损失：约 0.0309。

训练日志：

```text
Epoch 01/5 | train_loss=0.3079 train_acc=90.78% | val_loss=0.0923 val_acc=97.24%
Epoch 02/5 | train_loss=0.0893 train_acc=97.34% | val_loss=0.0613 val_acc=98.24%
Epoch 03/5 | train_loss=0.0634 train_acc=98.05% | val_loss=0.0508 val_acc=98.52%
Epoch 04/5 | train_loss=0.0500 train_acc=98.49% | val_loss=0.0488 val_acc=98.56%
Epoch 05/5 | train_loss=0.0395 train_acc=98.76% | val_loss=0.0392 val_acc=98.96%
```

完整数值见 outputs/history.csv 和 outputs/test_results.json。模型见 outputs/best.pt；逐样本结果见 outputs/predictions.csv。

初次下载受执行环境的网络限制，开放网络访问后已下载成功并完成实际训练。压缩包不包含 MNIST 数据；用户首次运行需要联网下载，或按 README 说明复制离线数据。

随机种子固定为 42。不同 PyTorch 版本、设备和数值计算方式仍可能造成结果差异，99.00% 是本次结果而非保证值。
