"""独立进程加载已保存模型，在 MNIST 官方测试集评估。"""
import argparse
import csv

import torch
from torch.utils.data import DataLoader

from common import choose_device, load_mnist, project_path, save_json
from model import MNISTCNN


def parse_args():
    parser = argparse.ArgumentParser(description="独立测试 MNIST CNN")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size 必须大于 0")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = choose_device(args.device)
    checkpoint_path = project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"模型不存在: {checkpoint_path}。请先运行 train.py。")
    # 先在 CPU 加载，允许 GPU 训练的模型在 CPU 上测试。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = MNISTCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = load_mnist(project_path(args.data_dir), train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    confusion = torch.zeros(10, 10, dtype=torch.long)
    loss_sum = 0.0
    predictions = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        predicted = logits.argmax(dim=1)
        pairs = labels.cpu() * 10 + predicted.cpu()
        confusion += torch.bincount(pairs, minlength=100).reshape(10, 10)
        for label, prediction in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
            predictions.append({
                "index": len(predictions), "label": label, "prediction": prediction,
                "correct": int(label == prediction),
            })

    correct = confusion.diag().sum().item()
    total = len(dataset)
    per_class = {
        str(i): confusion[i, i].item() / confusion[i].sum().item()
        for i in range(10)
    }
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "device": str(device), "torch_version": str(torch.__version__),
        "test_samples": total, "correct": correct,
        "test_loss": loss_sum / total, "test_accuracy": correct / total,
        "per_class_accuracy": per_class,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
    }
    output_dir = project_path(args.output_dir)
    save_json(output_dir / "test_results.json", result)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "label", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Device: {device}; checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test loss: {result['test_loss']:.4f}")
    print(f"Test accuracy: {correct / total:.2%} ({correct}/{total})")
    print("各数字的分类正确率:", {k: f"{v:.2%}" for k, v in per_class.items()})
    print(f"结果已保存到: {output_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
