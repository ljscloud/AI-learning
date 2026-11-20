"""右键直接运行，或执行 python train.py --epochs 5。"""
import argparse
import csv
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from common import (choose_device, evaluate, load_mnist, project_path,
                    save_json, seed_everything)
from model import MNISTCNN


def parse_args():
    parser = argparse.ArgumentParser(description="训练 MNIST CNN")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error("epochs/batch-size/lr 必须为正数，num-workers 必须非负")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = choose_device(args.device)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_mnist(project_path(args.data_dir), train=True)
    train_set, val_set = random_split(
        dataset, [55000, 5000],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_accuracy = -1.0
    history = []
    print(f"Device: {device}; train={len(train_set)}; validation={len(val_set)}")

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()  # 每轮重新开启训练模式。
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        val_loss, val_accuracy = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch, "train_loss": loss_sum / total,
            "train_accuracy": correct / total, "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch, "val_accuracy": val_accuracy,
            "seed": args.seed, "args": vars(args),
            "torch_version": str(torch.__version__),
            "model_name": "MNISTCNN",
        }
        # 文件用于重新加载并推理；未保存优化器状态，不用于精确断点续训。
        torch.save(checkpoint, output_dir / "last.pt")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(checkpoint, output_dir / "best.pt")
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.2%} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.2%} | "
            f"{row['seconds']:.1f}s", flush=True
        )

    save_json(output_dir / "train_config.json", {
        "arguments": vars(args), "device": str(device),
        "torch_version": str(torch.__version__),
        "best_val_accuracy": best_accuracy,
        "train_samples": len(train_set), "validation_samples": len(val_set),
    })
    print(f"训练完成。最佳验证准确率: {best_accuracy:.2%}")
    print(f"已保存模型: {output_dir / 'best.pt'}")
    print("下一步：运行 test.py（自定义输出目录时，用 --checkpoint 指向模型）")


if __name__ == "__main__":
    main()
