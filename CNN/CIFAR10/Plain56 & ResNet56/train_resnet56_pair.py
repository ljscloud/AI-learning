"""Train CIFAR-10 ResNet-56 and Plain-56, then save one comparison figure."""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_shortcut):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.use_shortcut = use_shortcut
        self.pad_channels = (out_channels - in_channels) // 2

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_shortcut:
            # CIFAR Option A: identity for equal shapes, spatial subsampling
            # and symmetric zero channel padding at stage transitions.
            shortcut = x
            if self.stride != 1:
                shortcut = shortcut[:, :, ::self.stride, ::self.stride]
            if self.pad_channels:
                batch, _, height, width = shortcut.shape
                padding = shortcut.new_zeros((batch, self.pad_channels, height, width))
                shortcut = torch.cat((padding, shortcut, padding), dim=1)
            out = out + shortcut

        return self.relu(out)


class CifarNet56(nn.Module):
    def __init__(self, use_shortcut):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self.make_stage(16, 16, 1, use_shortcut)  # [B, 16, 32, 32]
        self.stage2 = self.make_stage(16, 32, 2, use_shortcut)  # [B, 32, 16, 16]
        self.stage3 = self.make_stage(32, 64, 2, use_shortcut)  # [B, 64, 8, 8]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, 10)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def make_stage(in_channels, out_channels, stride, use_shortcut):
        blocks = [BasicBlock(in_channels, out_channels, stride, use_shortcut)]
        for _ in range(8):
            blocks.append(BasicBlock(out_channels, out_channels, 1, use_shortcut))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(self.pool(x).flatten(1))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_datasets(args, output_dir):
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_images = datasets.CIFAR10(
        args.data, train=True, transform=train_transform, download=True,
    )
    val_images = datasets.CIFAR10(args.data, train=True, transform=eval_transform)
    test_set = datasets.CIFAR10(args.data, train=False, transform=eval_transform)

    # Match the existing project's stratified 45,000 / 5,000 split.
    generator = torch.Generator().manual_seed(args.split_seed)
    labels = torch.tensor(train_images.targets)
    train_indices, val_indices = [], []
    for label in range(10):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)].tolist()
        val_indices.extend(indices[:500])
        train_indices.extend(indices[500:])
    (output_dir / "split.json").write_text(json.dumps({
        "split_seed": args.split_seed, "train": train_indices, "val": val_indices,
    }), encoding="utf-8")
    return Subset(train_images, train_indices), Subset(val_images, val_indices), test_set


def make_loader(dataset, args, *, shuffle, seed):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
        generator=torch.Generator().manual_seed(seed),
    )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_size
    return total_loss / count, 100 * correct / count


def train_one(name, use_shortcut, args, data, output_dir):
    # The shortcut flag adds no modules, so both models start with the
    # same trainable tensors and BatchNorm state at this seed.
    seed_all(args.seed)
    model = CifarNet56(use_shortcut=use_shortcut).to(args.device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_set, val_set, test_set = data
    val_loader = make_loader(val_set, args, shuffle=False, seed=args.seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint = output_dir / f"{name}_best.pt"
    columns = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"]
    history = []
    best_val_acc, best_epoch = -1.0, 0
    with (output_dir / f"{name}_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_seed = args.seed + epoch
            seed_all(epoch_seed)
            train_loader = make_loader(train_set, args, shuffle=True, seed=epoch_seed)
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = run_epoch(model, train_loader, args.device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, args.device)

            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                torch.save(model.state_dict(), checkpoint)
            row = dict(epoch=epoch, lr=lr, train_loss=train_loss, train_acc=train_acc,
                       val_loss=val_loss, val_acc=val_acc)
            history.append(row)
            writer.writerow(row)
            f.flush()
            scheduler.step()
            print(f"{name} {epoch:03d}/{args.epochs} | "
                  f"train loss {train_loss:.4f}, accuracy {train_acc:.2f}% | "
                  f"val loss {val_loss:.4f}, accuracy {val_acc:.2f}%", flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    test_loader = make_loader(test_set, args, shuffle=False, seed=args.seed)
    test_loss, test_acc = run_epoch(model, test_loader, args.device)
    result = dict(model=name, seed=args.seed, split_seed=args.split_seed,
                  parameters=parameters, best_epoch=best_epoch, best_val_acc=best_val_acc,
                  test_loss=test_loss, test_acc=test_acc)
    print(f"{name} | {parameters:,} parameters | test accuracy {test_acc:.2f}%", flush=True)
    return history, result


def plot_comparison(histories, results, output_dir):
    """Use only recorded metrics to produce a single four-panel comparison."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["ResNet-56", "Plain-56"]
    colors = ["#2563A6", "#D1682A"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    loss_ax, error_ax, val_ax, test_ax = axes.flat
    for history, result, label, color, style in zip(histories, results, labels, colors, ("-", "--")):
        epochs = [row["epoch"] for row in history]
        series = (
            (loss_ax, [row["train_loss"] for row in history]),
            (error_ax, [100 - row["train_acc"] for row in history]),
            (val_ax, [row["val_acc"] for row in history]),
        )
        for ax, values in series:
            ax.plot(epochs, values, label=label, color=color, linestyle=style,
                    linewidth=2, marker="o" if len(epochs) == 1 else None)
        val_ax.scatter(result["best_epoch"], result["best_val_acc"], color=color, zorder=3)

    loss_ax.set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    error_ax.set(title="Training error", xlabel="Epoch", ylabel="Error (%)", ylim=(0, 100))
    val_ax.set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))

    test_scores = [result["test_acc"] for result in results]
    bars = test_ax.bar(labels, test_scores, width=0.5, color=colors)
    test_ax.bar_label(bars, labels=[f"{score:.2f}%" for score in test_scores], padding=4)
    delta = test_scores[0] - test_scores[1]
    test_ax.set(title=f"Test accuracy | ResNet minus Plain: {delta:+.2f} pp",
                ylabel="Accuracy (%)", ylim=(0, 108))
    test_ax.set_yticks(range(0, 101, 20))

    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    loss_ax.legend(frameon=False)
    parameter_count = results[0]["parameters"]
    fig.suptitle(f"CIFAR-10: ResNet-56 vs Plain-56\n"
                 f"{parameter_count:,} parameters each | seed {results[0]['seed']} | "
                 f"{len(histories[0])} epochs\n"
                 "Dots mark the validation checkpoints used for the test results",
                 fontsize=13)
    path = output_dir / "resnet56_vs_plain56.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default="./outputs_resnet56_pair")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    if not 0 < args.lr < float("inf"):
        parser.error("lr must be positive and finite")

    output_dir = Path(args.out) / f"seed_{args.seed}"
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError:
        parser.error(f"Output directory already exists: {output_dir}. Choose a new --out.")
    config = dict(vars(args), torch_version=torch.__version__)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    data = make_datasets(args, output_dir)

    histories, results = [], []
    for name, use_shortcut in (("resnet56", True), ("plain56", False)):
        history, result = train_one(name, use_shortcut, args, data, output_dir)
        histories.append(history)
        results.append(result)

    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    path = plot_comparison(histories, results, output_dir)
    print(f"Saved comparison figure to {path}")


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
