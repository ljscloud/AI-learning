"""Train only DeepLeNet on CIFAR-10, using the original LeNet experiment settings."""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class DeepLeNet(nn.Module):
    def __init__(self, dropout=0.5, *, added_init="default"):
        super().__init__()
        # Initialize the five shared weight layers in the same order as LeNet.
        conv1 = nn.Conv2d(3, 6, kernel_size=5)
        conv2 = nn.Conv2d(6, 16, kernel_size=5)
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10),
        )

        def added_conv(channels):
            conv = nn.Conv2d(channels, channels, 3, padding=1)
            if added_init == "kaiming":
                nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(conv.bias)
            return conv

        # The four added convolutions keep the spatial size and channel count.
        self.features = nn.Sequential(
            conv1, nn.ReLU(),  # [B, 6, 28, 28]
            added_conv(6), nn.ReLU(),
            added_conv(6), nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 6, 14, 14]
            conv2, nn.ReLU(),  # [B, 16, 10, 10]
            added_conv(16), nn.ReLU(),
            added_conv(16), nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 16, 5, 5]
        )
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.features(x))  # [B, 10] logits


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_data(data_dir, split_seed, output_dir):
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    training_images = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_transform,
    )
    validation_images = datasets.CIFAR10(
        data_dir, train=True, download=False, transform=eval_transform,
    )
    test_set = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=eval_transform,
    )

    # Same split algorithm and default seed as train_three.py.
    generator = torch.Generator().manual_seed(split_seed)
    labels = torch.tensor(training_images.targets)
    train_indices, val_indices = [], []
    for label in range(10):
        class_indices = torch.where(labels == label)[0]
        order = torch.randperm(len(class_indices), generator=generator)
        indices = class_indices[order].tolist()
        val_indices.extend(indices[:500])
        train_indices.extend(indices[500:])

    (output_dir / "split.json").write_text(json.dumps({
        "split_seed": split_seed, "train": train_indices, "val": val_indices,
    }), encoding="utf-8")
    train_set = Subset(training_images, train_indices)
    val_set = Subset(validation_images, val_indices)
    return train_set, val_set, test_set


def make_loader(dataset, args, *, shuffle, seed):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
        worker_init_fn=seed_worker,
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

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)

    return total_loss / count, 100.0 * correct / count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default="./outputs_deep_lenet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--added-init", choices=("default", "kaiming"), default="default",
                        help="initialization of the four added convolutions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    if not 0 < args.lr < float("inf") or not 0 <= args.dropout <= 1:
        parser.error("lr must be positive and finite; dropout must be in [0, 1]")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.out) / f"seed_{args.seed}"
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError:
        raise SystemExit(f"Output directory already exists: {output_dir}. Choose a new --out.")

    config = dict(vars(args), model="deep_lenet", torch_version=torch.__version__)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train_set, val_set, test_set = make_data(args.data, args.split_seed, output_dir)

    seed_all(args.seed)
    model = DeepLeNet(args.dropout, added_init=args.added_init).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DeepLeNet | device={device} | trainable parameters={parameters:,}", flush=True)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    val_loader = make_loader(val_set, args, shuffle=False, seed=args.seed)
    checkpoint = output_dir / "deep_lenet_best.pt"
    best_val_acc, best_epoch = -1.0, 0

    columns = ("epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc")
    with (output_dir / "deep_lenet_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for epoch in range(1, args.epochs + 1):
            epoch_seed = args.seed + epoch
            seed_all(epoch_seed)
            train_loader = make_loader(train_set, args, shuffle=True, seed=epoch_seed)
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device)

            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                torch.save(model.state_dict(), checkpoint)

            writer.writerow((epoch, lr, train_loss, train_acc, val_loss, val_acc))
            f.flush()
            scheduler.step()
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"train loss={train_loss:.4f}, acc={train_acc:.2f}% | "
                  f"val loss={val_loss:.4f}, acc={val_acc:.2f}%", flush=True)

    # Evaluate the test set once, after selecting weights by validation accuracy.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_loader = make_loader(test_set, args, shuffle=False, seed=args.seed)
    test_loss, test_acc = run_epoch(model, test_loader, device)
    result = dict(model="deep_lenet", seed=args.seed, split_seed=args.split_seed,
                  parameters=parameters, best_epoch=best_epoch,
                  best_val_acc=best_val_acc, test_loss=test_loss, test_acc=test_acc,
                  added_init=args.added_init)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(f"Best validation: {best_val_acc:.2f}% (epoch {best_epoch}) | Test: {test_acc:.2f}%")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
