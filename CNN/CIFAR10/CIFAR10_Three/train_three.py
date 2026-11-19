"""Compare LeNet, ResNet18 and ResNet18 without skips on CIFAR-10."""
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
from torchvision import datasets, models, transforms

from plot_results import MODEL_ORDER, plot_comparison, require_matplotlib


class LeNet(nn.Module):
    # RGB + ReLU + max pooling + optional dropout: not the original LeNet-5.
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class NoSkipBasicBlock(nn.Module):
    """Keep a BasicBlock's initialized main branch; remove its whole shortcut."""

    def __init__(self, block, *, use_bn=True):
        super().__init__()
        self.conv1 = block.conv1
        # Identity preserves the forward layout without normalization or parameters.
        self.bn1 = block.bn1 if use_bn else nn.Identity()
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2 if use_bn else nn.Identity()
        # No identity addition or shortcut-only downsample Conv/BN modules.

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out)


def build_model(name, dropout):
    if name == "lenet":
        return LeNet(dropout)
    if name not in ("resnet18", "resnet18_no_skip", "resnet18_no_skip_no_bn"):
        raise ValueError(f"Unknown model: {name}")
    # Construct both variants identically before removing skips. With the same
    # seed their shared Conv/BN/FC weights start at exactly the same values.
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    use_bn = name != "resnet18_no_skip_no_bn"
    if not use_bn:
        model.bn1 = nn.Identity()
    if name != "resnet18":
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
            for index, block in enumerate(layer):
                layer[index] = NoSkipBasicBlock(block, use_bn=use_bn)
    return model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_data(args, out):
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    train_base = datasets.CIFAR10(args.data, train=True, download=True,
                                  transform=train_tf)
    val_base = datasets.CIFAR10(args.data, train=True, download=False,
                                transform=eval_tf)
    test = datasets.CIFAR10(args.data, train=False, download=True,
                            transform=eval_tf)
    # Fixed stratified split: 4500 train + 500 validation per class.
    g = torch.Generator().manual_seed(args.split_seed)
    targets = torch.tensor(train_base.targets)
    train_idx, val_idx = [], []
    for c in range(10):
        idx = torch.where(targets == c)[0]
        idx = idx[torch.randperm(len(idx), generator=g)].tolist()
        val_idx.extend(idx[:500])
        train_idx.extend(idx[500:])
    (out / "split.json").write_text(json.dumps({
        "split_seed": args.split_seed, "train": train_idx, "val": val_idx,
    }), encoding="utf-8")
    return Subset(train_base, train_idx), Subset(val_base, val_idx), test


def make_loader(dataset, args, shuffle, seed):
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            count += y.size(0)
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    return loss_sum / count, 100.0 * correct / count


def inspect_model(model):
    model.eval()
    hooks = []
    for name, module in model.named_children():
        def show(_module, _inputs, output, name=name):
            print(f"  {name}: {tuple(output.shape)}")
        hooks.append(module.register_forward_hook(show))
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 32, 32))
    for hook in hooks:
        hook.remove()
    assert output.shape == (2, 10)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable parameters: {params:,}")
    return params


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(name, args, data, out):
    seed_all(args.seed)
    model = build_model(name, args.dropout)
    print(f"\n{name}, input: (2, 3, 32, 32)")
    params = inspect_model(model)
    model.to(args.device)
    train_set, val_set, test_set = data
    val_loader = make_loader(val_set, args, False, args.seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    history, best_acc, best_epoch = [], -1.0, 0
    checkpoint = out / f"{name}_best.pt"
    for epoch in range(1, args.epochs + 1):
        # Match epoch RNG as well as shuffle/worker seeds across the models.
        # With workers > 0, augmentation RNG is isolated from model Dropout RNG.
        seed_all(args.seed + epoch)
        train_loader = make_loader(train_set, args, True, args.seed + epoch)
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            model, train_loader, args.device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, args.device)
        history.append(dict(epoch=epoch, lr=lr, train_loss=train_loss,
                            train_acc=train_acc, val_loss=val_loss,
                            val_acc=val_acc))
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), checkpoint)
        scheduler.step()
        print(f"{name} {epoch:03d}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")
        write_csv(out / f"{name}_history.csv", history)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device,
                                    weights_only=True))
    # The test set is evaluated only after validation-based selection.
    test_loader = make_loader(test_set, args, False, args.seed)
    test_loss, test_acc = run_epoch(model, test_loader, args.device)
    result = dict(model=name, seed=args.seed, split_seed=args.split_seed,
                  parameters=params, best_epoch=best_epoch,
                  best_val_acc=best_acc, test_loss=test_loss, test_acc=test_acc)
    print("Result:", result)
    return result


def parse_args(default_out="./outputs_three_models"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive; workers must be nonnegative")
    return args


def prepare_run(args):
    out = Path(args.out) / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), torch_version=torch.__version__)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return out


def main():
    args = parse_args()
    if args.inspect_only:
        for name in MODEL_ORDER:
            print(name)
            inspect_model(build_model(name, args.dropout))
        return
    # Report a missing plotting dependency before spending time on training.
    require_matplotlib()
    out = prepare_run(args)
    data = make_data(args, out)
    results = []
    for name in MODEL_ORDER:
        results.append(train_one(name, args, data, out))
        write_csv(out / "comparison.csv", results)
    plot_comparison(out)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
