from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    *,
    epoch: int | None = None,
    total_epochs: int | None = None,
    progress: bool = True,
) -> Tuple[float, float]:
    # 對訓練子集執行一次最佳化；長時間 GPU 訓練時可選擇顯示進度。
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)
    progress_step = max(1, total_batches // 10)
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if progress and (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % progress_step == 0
        ):
            prefix = f"Epoch {epoch}/{total_epochs}" if epoch and total_epochs else "Train"
            print(
                f"\r{prefix} | batch {batch_index}/{total_batches} "
                f"({batch_index / max(total_batches, 1):6.1%}) "
                f"loss={loss.item():.4f}",
                end="",
                flush=True,
            )
    if progress:
        print()
    loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    # 評估階段不計算梯度，也不更新模型參數。
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return loss, accuracy


def fit_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    epochs: int,
    patience: int,
    device: torch.device,
    checkpoint_path: Path,
    progress: bool = True,
) -> Dict[str, List[float]]:
    """使用 early stopping 訓練模型，並保存最佳 checkpoint。"""
    # 每個 seed 都建立新的 optimizer；early stopping 後只保存並恢復
    # 驗證損失最低的模型。
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    history = {"epochs": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        if progress:
            print(f"\nStarting epoch {epoch}/{epochs} on {device}...")
        train_loss, train_acc = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=epochs,
            progress=progress,
        )
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_loss < best_val_loss - 1e-6
        if progress:
            status = "best checkpoint updated" if improved else "no improvement"
            print(
                f"Epoch {epoch}/{epochs} complete | "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.4f} | {status}"
            )

        if improved:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """回傳資料載入器樣本的分類機率。"""
    # 回傳 CPU tensor，避免 ensemble 平均與 CSV 匯出時持續占用 GPU 記憶體。
    model.eval()
    outputs = []
    for images, _ in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        outputs.append(probs.cpu())
    if not outputs:
        return torch.empty(0, 2)
    return torch.cat(outputs, dim=0)
