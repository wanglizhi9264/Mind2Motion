"""Train the Mind2Motion EEG encoder with CLIP alignment and classification."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import clip
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model import EEGEncoder


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EEGDataset(Dataset):
    def __init__(self, eeg: np.ndarray, labels: np.ndarray, text_targets: np.ndarray) -> None:
        self.eeg = torch.from_numpy(eeg.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))
        self.text_targets = torch.from_numpy(text_targets.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, index: int):
        return self.eeg[index], self.labels[index], self.text_targets[index]


def multi_positive_nce(
    eeg_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    text_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = eeg_embeddings @ text_embeddings.T / temperature
    positive = text_ids[:, None].eq(text_ids[None, :])
    neg_inf = torch.finfo(logits.dtype).min
    eeg_to_text = -(
        torch.logsumexp(logits.masked_fill(~positive, neg_inf), dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()
    text_to_eeg = -(
        torch.logsumexp(logits.T.masked_fill(~positive.T, neg_inf), dim=1)
        - torch.logsumexp(logits.T, dim=1)
    ).mean()
    return 0.5 * (eeg_to_text + text_to_eeg)


@torch.inference_mode()
def encode_texts(texts: list[str], device: torch.device, cache: Path) -> torch.Tensor:
    model, _ = clip.load("ViT-B/32", device=device, download_root=str(cache))
    model.eval()
    chunks = []
    for start in range(0, len(texts), 256):
        tokens = clip.tokenize(texts[start : start + 256]).to(device)
        chunks.append(model.encode_text(tokens).float())
    return F.normalize(torch.cat(chunks), dim=1)


@torch.inference_mode()
def evaluate(
    model: EEGEncoder,
    loader: DataLoader,
    text_table: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = total = 0
    reciprocal_ranks = []
    for eeg, labels, targets in loader:
        eeg, labels, targets = eeg.to(device), labels.to(device), targets.to(device)
        embeddings, logits = model(eeg)
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += len(labels)
        similarities = embeddings @ text_table.T
        target_scores = similarities.gather(1, targets[:, None])
        ranks = (similarities > target_scores).sum(dim=1) + 1
        reciprocal_ranks.append(1.0 / ranks.float())
    return {
        "accuracy": 100.0 * correct / total,
        "mrr": float(torch.cat(reciprocal_ranks).mean()),
    }


def load_data(path: Path):
    with h5py.File(path, "r") as handle:
        eeg = handle["X"][:]
        labels = handle["y"][:]
        split = decode(handle["split"][:])
        text_ids = decode(handle["text_id"][:])
        texts = decode(handle["text"][:])

    unique_ids: list[str] = []
    id_to_text: dict[str, str] = {}
    for text_id, text in zip(text_ids, texts):
        if text_id not in id_to_text:
            unique_ids.append(text_id)
            id_to_text[text_id] = text
    id_to_index = {text_id: index for index, text_id in enumerate(unique_ids)}
    targets = np.asarray([id_to_index[text_id] for text_id in text_ids], dtype=np.int64)
    candidate_texts = [id_to_text[text_id] for text_id in unique_ids]
    return eeg, labels, split, targets, candidate_texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip-cache", type=Path, default=Path("checkpoints/clip"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg, labels, split, targets, candidate_texts = load_data(args.h5)
    text_table = encode_texts(candidate_texts, device, args.clip_cache)

    loaders = {}
    for name in ("train", "val", "test"):
        mask = split == name
        loaders[name] = DataLoader(
            EEGDataset(eeg[mask], labels[mask], targets[mask]),
            batch_size=args.batch_size if name == "train" else args.eval_batch_size,
            shuffle=name == "train",
            drop_last=name == "train",
            num_workers=0,
        )

    model = EEGEncoder(
        channels=eeg.shape[1],
        embedding_dim=text_table.shape[1],
        num_classes=int(labels.max()) + 1,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    classifier_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_accuracy = -1.0
    best_state = None
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_eeg, batch_labels, batch_targets in loaders["train"]:
            batch_eeg = batch_eeg.to(device)
            batch_labels = batch_labels.to(device)
            batch_targets = batch_targets.to(device)
            target_embeddings = text_table[batch_targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                embeddings, logits = model(batch_eeg)
                nce = multi_positive_nce(
                    embeddings, target_embeddings, batch_targets, args.temperature
                )
                cosine = (1.0 - (embeddings * target_embeddings).sum(dim=1)).mean()
                classification = classifier_loss(logits, batch_labels)
                loss = 0.25 * nce + 0.025 * cosine + 2.0 * classification
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()

        validation = evaluate(model, loaders["val"], text_table, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], text_table, device)
    result = {
        "seed": args.seed,
        "best_validation_accuracy": best_accuracy,
        "test": test_metrics,
        "epochs_ran": len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    torch.save(
        {"model": best_state, "result": result, "args": vars(args)},
        args.output / "best.pt",
    )
    (args.output / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
