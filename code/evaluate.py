"""Evaluate a trained EEG encoder on classification and text retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import clip
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import EEGEncoder
from splits import decode, load_splits


@torch.inference_mode()
def encode_candidate_texts(
    texts: list[str],
    device: torch.device,
    cache: Path,
) -> torch.Tensor:
    clip_model, _ = clip.load("ViT-B/32", device=device, download_root=str(cache))
    clip_model.eval()
    embeddings = []
    for start in range(0, len(texts), 256):
        tokens = clip.tokenize(texts[start : start + 256]).to(device)
        embeddings.append(clip_model.encode_text(tokens).float())
    return F.normalize(torch.cat(embeddings), dim=1)


def load_evaluation_data(h5_path: Path, split_name: str):
    indices, split_summary = load_splits(h5_path)
    rows = indices[split_name]
    with h5py.File(h5_path, "r") as handle:
        eeg = handle["X"][:].astype(np.float32)
        labels = handle["y"][:].astype(np.int64)
        text_ids = decode(handle["text_id"][:])
        texts = decode(handle["text"][:])

    ordered_ids: list[str] = []
    id_to_text: dict[str, str] = {}
    for text_id, text in zip(text_ids, texts):
        if text_id not in id_to_text:
            ordered_ids.append(text_id)
            id_to_text[text_id] = text
    id_to_index = {text_id: index for index, text_id in enumerate(ordered_ids)}
    targets = np.asarray([id_to_index[text_id] for text_id in text_ids], dtype=np.int64)
    candidates = [id_to_text[text_id] for text_id in ordered_ids]
    return eeg[rows], labels[rows], targets[rows], candidates, split_summary


def classification_metrics(confusion: np.ndarray) -> dict[str, float]:
    true_positive = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0)
    actual = confusion.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros_like(true_positive),
        where=actual != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": 100.0 * float(true_positive.sum() / confusion.sum()),
        "balanced_accuracy": 100.0 * float(recall.mean()),
        "macro_f1": 100.0 * float(f1.mean()),
    }


@torch.inference_mode()
def evaluate(
    model: EEGEncoder,
    loader: DataLoader,
    text_table: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    ranks = []
    paired_cosines = []

    for eeg, labels, targets in loader:
        eeg, labels, targets = eeg.to(device), labels.to(device), targets.to(device)
        embeddings, logits = model(eeg)
        predictions = logits.argmax(dim=1)
        np.add.at(
            confusion,
            (labels.cpu().numpy(), predictions.cpu().numpy()),
            1,
        )

        similarities = embeddings @ text_table.T
        target_scores = similarities.gather(1, targets[:, None])
        ranks.append(((similarities > target_scores).sum(dim=1) + 1).cpu())
        paired_cosines.append(target_scores.squeeze(1).cpu())

    rank = torch.cat(ranks).numpy()
    metrics: dict[str, object] = classification_metrics(confusion)
    metrics.update(
        {
            "R@1": 100.0 * float(np.mean(rank <= 1)),
            "R@3": 100.0 * float(np.mean(rank <= 3)),
            "R@5": 100.0 * float(np.mean(rank <= 5)),
            "R@10": 100.0 * float(np.mean(rank <= 10)),
            "MRR": float(np.mean(1.0 / rank)),
            "paired_cosine": float(torch.cat(paired_cosines).mean()),
            "samples": int(len(rank)),
            "confusion_matrix": confusion.tolist(),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clip-cache", type=Path, default=Path("checkpoints/clip"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg, labels, targets, candidates, split_summary = load_evaluation_data(
        args.h5, args.split
    )
    text_table = encode_candidate_texts(candidates, device, args.clip_cache)
    model = EEGEncoder(
        channels=eeg.shape[1],
        embedding_dim=text_table.shape[1],
        num_classes=int(labels.max()) + 1,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.to(device)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(eeg),
            torch.from_numpy(labels),
            torch.from_numpy(targets),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    result = {
        "split": args.split,
        "split_summary": split_summary,
        "metrics": evaluate(
            model,
            loader,
            text_table,
            num_classes=int(labels.max()) + 1,
            device=device,
        ),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
