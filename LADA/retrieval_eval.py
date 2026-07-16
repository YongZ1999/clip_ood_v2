"""In-process retrieval evaluation for the official LADA implementation.

The official LADA launcher starts a fresh Python process for every X-TAIL
task.  Its current text AdaptFormer is therefore only available immediately
after that task has trained; this module evaluates it at that point and
appends a durable per-task record to ``retrieval.json``.
"""

import json
from collections import defaultdict
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# Reuse the project-wide dataset parsing and exact R@K implementation so the
# official-LADA and main_incremental protocols have identical relevance rules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.retrieval_eval import (  # noqa: E402
    RetrievalImageDataset,
    build_caption_index,
    compute_retrieval_metrics,
    load_retrieval_dataset,
    parse_recall_ks,
    parse_retrieval_roots,
)


@torch.no_grad()
def _encode_images(model, dataset, device, batch_size, num_workers):
    loader = DataLoader(
        RetrievalImageDataset(dataset), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    features, seen_indices = [], []
    for images, indices in tqdm(loader, desc="LADA retrieval images", leave=False):
        image_features = model.image_encoder(images.to(device))
        features.append(F.normalize(image_features.float(), dim=-1).cpu())
        seen_indices.append(indices.cpu())

    features = torch.cat(features, dim=0)
    seen_indices = torch.cat(seen_indices, dim=0)
    if not torch.equal(seen_indices, torch.arange(len(dataset))):
        raise RuntimeError("Image dataloader changed retrieval order")
    return features


@torch.no_grad()
def _encode_texts(model, tokenize, captions, device, batch_size, text_tuner):
    features = []
    for start in tqdm(range(0, len(captions), batch_size), desc="LADA retrieval captions", leave=False):
        tokens = tokenize(captions[start:start + batch_size]).to(device)
        text_features = model.text_encoder(tokens, text_tuner)
        features.append(F.normalize(text_features.float(), dim=-1).cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def evaluate_lada_retrieval_model(model, tokenize, dataset, device, batch_size,
                                  num_workers, recall_ks, similarity_chunk_size,
                                  text_tuner):
    image_features = _encode_images(model, dataset, device, batch_size, num_workers)
    captions, image_to_text, text_to_image = build_caption_index(dataset)
    text_features = _encode_texts(model, tokenize, captions, device, batch_size * 2, text_tuner)
    metrics = compute_retrieval_metrics(
        image_features, text_features, image_to_text, text_to_image,
        parse_recall_ks(recall_ks), similarity_chunk_size,
    )
    metrics["num_images"] = int(image_features.shape[0])
    metrics["num_captions"] = int(text_features.shape[0])
    return metrics


def _summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] == "lada":
            grouped[row["dataset"]].append(row)

    summary = {}
    for dataset, dataset_rows in grouped.items():
        dataset_rows = sorted(dataset_rows, key=lambda row: row["step"])
        summary[dataset] = {
            "num_tasks": len(dataset_rows),
            "average": {
                key: sum(row[key] for row in dataset_rows) / len(dataset_rows)
                for key in ("i2t_r@1", "t2i_r@1", "i2t_r@5", "t2i_r@5", "i2t_r@10", "t2i_r@10")
            },
            "last": {
                key: dataset_rows[-1][key]
                for key in ("i2t_r@1", "t2i_r@1", "i2t_r@5", "t2i_r@5", "i2t_r@10", "t2i_r@10")
            },
        }
    return summary


def _load_existing(path):
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported retrieval schema in {path}")
    return payload.get("rows", [])


def evaluate_and_save_lada_retrieval(cfg, model, tokenize, device):
    """Evaluate current LADA text tuner and append per-task results.

    A frozen-text row is recorded alongside each LADA row.  The visual encoder
    is shared, so this is an exact in-process Frozen CLIP reference under the
    official LADA preprocessing and tokenizer.
    """
    dataset_names = [name.strip() for name in cfg.retrieval_datasets.split(",") if name.strip()]
    if not dataset_names:
        raise ValueError("retrieval_eval=True requires retrieval_datasets")

    task = cfg.dataset
    if task not in cfg.dataset_sequence:
        raise ValueError(f"Task {task!r} is not present in dataset_sequence")
    step = cfg.dataset_sequence.index(task) + 1
    roots = parse_retrieval_roots(cfg.retrieval_roots)
    output_path = Path(cfg.output_dir) / "retrieval.json"
    rows = [
        row for row in _load_existing(output_path)
        if not (row.get("task") == task and row.get("dataset") in dataset_names)
    ]

    model.eval()
    for dataset_name in dataset_names:
        dataset_root = roots.get(dataset_name, Path(cfg.retrieval_root))
        dataset = load_retrieval_dataset(dataset_name, dataset_root, cfg.retrieval_max_images)
        common = {
            "step": step,
            "task": task,
            "dataset": dataset.name,
            "dataset_meta": dataset.metadata(),
        }
        if cfg.retrieval_include_frozen:
            frozen_metrics = evaluate_lada_retrieval_model(
                model, tokenize, dataset, device, cfg.retrieval_batch_size,
                cfg.num_workers, cfg.retrieval_recall_ks,
                cfg.retrieval_similarity_chunk_size, text_tuner=None,
            )
            rows.append({"method": "frozen_clip", "metrics": frozen_metrics, **common,
                         **_flatten_metrics(frozen_metrics)})

        lada_metrics = evaluate_lada_retrieval_model(
            model, tokenize, dataset, device, cfg.retrieval_batch_size,
            cfg.num_workers, cfg.retrieval_recall_ks,
            cfg.retrieval_similarity_chunk_size, text_tuner=model.text_tuner,
        )
        rows.append({"method": "lada", "metrics": lada_metrics, **common,
                     **_flatten_metrics(lada_metrics)})
        print(f"[LADA Retrieval Task {step} | {dataset.name}] "
              f"I2T R@1={lada_metrics['i2t'].get('r@1', 0.0):.2f} "
              f"T2I R@1={lada_metrics['t2i'].get('r@1', 0.0):.2f}")

    rows.sort(key=lambda row: (row["step"], row["dataset"], row["method"]))
    payload = {
        "schema_version": 1,
        "protocol": "official_lada_in_process_text_tuner",
        "rows": rows,
        "lada_summary": _summary(rows),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"LADA retrieval results saved to {output_path}")
    return payload


def _flatten_metrics(metrics):
    return {
        "i2t_r@1": metrics["i2t"].get("r@1", 0.0),
        "i2t_r@5": metrics["i2t"].get("r@5", 0.0),
        "i2t_r@10": metrics["i2t"].get("r@10", 0.0),
        "t2i_r@1": metrics["t2i"].get("r@1", 0.0),
        "t2i_r@5": metrics["t2i"].get("r@5", 0.0),
        "t2i_r@10": metrics["t2i"].get("r@10", 0.0),
    }
