from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from src.data.loader import load_graph_tables, graph_summary


def _text_features(texts: list[str], dim: int, seed: int) -> np.ndarray:
    """Create deterministic lightweight text features without external models."""
    features = np.zeros((len(texts), dim), dtype=np.float32)
    for row_id, text in enumerate(texts):
        for token in text.lower().split():
            index = abs(hash((token, seed))) % dim
            features[row_id, index] += 1.0
        norm = np.linalg.norm(features[row_id])
        if norm > 0:
            features[row_id] /= norm
    return features


def prepare_data(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    feature_dim: int = 64,
) -> dict:
    tables = load_graph_tables(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    author_x = _text_features([x["name"] for x in tables.authors], feature_dim, 1)
    paper_x = _text_features(
        [f'{x["title"]} {x["abstract"]}' for x in tables.papers], feature_dim, 2
    )
    concept_x = _text_features([x["name"] for x in tables.concepts], feature_dim, 3)

    np.save(processed_dir / "author_features.npy", author_x)
    np.save(processed_dir / "paper_features.npy", paper_x)
    np.save(processed_dir / "concept_features.npy", concept_x)
    np.save(
        processed_dir / "author_paper_edges.npy",
        np.asarray(tables.author_paper, dtype=np.int64).T,
    )
    np.save(
        processed_dir / "paper_concept_edges.npy",
        np.asarray(tables.paper_concept, dtype=np.int64).T,
    )

    summary = graph_summary(tables)
    summary["feature_dim"] = feature_dim
    (processed_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
