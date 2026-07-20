from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.loader import load_graph_tables


def recommend_concepts(paper_id: int, data_dir: str = "data/raw", top_k: int = 5):
    tables = load_graph_tables(data_dir)
    paper = next((p for p in tables.papers if int(p["paper_id"]) == paper_id), None)
    if paper is None:
        raise ValueError(f"Unknown paper_id: {paper_id}")

    text = f'{paper["title"]} {paper["abstract"]}'.lower()
    results = []
    for concept in tables.concepts:
        tokens = concept["name"].lower().split()
        overlap = sum(token in text for token in tokens)
        score = min(0.99, 0.35 + 0.20 * overlap)
        results.append(
            {
                "paper_id": paper_id,
                "paper_title": paper["title"],
                "concept_id": int(concept["concept_id"]),
                "concept": concept["name"],
                "confidence": round(score, 3),
            }
        )
    return sorted(results, key=lambda x: x["confidence"], reverse=True)[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    print(json.dumps(recommend_concepts(args.paper_id, args.data_dir, args.top_k), indent=2))


if __name__ == "__main__":
    main()
