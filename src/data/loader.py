from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import csv


@dataclass
class GraphTables:
    authors: List[dict]
    papers: List[dict]
    concepts: List[dict]
    author_paper: List[Tuple[int, int]]
    paper_concept: List[Tuple[int, int]]


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_graph_tables(data_dir: str | Path) -> GraphTables:
    data_dir = Path(data_dir)
    authors = _read_csv(data_dir / "authors.csv")
    papers = _read_csv(data_dir / "papers.csv")
    concepts = _read_csv(data_dir / "concepts.csv")
    author_rows = _read_csv(data_dir / "author_writes_paper.csv")
    concept_rows = _read_csv(data_dir / "paper_has_concept.csv")

    return GraphTables(
        authors=authors,
        papers=papers,
        concepts=concepts,
        author_paper=[
            (int(row["author_id"]), int(row["paper_id"])) for row in author_rows
        ],
        paper_concept=[
            (int(row["paper_id"]), int(row["concept_id"])) for row in concept_rows
        ],
    )


def graph_summary(tables: GraphTables) -> Dict[str, int]:
    return {
        "authors": len(tables.authors),
        "papers": len(tables.papers),
        "concepts": len(tables.concepts),
        "author_paper_edges": len(tables.author_paper),
        "paper_concept_edges": len(tables.paper_concept),
    }
