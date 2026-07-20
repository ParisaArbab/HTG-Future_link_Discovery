from __future__ import annotations

import random


def sample_negative_edges(
    positive_edges: list[tuple[int, int]],
    num_papers: int,
    num_concepts: int,
    count: int,
    seed: int = 42,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    positives = set(positive_edges)
    negatives: set[tuple[int, int]] = set()

    max_negative_count = num_papers * num_concepts - len(positives)
    count = min(count, max_negative_count)

    while len(negatives) < count:
        edge = (rng.randrange(num_papers), rng.randrange(num_concepts))
        if edge not in positives:
            negatives.add(edge)

    return list(negatives)
