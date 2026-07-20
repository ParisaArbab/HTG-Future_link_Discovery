# Data

This repository includes a small MAG-style sample dataset so the complete pipeline can run locally.

## Node files

- `authors.csv`
- `papers.csv`
- `concepts.csv`

## Edge files

- `author_writes_paper.csv`
- `paper_has_concept.csv`

The sample data is synthetic and is intended for demonstration, testing, and portfolio use. It does not contain the full Microsoft Academic Graph because the full dataset is very large and cannot be distributed inside a small GitHub repository.

To use a larger dataset, replace the CSV files with files that use the same column names, then run:

```bash
python scripts/prepare_data.py
```
