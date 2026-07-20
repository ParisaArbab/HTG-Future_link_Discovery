import json
from pathlib import Path

from src.training.train import similarity_baseline

if __name__ == "__main__":
    metrics_path = Path("artifacts/metrics.json")
    if not metrics_path.exists():
        similarity_baseline("data/raw", "artifacts")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("Evaluation Results")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:20s}: {value:.4f}")
        else:
            print(f"{name:20s}: {value}")
