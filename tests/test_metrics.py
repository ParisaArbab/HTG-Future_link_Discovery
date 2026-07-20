from src.evaluation.metrics import binary_metrics, precision_at_k


def test_binary_metrics_perfect_predictions():
    metrics = binary_metrics([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_precision_at_k():
    value = precision_at_k([1, 0, 1], [0.9, 0.8, 0.7], k=2)
    assert value == 0.5
