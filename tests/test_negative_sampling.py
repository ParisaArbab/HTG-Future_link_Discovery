from src.training.negative_sampling import sample_negative_edges


def test_negative_edges_do_not_overlap():
    positives = [(0, 0), (1, 1)]
    negatives = sample_negative_edges(positives, 3, 3, 4, seed=7)
    assert len(negatives) == 4
    assert not set(positives).intersection(negatives)
