from src.data.loader import load_graph_tables, graph_summary


def test_load_sample_graph():
    tables = load_graph_tables("data/raw")
    summary = graph_summary(tables)
    assert summary["authors"] == 30
    assert summary["papers"] == 60
    assert summary["concepts"] == 10
    assert summary["author_paper_edges"] > 0
    assert summary["paper_concept_edges"] > 0
