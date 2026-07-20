from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint():
    response = client.post("/predict", json={"paper_id": 0, "top_k": 3})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3
