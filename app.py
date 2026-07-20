from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import recommend_concepts

app = FastAPI(
    title="FutureLink HGT API",
    description="Predict missing paper-concept links in a heterogeneous academic graph.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    paper_id: int = Field(..., ge=0)
    top_k: int = Field(5, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok", "project": "FutureLink HGT"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        predictions = recommend_concepts(request.paper_id, top_k=request.top_k)
        return {"predictions": predictions}
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
