# app.py
# python -m uvicorn backend.app:app --reload --port 8000  
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os
import json
from fastapi.middleware.cors import CORSMiddleware

# Your modules
from baseline.pipeline import Pipeline

pipeline = Pipeline()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load index
pipeline.setup_index()

class QueryRequest(BaseModel):
    question: str
    task: str = "qa"
    options: list[str] | None = None
    context: str | None = None

@app.post("/ask")
async def ask_question(req: QueryRequest):
    print(req)
    result = pipeline.process_question(
        question=req.question,
        task=req.task,
        options=req.options,
        context=req.context
    )
    return {
        "question": req.question,
        "answer": result["answer"],
        "context": result["context"],
        "prompt": result["prompt"],
        "timestamp": datetime.now().isoformat(timespec='seconds')
    }
