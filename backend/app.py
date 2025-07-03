# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os
import json
from fastapi.middleware.cors import CORSMiddleware

# Your modules
from baseline.retriever.retreiver import Retriever
from baseline.generator.generator import Generator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever()
generator = Generator()

# Load your existing index
index_path = "retriever_index"
if os.path.exists(f"{index_path}.faiss"):
    print("Loading index...")
    retriever.load(index_path)
else:
    raise RuntimeError("Index not found. Please build it first.")

# Request body
class QueryRequest(BaseModel):
    question: str
    task: str = "qa"
    options: list[str] | None = None
    context: str | None = None

@app.post("/ask")
async def ask_question(req: QueryRequest):
    # Same logic as your evaluation, but dynamic
    question = req.question
    task = req.task
    options = req.options

    # Retrieve context if not provided
    if not req.context:
        if task in ["qa", "classification"]:
            retrieved_chunks, _ = retriever.query(question, k=1)
            context = "\n\n".join(retrieved_chunks)
        else:
            context = ""
    else:
        context = req.context

    # Build prompt + generate answer
    prompt = generator.build_prompt(
        context=context,
        task_input=question,
        mode=task,
        options=options
    )
    answer = generator.generate_answer(
        prompt, mode=task, options=options
    )

    # Minimal normalizations for classification / mcq
    if task == "classification":
        answer = answer.strip().lower()
        if "offensive" in answer:
            answer = "Offensive"
        elif "non-offensive" in answer:
            answer = "Non-offensive"
        else:
            answer = "Unclear"
    elif task == "mcq":
        valid_letters = [chr(97+i) for i in range(len(options or []))]
        answer = answer.strip().lower()
        if answer not in valid_letters:
            for letter in valid_letters:
                if letter in answer:
                    answer = letter
                    break
            else:
                answer = "invalid"

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(timespec='seconds')
    }
