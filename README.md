# RAG Project – Summer Semester 2025. 
## Team NNN
### Nabanita Bhowmick
### Nafees Iqbal
### Naznin Ahmed

---

## Previous weeks tasks Info

All previous weeks work and corresponding README files can be found in their respective branches.  
In the `main` branch, only the **current week's tasks** are documented in this main branch README file.

---

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for university level course material. The goal is to retrieve relevant document chunks using semantic search and generate answers using a transformer based language model (`Flan-T5-base`). Our model is trained on **University of Bamberg's** various courses `course modules`, `slides` and `previous exam questions`  from bachelor and masters degree programs. So that our model can generate and predict current semester's `exam questions`, `summarize` course content and `answer` various questions of any specific course.

## Functionality

The system supports:
- Loading multi format course documents (`.txt`, `.pdf`, `.docx`)
- Semantic chunking using sentence-level splitting
- FAISS-based vector search over sentence embeddings
- Prompt based answer generation using `Flan-T5-base`
- Log based QA evaluation with test set and performance reporting

## Project Structure

```
NLProc-Proj-M-SS25/
├── baseline/
│ ├── pipeline.py # Entry-point to run the full RAG pipeline
│ ├── retriever/
│ │ └── retriever.py # Handles chunking, embedding, FAISS indexing, and querying
│ └── generator/
│ └── generator.py # Builds prompts and generates answers using Flan-T5-base
├── evaluation/
│ ├── evaluation.py # Logs predictions and compares against test Q&A 
│ ├── logs/
│ │ └── 21-05-2025.json # JSON log of queries and outputs (auto generated)
│ └── test/
│   └── test_sample_question_answer.json # Manually created test Q&A pairs
    └── unit_test.py # Scores predictions and visualizes results
├── data/
│ └── uni-bamberg-courses/
│ └── dsg-dsam-m/ # Raw course materials: slides, module sheets, exams
├── requirements.txt # Dependencies
└── README.md # Project instructions and overview
```

## To run the project first run the requirements.txt to install necessary packages
```python 
pip install -r requirements.txt
```
```python 
python -m spacy download en_core_web_sm
```
```python 
python -m baseline.pipeline
```

# Week 5

## Let's first go through the paper `https://arxiv.org/pdf/2210.11416` where we find the ''Model Card for FLAN-T5 base''

![model card 1](https://github.com/nafees-iqbal/NLProc-Proj-M-SS25/blob/main/images/flan2_architecture.jpg?raw=true)

