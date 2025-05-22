# RAG Project – Summer Semester 2025. 
## Team NNN
### Nabanita Bhowmik Shathi
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

## Let's Now jump into the demo
## Demo: End-to-End Example

**Query:**  
`What is the full form of Java EE?`

**Retrieved Chunk:**  
`Java EE stands for Java Platform, Enterprise Edition...`

**Prompt Format:**

```python
Context:
Java EE stands for Java Platform, Enterprise Edition...

Question:
What is the full form of Java EE?

Answer:
```


**Generated Answer:**  
`Java EE full form is Java Enterprise Edition.`

**Evaluation:** Match with 85% accuracy (Score: 0.85)

---

## Evaluation Summary

- All queries and generated answers are logged to JSON file with generated date filename.
- The `unit_test.py` script checks for semantic similarity using embeddings.
- A pie chart visualizes matched vs unmatched answers.

---

## What Works Well

- Semantic chunking preserves contextual integrity.
- FAISS provides accurate retrieval across materials.
- Prompt based Flan-T5-base inference performs well on QA.
- Logging + test comparison is modular and automated.

---

## What Needs Improvement

- Summarization style prompts still often return questions.
- Answers with varied word may show low similarity using lexical matching.
- Better performance for open-ended queries is needed.

---

## Specialization Proposal (Phase 2)

### Goal:
Improve the system to generate real university exam questions and answers based on past exams, course slides, and module goals.

### Changes:
- Replace one shot prompts with few shot dynamic examples.
- Finevtune Flan-T5-base or switch to domain adapted models (like BioBERT for medicine).
- Build UI with Streamlit for interactive testing.

### Evaluation:
- Precision on test Q&A.
- Human judgment: exam style validity.
- score evaluated by all-MiniLM-L6-v2's SentenceTransformer: Is the answer actually in retrieved context?
![model card 2](https://github.com/nafees-iqbal/NLProc-Proj-M-SS25/blob/main/images/evaluation-score-week-5.png?raw=true)
![model card 3](https://github.com/nafees-iqbal/NLProc-Proj-M-SS25/blob/main/images/pie-chart-of-week-5-evaluation.png?raw=true)

### Data Plan:
- Extend with new course folders (like MOBI, SOA).
- Add scraped academic QA data for generalization.
- Clean and align past exam sets.


