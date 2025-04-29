# RAG Project – Summer Semester 2025

# Week 2
## How Vector Search Works

This project demonstrates how **semantic search** can be performed using **vector embeddings** — both with and without FAISS.

---

### What is Vector Search?

Vector search is a technique where we:
1. Convert text into **dense vectors** using a pretrained model (like `all-MiniLM-L6-v2`).
2. Store those vectors in memory (or a FAISS index).
3. When a user enters a query, we convert it to a vector too.
4. We compare that query vector to all stored vectors to find the most **semantically similar** text.

---

### Two Methods Used in This Project

#### 1. Manual Vector Search (without FAISS)
For small datasets, we use:
- **Cosine Similarity** (angle-based)
- **Euclidean Distance** (distance-based)

