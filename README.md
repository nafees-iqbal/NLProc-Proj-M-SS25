# RAG Project – Summer Semester 2025

## Week 2

### How Vector Search Works

This below content demonstrates how **semantic search** can be performed using vector embeddings, with and without FAISS.

### What is Vector Search?

Vector search is a technique where we:
1. Convert text into vectors using a pretrained model (like `all-MiniLM-L6-v2`).
2. Store those vectors in memory or a FAISS index data structure.
3. When a user enters a query, we convert that query into a vector too.
4. Then compare that query vector to all stored vectors to find the most semantically similar text.

### Two Methods Used: 

#### 1. Manual Vector Search (without FAISS)
For small datasets, we use:
- **Cosine Similarity** (angle-based)

#### 2. FAISS-Based Vector Search

For larger datasets, we use **FAISS** (Facebook AI Similarity Search), a library for fast vector search.

- FAISS builds an optimized index data structure to store all document vectors.
- It supports efficient nearest neighbor search using L2 (Euclidean) distance or inner product.
- It is highly scalable and can handle millions of vectors efficiently.
- It can also trade off a small amount of accuracy for dramatic speed improvements if needed.

This approach is ideal when working with a large number of documents and queries, enabling real-time retrieval performance.

