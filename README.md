# RAG Project – Summer Semester 2025

## Week 2

### How Vector Search Works

This below content demonstrates how **semantic search** can be performed using vector embeddings, with and without FAISS.

### What is Vector Search?

Vector search is a technique where we:
1. Convert text into vectors using a pretrained model (like `all-MiniLM-L6-v2`).
2. Store those vectors in memory or a FAISS index data structure.
3. When a user enters a query, we also convert that query into a vector too.
4. Then compare that query vector to all stored vectors, so that we can find the most semantically similar text.

### Two Methods Used: 

#### 1. Manual Vector Search (without FAISS)
For smaller datasets, we use:
- **Cosine Similarity** (angle-based)

**How it works in this project:**

After computing embeddings for all text documents inside data/text-samples, we compare them using cosine similarity.

(The below code snippet from retreiver.py)

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute pairwise cosine similarity between text document embeddings
similarity_matrix = cosine_similarity(embeddings)

for i, row in enumerate(similarity_matrix):
    for j, score in enumerate(row):
        print(f"Similarity between Document {i} and Document {j}: {score:.2f}") 

```

#### 2. FAISS-Based Vector Search

For larger datasets, we use **FAISS** (Facebook AI Similarity Search), a library for fast vector search.

- FAISS builds an optimized index data structure to store all data vectors.
- It supports efficient nearest neighbor search using L2 (Euclidean) distance.
- It is highly scalable and can handle millions of vectors efficiently.
- It can also trade off a small amount of accuracy for search speed improvements.

**How faiss works in this project:**

(The below code snippet from retreiver.py)

```python
import faiss
import numpy as np

# Build FAISS index data structure
dimension = embeddings.shape[1] 
index = faiss.IndexFlatL2(dimension)  
index.add(np.array(embeddings, dtype='float32'))  

query_embedding = model.encode([query_text])

D, I = index.search(np.array(query_embedding, dtype='float32'), k=3)

# show top k search results
for idx, distance in zip(I[0], D[0]):
    print(f"Retrieved Document {idx} with distance: {distance:.4f}")

```

