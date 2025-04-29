# retriever.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=200):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.index = None
        self.chunks = []

    def load_document(self, folder_path):
        texts = []
        filenames = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    texts.append(file.read())
                    filenames.append(filename)
        return filenames, texts
    
    def compute_embeddings(self, texts):
        return self.model.encode(texts) # convert text into vector embeddings
    
    def compare_embeddings(self, embeddings, filenames):
        similarity_matrix = cosine_similarity(embeddings) # make a 2D matrix for pairwise cosine similarity
        print("\nCosine Similarity Matrix:")
        print("".ljust(20), end='')
        for name in filenames:
            print(name.ljust(20), end='')
        print()
        for i, row in enumerate(similarity_matrix):
            print(filenames[i].ljust(20), end='')
            for val in row:
                print(f"{val:.2f}".ljust(20), end='')
            print()
    
    def retrieve(self, query, k=3):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        retrieved_chunks = [self.chunks[i] for i in I[0]]
        return retrieved_chunks
