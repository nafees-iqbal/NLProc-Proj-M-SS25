import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the Retriever with a given embedding model.
        """
        self.model = SentenceTransformer(model_name)
        self.texts = []
        self.embeddings = None
        self.index = None

    def add_documents(self, folder_path):
        """
        Loads .txt files from the folder, encodes them, and prepares for semantic search.

        Parameters:
        folder_path: Path to the folder containing .txt files.
        """
        texts = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    texts.append(file.read())

        self.texts = texts
        self.embeddings = self.model.encode(texts)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings, dtype='float32'))

    def query(self, query_text, k=3):
        """
        Searches for the top k most relevant documents to the given query.

        Parameters:
        query_text: The input query string.
        k: Number of top matches to return.

        Returns:
        List of top k matched texts and their distances.
        """
        query_embedding = self.model.encode([query_text])
        D, I = self.index.search(np.array(query_embedding, dtype='float32'), k)
        return [self.texts[i] for i in I[0]], D[0]

    def save(self, path="retriever_index"):
        """
        Saves the FAISS index and associated document texts to disk.

        Parameters:
        path: Base filename to save the index and text data.
        """
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, path="retriever_index"):
        """
        Loads the FAISS index and associated document texts from disk.

        Parameters:
        path (str): Base filename to load the index and text data from.
        """
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_texts.pkl", "rb") as f:
            self.texts = pickle.load(f)
