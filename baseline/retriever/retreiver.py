import os
import numpy as np
import faiss
import pickle
import spacy
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the Retriever with a given embedding model.
        Loads spaCy for sentence-based semantic chunking.
        """
        self.model = SentenceTransformer(model_name)
        self.texts = []
        self.embeddings = None
        self.index = None
        self.nlp = spacy.load("en_core_web_sm")

    def semantic_chunk(self, text, max_tokens=200, overlap=50):
        """
        Splits the input text into semantically coherent chunks based on sentence boundaries.

        This strategy improves information retrieval in RAG systems
        by preserving the meaning and context within each chunk. Unlike fixed size text splitting,
        semantic chunking reduces the chance of splitting mid sentence or breaking logical structure.

        Parameters:
        - text (str): Raw text to be split.
        - max_tokens (int): Maximum number of token per chunk.
        - overlap (int): Number of tokens repeated between chunks.

        Returns:
        - List[str]: List of semantic chunks.

        Example output:

        [
            "Sentence 1. Sentence 2. Sentence 3.",
            "Sentence 3. Sentence 4. Sentence 5.",
            "Sentence 5. Sentence 6. Sentence 7.",
        ]

        """
        doc = self.nlp(text) # Used spaCy to split the input text into full sentences.
        sentences = [sent.text for sent in doc.sents]

        chunks = []
        current_chunk = []
        current_len = 0 # how many tokens are currently in current_chunk

        for sentence in sentences:
            sent_len = len(self.nlp(sentence))
            if current_len + sent_len <= max_tokens:
                current_chunk.append(sentence)
                current_len += sent_len
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:] + [sentence]
                current_len = sum(len(self.nlp(s)) for s in current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def add_documents(self, folder_path):
        """
        Loads .txt files from a folder, applies semantic chunking, encodes them into embeddings,
        and builds a FAISS index for similarity search.

        This function uses sentence based chunking to ensure that each chunk is meaningful. It enhances both retrieval relevance and LLM generation quality in downstream tasks.

        Parameters:
        - folder_path (str): Path to the folder containing .txt files.
        """
        all_chunks = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    chunks = self.semantic_chunk(text)
                    all_chunks.extend(chunks)

        self.texts = all_chunks
        self.embeddings = self.model.encode(all_chunks)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings, dtype='float32'))

    def query(self, query_text, k=3):
        """
        Searches for the top-k most relevant chunks to the given query.

        Parameters:
        - query_text (str): The user query in plain language.
        - k (int): Number of top matches to return.

        Returns:
        - List[str]: Top k matched text chunks.
        - List[float]: Corresponding distances from the query vector.
        """
        query_embedding = self.model.encode([query_text])
        D, I = self.index.search(np.array(query_embedding, dtype='float32'), k)
        return [self.texts[i] for i in I[0]], D[0]

    def save(self, path="retriever_index"):
        """
        Saves the FAISS index and the chunked texts to disk for reuse.

        Parameters:
        - path (str): path to save index and text data.

        .pkl file is used for storing and loading python's object like lists, dictionaries etc.
        """
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_texts.pkl", "wb") as faiss:
            pickle.dump(self.texts, faiss)

    def load(self, path="retriever_index"):
        """
        Loads a previously saved FAISS index and associated text chunks.

        Parameters:
        - path (str): Base path (without extension) from which to load index and text data.
        """
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_texts.pkl", "rb") as f:
            self.texts = pickle.load(f)
