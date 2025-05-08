# RAG Project – Summer Semester 2025. 
## Team NNN
### Nabanita Bhowmick
### Nafees Iqbal
### Naznin Ahmed

---

## To run the project first run the requirements.txt to install necessary packages
**pip install -r requirements.txt**

---

## Branch Info

All previous week's work and corresponding README files can be found in their respective branches.  
In the `main` branch, only the **current week's tasks** are documented in this main branch README file.

---

# Week 3

## Chunking strategies

![Chunking strategies diagram 1](https://github.com/nafees-iqbal/NLProc-Proj-M-SS25/blob/main/images/1_VhFr2tr_FbTjzNyNv5DjWw.png?raw=true)

For text splitting or chunking, there are several approaches, inspired by  [this tutorial](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)

- Level 1: Character Splitting - Simple static character chunks of data
- Level 2: Recursive Character Text Splitting - Recursive chunking based on a list of separators
- Level 3: Document Specific Splitting - Various chunking methods for different document types (PDF, Python, Markdown)
- Level 4: Semantic Splitting - Embedding walk based chunking
- Level 5: Agentic Splitting - Experimental method of splitting text with an agent-like system. Good for if you believe that token cost will trend to $0.00
- *Bonus Level:* Alternative Representation Chunking + Indexing - Derivative representations of your raw text that will aid in retrieval and indexing

![Chunking strategies diagram](https://github.com/nafees-iqbal/NLProc-Proj-M-SS25/blob/main/images/92c70184-ba0f-4877-9a55-e4add0e311ad_870x1116.gif?raw=true)

## For example, the below image demonstrating how character spilling actually works:

![Chunking strategies diagram](https://github.com/nafees-iqbal/NLProc-Proj-M-SS25/blob/main/images/1_sBEoJ2xomZl77X6wUmdOlw.png?raw=true)


---

## Chunking Method Chosen for This Project

We have chosen **Level 4: Semantic Chunking** as our strategy for this project.

### Why Semantic Chunking?

Semantic chunking splits text based on **sentence boundaries and contextual meaning**, rather than arbitrary character splitting. This allows each chunk to remain **coherent, meaningful, and context rich**, which is critical for:

- **Retrieval accuracy**: More relevant chunks are returned during similarity search.
- **LLM performance**: Better prompts = better generated answers.

![semantic chunking strategies diagram](https://github.com/nafees-iqbal/NLProc-Proj-M-SS25/blob/main/images/images/semantic-chunking.png?raw=true)

---

### How It Works in Our Project

- We split the text into **sentences** using `spaCy`.
- Sentences are grouped into chunks of around 200 tokens.
- We apply **50 token overlaps** between chunks to preserve context.

---


---