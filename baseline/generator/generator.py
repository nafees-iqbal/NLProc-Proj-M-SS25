
"""
LLMs model the statistical distribution of text using model weights, i.e.
    - LLMs (like GPT) are trained on huge amounts of text.
    - They learn how likely one word is to follow another, this is called a statistical language model.
    - These learned "likelihoods/patterns" are stored in the model's weights (its memory).
    - It doesn't "memorize" facts, it learns patterns in language.
What does an LLM do?
    "A model that takes text as input (prompt) and generates text as output"

You give it a prompt like:
"Summarize the story"
It generates a response, word by word, that it thinks is most likely based on the prompt.

This is generative behavior, it creates answers, not just retrieve/copy them 

"""
from transformers import T5Tokenizer, T5ForConditionalGeneration # transformers are a type of architecture like T5, BERT
import torch

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initializes the generator with a pretrained instruction-tuned model (e.g., Flan-T5).
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    def build_prompt(self, context: str, task_input: str, mode: str = "qa", options: list = None) -> str:
        """
        Builds dynamic prompts with one-shot examples for different RAG task types.

        Parameters:
        - context (str): Retrieved context or reference text.
        - task_input (str): The user's input (question or sentence).
        - mode (str): One of "qa", "summarization", "mcq", or "classification".
        - options (list): Multiple choice options (used in mcq mode).

        Returns:
        - str: A full prompt with one-shot examples.
        """
        if mode == "qa":
            return (
                "You are an assistant for a university-level course.\n"
                "Use only the provided context to answer the question.\n"
                "If the answer is not in the context, respond with: I don't know.\n\n"
                "Example:\n"
                "Context:""\nJava EE stands for Java Platform, Enterprise Edition, which is used to develop enterprise level applications.\n"
                "Question:\n"
                "What is the full form of Java EE?\n"
                "Answer:\n"
                "Java EE full form is Java Enterprise Edition.\n\n"
                "Now use the following context to answer the question.\n"
                f"Context:\n{context}\n"
                f"Question:\n{task_input}\n"
                "Answer:"
            )

        elif mode == "summarization":
            return (
                "You are an academic assistant.\n"
                "Summarize the following content clearly and concisely.\n\n"
                "Example:\n"
                "Content:\n"
                "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.\n"
                "Summary:\n"
                "Machine learning enables computers to learn from data using statistical methods.\n\n"
                "Now summarize the following content:\n"
                f"Content:\n{context}"
            )

        elif mode == "mcq":
            option_text = '\n'.join([f"{chr(97+i)}) {opt}" for i, opt in enumerate(options)])
            return (
                "You are a quiz assistant. Use the provided context to answer the question. Choose one letter only from the given options.Always respond with a single letter (a, b, c, ...).\n\n"
                "Example:\n"
                "Context:\n"
                "Paris is the capital of France.\n"
                "Question:\n"
                "What is the capital of France?\n"
                "Options:\n"
                "a) Rome\n"
                "b) Berlin\n"
                "c) Paris\n"
                "Answer:\n"
                "c\n\n"
                "Now answer the following question:\n"
                f"Context:\n{context}\n"
                f"Question:\n{task_input}\n"
                f"Options:\n{option_text}\n"
                "Answer:"
            )


        elif mode == "classification":
            return (
                "You are a content moderation system. Use the following reference rules to decide whether the input is Offensive or Non-offensive. Only respond with one of the two categories: Offensive or Non-offensive.\n\n"
                "Example:\n"
                "Rules:\n"
                "Profanity, hate speech, and personal attacks are considered offensive.\n"
                "Input:\n"
                "You are a terrible person!\n"
                "Classification:\n"
                "Offensive\n\n"
                "Now classify the following input:\n"
                f"Rules:\n{context}\n"
                f"Input:\n{task_input}\n"
                "Classification:"
            )


        else:
            raise ValueError(f"Unknown mode: {mode}")


    def generate_answer(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generates an answer from the prompt using the Flan-T5 model.

        Parameters:
        prompt (str): Prompt containing context and question
        max_tokens (int): Maximum number of tokens in the generated answer

        Returns:
        answer (str): Generated answer from the model
        """

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True) # tensors are nothing just a 3D+ matrix
        with torch.no_grad(): # track that no memory is used for saving data
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)