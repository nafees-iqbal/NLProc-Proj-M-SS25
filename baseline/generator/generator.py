
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

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AutoTokenizer, AutoModelForQuestionAnswering,
    GPT2Tokenizer, GPT2LMHeadModel
)


class Generator:
    def __init__(self):
        self.device = torch.device("cpu")

        self.qa_tokenizer = AutoTokenizer.from_pretrained("deepset/tinyroberta-squad2")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/tinyroberta-squad2").to(self.device)

        self.summ_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.summ_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(self.device)

        self.mcq_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.mcq_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.mcq_tokenizer.pad_token = self.mcq_tokenizer.eos_token

        self.classifier_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.classifier_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(self.device)

    def build_prompt(self, context: str, task_input: str, mode: str = "qa", options: list = None) -> str:
        if mode == "qa":
            return (
                "You are a helpful assistant for a university-level course.\n"
                "Based on the provided context, answer the question concisely.\n"
                "If the answer is clearly not present in the context, say: I don't know.\n\n"
                f"Context:\n{context}\n"
                f"Question:\n{task_input}\n"
                "Answer:"
            )

        elif mode == "summarization":
            return (
                "You are an expert summarizer.\n"
                "Rewrite the following explanation into a concise, formal summary using factual, objective tone. "
                "Avoid phrases like 'learn' or 'understand'. Focus on the core technical content.\n\n"
                f"Content:\n{context}\n\n"
                "Summary:"
            )

        elif mode == "mcq":
            option_text = '\n'.join([f"{chr(97+i)}) {opt}" for i, opt in enumerate(options)])
            return (
                "You are a quiz assistant. Use the provided context to answer the question. Choose one letter only from the given options.\n\n"
                f"Context:\n{context}\n"
                f"Question:\n{task_input}\n"
                f"Options:\n{option_text}\n"
                "Answer:"
            )

        elif mode == "classification":
            return f"Classify the following sentence:\n{task_input}"

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate_answer(self, prompt: str, mode: str = "qa", options: list = None, max_tokens: int = 300) -> str:
        if mode == "qa":
            inputs = self.summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.summ_model.generate(
                    **inputs,
                    max_new_tokens=60,
                    num_beams=4,
                    early_stopping=True
                )
            return self.summ_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        elif mode == "summarization":
            inputs = self.summ_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                outputs = self.summ_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    min_length=60,  # force longer output
                    num_beams=5,  # improve coherence
                    length_penalty=2.0,  # discourage very short answers
                    repetition_penalty=1.2,  # avoid "Learn what a..." loops
                    no_repeat_ngram_size=3,  # avoid repeating phrases
                    early_stopping=True
                )

            return self.summ_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        elif mode == "mcq":
            inputs = self.mcq_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.mcq_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    num_beams=4,
                    early_stopping=True
                )
            result = self.mcq_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            valid_letters = [chr(97+i) for i in range(len(options))]
            for letter in valid_letters:
                if letter in result:
                    return letter
            return "invalid"

        elif mode == "classification":
            inputs = self.classifier_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.classifier_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs).item()
            return "Offensive" if label == 1 else "Non-offensive"

        else:
            return "Unsupported mode"


