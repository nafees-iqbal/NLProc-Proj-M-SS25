#implement your generator here

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
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initializes the generator with a pretrained instruction-tuned model (e.g., Flan-T5).
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    def build_prompt(self, context: str, task: str, example_context: str = None, example_output: str = None) -> str:
        
        return f"""
                You are an assistant for a university level course.

                Use only the information from the context. If the answer is not present in the context, say "I don't know." 
                Respond with one exam-style question, clearly phrased.

                Example:
                Context:
                A distributed system is a collection of independent computers that appear to the user as a single system.

                Question:
                What is a distributed system?

                ---

                Now use the following context to generate a question.

                Context:
                {context}

                Question:
            """                                                                
                                                            


    def generate_answer(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generates an answer from the prompt using the Flan-T5 model.

        Parameters:
        prompt (str): Prompt containing context and question
        max_tokens (int): Maximum number of tokens in the generated answer

        Returns:
        answer (str): Generated answer from the model
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)