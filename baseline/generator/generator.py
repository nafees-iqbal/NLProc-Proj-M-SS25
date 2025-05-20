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

def generate():