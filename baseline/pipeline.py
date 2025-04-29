# put the code that calls all the different components here

from retriever.retreiver import Retriever


# Create retriever instance
retriever = Retriever()

def compare_text_samples_using_cosine_similarity(retriever, folder_path="data/text-samples"):
    
    print("\nComparing text samples for similarity:")
    filenames, texts = retriever.load_document(folder_path)
    embeddings = retriever.compute_embeddings(texts)
    retriever.compare_embeddings(embeddings, filenames)

compare_text_samples_using_cosine_similarity(retriever)
