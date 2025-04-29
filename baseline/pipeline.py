# put the code that calls all the different components here

from retriever.retreiver import Retriever


# Create retriever instance
retriever = Retriever()

folder_path="data/text-samples"
filenames, texts = retriever.load_document(folder_path)
embeddings = retriever.compute_embeddings(texts)

def compare_text_samples_using_cosine_similarity(retriever, folder_path="data/text-samples"):

    print("\nComparing text samples for similarity:")
    retriever.compare_embeddings(embeddings, filenames)
    retriever.visualize_embeddings_pca(embeddings, filenames)

def test_query_variants_with_faiss():
    retriever.build_faiss_index()

    query_variants = [
        "a princess who dances with magical shoes",
        "story about dancing princesses",
        "princesses disappear at night to dance",
        "mystery of girls with worn-out shoes",
        "where do the twelve princesses go every night?"
    ]

    for query in query_variants:
        results, distances = retriever.search_faiss(query, k=3)
        print(" "*80)
        print(f"\nQuery: {query}")
        print("\nTop 3 matches from FAISS:")
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (distance={distances[i]:.4f}):")
            print(result[:300] + '...')


#compare_text_samples_using_cosine_similarity(retriever)
test_query_variants_with_faiss()
