import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import ollama

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

collection = chroma_client.create_collection(name="my_collection")

# collection._embedding_function = embedding_functions.OllamaEmbeddingFunction(
#     url="http://localhost:11434/api/embeddings",
#     model_name="llama3.2",
# )

collection.add(
    documents=[
        "Brown University is located in Providence",
        "Harvard is located in Boston",
        "Rust is a systems programming language",
        "f(x) = 2x + 3 is a cool math function",
    ],
    ids=["id1", "id2", "id3", "id4"],
)


question = "What is Rust programming language?"


results = collection.query(
    query_texts=[question],  # Chroma will embed this for you
    n_results=4,  # how many results to return
)

print(results)

context = "\n".join(results["documents"][0])


PROMPT = f"""Answer the question based only on the following context:
{context}

Question: {question}
"""

print(PROMPT)


print(ollama.generate(model="llama3.2", prompt=PROMPT))

#
#
# #
# ollama_ef = embedding_functions.OllamaEmbeddingFunction(
#     url="http://localhost:11434/api/embeddings",
#     model_name="llama2",
# )
#
# embeddings = ollama_ef(["This is my first text to embed", "This is my second document"])
# #
# print(embeddings)
