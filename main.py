# Flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS

# Web scraping imports
import requests
import bs4

# Ollama imports
import ollama

# Chroma imports
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions


# Set up the Flask app
app = Flask(__name__)
CORS(app)

# Set up chromadb
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")


def error_response(message: str, status_code: int):
    """Return a JSON error response with a given message and HTTP status code."""
    return jsonify({"error": message}), status_code


def get_text_from_url(url: str) -> str:
    """
    Get the text from a URL
    """
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    text = " ".join([p.text for p in soup.find_all("p")])

    return text


@app.route("/documents", methods=["POST"])
def add_documents():
    """
    Add a list of documents (represented as a list of URLs) to the vector database
    """
    urls = request.get_json()
    if not urls:
        return error_response("No data provided", 400)
    if not isinstance(urls, list):
        return error_response("Expected a list of URLs", 400)

    all_docs = collection.get()
    all_ids = all_docs["ids"]
    if all_ids:
        collection.delete(ids=all_ids)

    # TASK 2: Extract the text from each URL
    documents = []

    ##
    for url in urls:
        text = get_text_from_url(url)
        documents.append(text)
    ##

    # TASK 3: Add the documents to the vector database

    ##
    collection.add(
        documents=documents,
        ids=urls,
    )
    ##

    return jsonify({"message": "Documents added successfully"}), 200


@app.route("/query", methods=["GET"])
def query_documents():
    """
    1. Use the vector database to find the most similar documents to a given query
    2. Get the most similar documents from the database
    3. Provide the query along with the most similar documents as context to the LLM
    4. Return the response from the LLM
    """
    query = request.args.get("query")
    if not query:
        return error_response("No query provided", 400)

    # TASK 4: Query the collection (the vector database) for the document most similar to the query
    results = None

    ##
    results = collection.query(query_texts=[query], n_results=1)
    ##

    # TASK 5: Convert the results to a format that can be passed to the LLM as context
    context = ""

    ##
    if results:
        context = "\n".join(results["documents"][0])
    ##

    PROMPT = query
    if context:
        PROMPT = f"Given the following documents, generate a summary of the query:\n{context}\nQuery: {query}"

    # TASK 1: Pass the PROMPT to the LLM and return the response
    response = "PLACEHOLDER"

    ##
    response = ollama.generate(model="llama3.2", prompt=PROMPT)["response"]
    ##

    return jsonify({"response": response, "message": "Query successful"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)
