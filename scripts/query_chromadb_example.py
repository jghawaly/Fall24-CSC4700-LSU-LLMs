import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# load environmental variables
load_dotenv('../.env')

# Establish client
chroma_client = chromadb.PersistentClient(path="../data/my_chromadb")

# Use ada-002 for embedding
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

# get the documents collection, with a provided embedding function
collection = chroma_client.get_collection(
    name="documents",
    embedding_function=openai_ef
)

# get the most semantically similar document
results = collection.query(query_texts=["Who is the second president?"], n_results=1)
print("Unfiltered Semantic Search: ", results)

# do the same, but now filter the results by title
results = collection.query(
    query_texts=["Who was the first leader?"],
    n_results=1,
    where={"category": "English kings"}
)
print("Filtered Semantic Search: ", results)
