import os
from dotenv import load_dotenv
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings

load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME","chatbot")


file_name = input("Enter the file name to be added to Knowledge Base: ")
loader = TextLoader(f"files/{file_name}")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=140,
    chunk_overlap=20
)
chunks = text_splitter.split_documents(docs)
print(chunks)


embeddings = OllamaEmbeddings(model='mistral')

sample_embedded = embeddings.embed_query(chunks[0])
print("Sample Embedding")
print(sample_embedded)
size = len(sample_embedded)
print(f"Embedding size is {size}")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def new_collection():
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
    )

    print(client.get_collections())

    qdrant = Qdrant.from_documents(
        chunks,
        embedding=embeddings, 
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        prefer_grpc=True,
    )
    print(f"{COLLECTION_NAME} created and data Injected successfully.")

def add_data_to_collection():

    client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
    qdrant = Qdrant(
        embeddings=embeddings, 
        client=client,
        collection_name=COLLECTION_NAME
    )
    qdrant.add_documents(chunks)
    print(f"Data Injected successfully in {COLLECTION_NAME}.")

options = int(input("Type 1 for collection creation and 2 for ingestion: "))
if options == 1:
    new_collection()
elif options == 2:
    add_data_to_collection()
else:
    print("Invalid Input")