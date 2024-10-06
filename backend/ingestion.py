import os
from dotenv import load_dotenv
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME","chatbot")

loader = TextLoader("files/sample.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=140,
    chunk_overlap=20
)
chunks = text_splitter.split_documents(docs)
print(chunks)


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                          task_type="retrieval_document",
                                          google_api_key=GOOGLE_API_KEY
                                          )

sample_embedded = embeddings.embed_query(chunks[0].page_content)
print("Sample Embedding")
print(sample_embedded)
size = len(sample_embedded)
print(f"Embedding size is {size}")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

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