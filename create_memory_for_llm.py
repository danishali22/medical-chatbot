import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
# print("Length of PDF Pages: ", len(documents))

# Step 2: Create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
# print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Store embeddings in Qdrant without deepcopy issues
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT_URL = os.getenv("QDRANT_ENDPOINT_URL")

qdrant_client = QdrantClient(
    url=QDRANT_ENDPOINT_URL,
    api_key=QDRANT_API_KEY
)

# Choose a collection name for your data
COLLECTION_NAME = "medibot"

# Ensure the collection exists
embedding_dimension = 384
existing_collections = qdrant_client.get_collections().collections
collection_names = [col.name for col in existing_collections]
if COLLECTION_NAME not in collection_names:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
    )

# create an instance of Qdrant and then add the documents.
db = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model
)

# Append documents (chunks) to the collection
db.add_documents(text_chunks)

print(f"Uploaded {len(text_chunks)} chunks to Qdrant collection '{COLLECTION_NAME}'.")
