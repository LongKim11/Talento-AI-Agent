from dotenv import load_dotenv
import os

from uuid import uuid4

from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# def prepare_chunks():
#     cwd = os.getcwd()

#     file_name = "Talento-Info.pdf"
#     file_path = cwd + "/" + file_name

#     loader = PyPDFLoader(file_path)
#     docs = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = text_splitter.split_documents(docs)

#     return chunks

pinecone_api_key = os.getenv("PINECONE_API_KEY")

index_name = "talento-info"

pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# chunks = prepare_chunks()

# uuids = [str(uuid4()) for _ in range(len(chunks))]

# vector_store.add_documents(chunks, ids=uuids)

# results = vector_store.similarity_search_with_score(
#     "Which plans is suitable for normal use", k=3,
# )

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# results = vector_store.similarity_search(
#     "What is Talento Network", k=3,
# )

# formatted = format_docs(results)

# for d in format_docs(results):
#     print("-----DOCUMENT-----")
#     print(d)
#     print("------------------")