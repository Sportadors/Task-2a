from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

loader=PyPDFLoader('F://ml//Attention.pdf')
doc=loader.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=60)
document=splitter.split_documents(doc)

from langchain_community.vectorstores import FAISS


vectorstore = FAISS.from_documents(document, embeddings)

query = "What is the main innovation used in the 'Attention is All You Need' Paper?"
retrieved_docs = vectorstore.similarity_search(query, k=3)
for doc in retrieved_docs:
    print(doc.page_content)

from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

model_name="mistralai/Mistral-7B-Instruct-v0.3"
