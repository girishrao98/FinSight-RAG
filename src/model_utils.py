import os
from typing import List, Optional

from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def build_vectorstore(documents: List[str], index_path: Optional[str] = None) -> FAISS:
    """Build a FAISS vector store from document text."""
    embeddings = OpenAIEmbeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    if index_path:
        vectorstore.save_local(index_path)
    return vectorstore


def load_vectorstore(index_path: str) -> FAISS:
    """Load a FAISS vector store from disk."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(index_path, embeddings)


def create_rag_chain(vectorstore: FAISS, openai_api_key: Optional[str] = None) -> RetrievalQA:
    """Create a retrieval-augmented generation chain."""
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = OpenAI(temperature=0.2)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
