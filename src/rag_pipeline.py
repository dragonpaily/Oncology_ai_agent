# In src/rag_pipeline.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def build_retriever(papers_path: str):
    """
    Builds a powerful ensemble retriever from PDF documents.
    This function combines a keyword-based BM25 retriever and a semantic vector retriever.
    """
    print("📚 Building RAG knowledge base...")
    
    # 1. Load the documents
    loader = DirectoryLoader(
        papers_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()
    
    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)
    
    # 3. Create the BM25 (keyword) retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3 # Retrieve top 3 results
    
    # 4. Create the semantic vector retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 5. Create the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5] # Give equal weight to keyword and semantic search
    )
    
    print("✅ RAG knowledge base ready (using Ensemble Retriever).")
    return ensemble_retriever