"""
FastAPI Server for RAG Chatbot System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorstore.store import VectorStore
from rag import RAGRetriever

app = FastAPI(
    title="RAG Chatbot API",
    description="Production RAG system with vector search",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
vector_store = None
retriever = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    system_prompt: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    context: str
    prompt: str


class AddDocumentsRequest(BaseModel):
    documents: List[str]
    metadata: Optional[List[dict]] = None


@app.on_event("startup")
async def startup():
    """Initialize vector store on startup."""
    global vector_store, retriever
    
    vector_store = VectorStore(embedding_model="all-MiniLM-L6-v2")
    retriever = RAGRetriever(vector_store, top_k=3)
    
    # Try to load existing index
    try:
        vector_store.load("./data/vectorstore")
    except:
        pass


@app.get("/")
def root():
    return {
        "service": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": ["/query", "/add_documents", "/health"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_indexed": len(vector_store.documents) if vector_store else 0
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    if not vector_store or len(vector_store.documents) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed")
    
    retriever.top_k = request.top_k
    context = retriever.retrieve(request.query)
    prompt = retriever.create_prompt(request.query, request.system_prompt)
    
    return QueryResponse(
        query=request.query,
        context=context,
        prompt=prompt
    )


@app.post("/add_documents")
async def add_documents(request: AddDocumentsRequest):
    """Add documents to the vector store."""
    vector_store.add_documents(request.documents, request.metadata)
    
    # Save updated index
    os.makedirs("./data/vectorstore", exist_ok=True)
    vector_store.save("./data/vectorstore")
    
    return {
        "status": "success",
        "documents_added": len(request.documents),
        "total_documents": len(vector_store.documents)
    }


@app.post("/search")
async def search(query: str, k: int = 5):
    """Search for similar documents."""
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    results = vector_store.search(query, k=k)
    
    return {
        "query": query,
        "results": [
            {"text": doc, "score": score, "metadata": meta}
            for doc, score, meta in results
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
