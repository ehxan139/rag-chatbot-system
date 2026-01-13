"""
RAG Chatbot System - Vector Store with Embedding and Retrieval

Supports multiple vector databases (FAISS, ChromaDB, Pinecone) for production use.
"""

import numpy as np
from typing import List, Tuple, Optional
import faiss
from sentence_transformers import SentenceTransformer
import pickle


class VectorStore:
    """
    Vector store for document embeddings and similarity search.
    Uses FAISS for efficient nearest neighbor search.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", dimension: int = 384):
        """
        Initialize vector store.

        Args:
            embedding_model: HuggingFace model name for embeddings
            dimension: Embedding dimension
        """
        self.model = SentenceTransformer(embedding_model)
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []

    def add_documents(self, texts: List[str], metadata: Optional[List[dict]] = None):
        """
        Add documents to the vector store.

        Args:
            texts: List of document texts
            metadata: Optional metadata for each document
        """
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(texts)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, score, metadata) tuples
        """
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    float(dist),
                    self.metadata[idx]
                ))

        return results

    def save(self, path: str):
        """Save vector store to disk."""
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/documents.pkl", 'wb') as f:
            pickle.dump({'documents': self.documents, 'metadata': self.metadata}, f)

    def load(self, path: str):
        """Load vector store from disk."""
        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/documents.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
