"""
RAG Pipeline - Document Loading, Chunking, and Retrieval
"""

import re
from typing import List, Dict
import pandas as pd


class DocumentProcessor:
    """Process and chunk documents for RAG."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_from_csv(self, filepath: str, text_column: str = "text") -> List[str]:
        """Load documents from CSV."""
        df = pd.read_csv(filepath)
        return df[text_column].dropna().tolist()

    def load_from_txt(self, filepath: str) -> List[str]:
        """Load from text file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_text(doc))
        return all_chunks


class RAGRetriever:
    """Retrieve relevant context for queries."""

    def __init__(self, vector_store, top_k: int = 3):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> str:
        """Retrieve relevant context for a query."""
        results = self.vector_store.search(query, k=self.top_k)

        context_parts = []
        for doc, score, _ in results:
            context_parts.append(doc)

        return "\n\n".join(context_parts)

    def create_prompt(self, query: str, system_prompt: str = None) -> str:
        """Create a prompt with retrieved context."""
        context = self.retrieve(query)

        if system_prompt is None:
            system_prompt = "Answer the question based on the context provided. Be concise and accurate."

        prompt = f"""Context:
{context}

{system_prompt}

Question: {query}
Answer:"""

        return prompt
