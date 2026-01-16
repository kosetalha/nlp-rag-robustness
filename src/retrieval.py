"""
Vector store retrieval module using ChromaDB.
Handles document indexing and similarity search.
"""
import sys
import shutil
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL

# Detect Colab environment
IN_COLAB = 'google.colab' in sys.modules


class RAGRetriever:
    """
    ChromaDB-based vector store for document retrieval.
    Uses EphemeralClient on Colab (avoids read-only issues), PersistentClient locally.
    """
    
    def __init__(self, collection_name: str = "rag_benchmark", reset_db: bool = True):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the ChromaDB collection
            reset_db: If True, clears existing database for fresh experiments
        """
        self.collection_name = collection_name
        
        # Use sentence-transformers for embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        if IN_COLAB:
            # Use in-memory client on Colab (avoids read-only filesystem issues)
            self.client = chromadb.EphemeralClient()
            print("🗑️ Using in-memory ChromaDB (Colab mode)")
        else:
            # Use persistent storage locally
            if reset_db and CHROMA_DB_DIR.exists():
                try:
                    shutil.rmtree(CHROMA_DB_DIR)
                    print("🗑️ Cleared previous vector database")
                except Exception as e:
                    print(f"⚠️ Could not clear old DB: {e}")
            
            CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        
        print(f"✅ ChromaDB initialized (collection: {collection_name})")
    
    def index_documents(
        self, 
        documents: List[str], 
        ids: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            ids: Unique IDs for each document
            metadatas: Optional metadata dictionaries
        """
        print(f"📚 Indexing {len(documents)} documents...")
        
        # ChromaDB requires non-empty metadata dicts
        if metadatas is None:
            metadatas = [{"indexed": "true"} for _ in documents]
        else:
            # Ensure no empty dicts (ChromaDB validation fails on empty)
            metadatas = [m if m else {"indexed": "true"} for m in metadatas]
        
        # Batch processing for stability
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end],
                ids=ids[i:end],
                metadatas=metadatas[i:end]
            )
        
        print(f"✅ Indexed {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve most relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of document texts
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        if results['documents']:
            return results['documents'][0]
        return []
    
    def retrieve_with_metadata(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve documents with their metadata.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with documents, metadatas, and distances
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
        print("🗑️ Collection cleared")
    
    @property
    def count(self) -> int:
        """Return number of documents in collection."""
        return self.collection.count()
