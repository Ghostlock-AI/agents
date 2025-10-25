"""
Vector Store for Semantic Search

Uses ChromaDB for storing and retrieving file contents and code snippets
based on semantic similarity.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings


class SearchResult(BaseModel):
    """Result from semantic search."""

    content: str
    filepath: str
    distance: float
    metadata: Dict[str, Any] = {}

    @property
    def similarity(self) -> float:
        """Convert distance to similarity score (0-1)."""
        # ChromaDB uses L2 distance, convert to similarity
        return 1.0 / (1.0 + self.distance)


class VectorStore:
    """Vector store for semantic search over files and code."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "file_contexts"
    ):
        """
        Initialize vector store.

        Args:
            persist_directory: Directory to persist the database (default: ./data/chroma)
            collection_name: Name of the collection to use
        """
        # Set up persist directory
        if persist_directory is None:
            persist_directory = os.path.join(os.getcwd(), "data", "chroma")

        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

    def add_file(
        self,
        filepath: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add or update a file in the vector store.

        Args:
            filepath: Absolute path to the file
            content: File content
            metadata: Additional metadata to store
        """
        # Prepare metadata
        file_metadata = metadata or {}
        file_metadata.update({
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "extension": os.path.splitext(filepath)[1],
        })

        # Use filepath as ID for easy updates
        doc_id = self._get_doc_id(filepath)

        # Check if already exists
        try:
            existing = self.collection.get(ids=[doc_id])
            if existing and existing['ids']:
                # Update existing
                self.collection.update(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[file_metadata]
                )
            else:
                # Add new
                self.collection.add(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[file_metadata]
                )
        except Exception:
            # If get fails, try add
            self.collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[file_metadata]
            )

    def remove_file(self, filepath: str) -> None:
        """
        Remove a file from the vector store.

        Args:
            filepath: Absolute path to the file
        """
        doc_id = self._get_doc_id(filepath)

        try:
            self.collection.delete(ids=[doc_id])
        except Exception:
            pass  # File not in store

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for semantically similar content.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"extension": ".py"})

        Returns:
            List of SearchResult objects
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata
        )

        search_results = []

        if not results or not results['documents'] or not results['documents'][0]:
            return search_results

        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []

        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0

            search_results.append(
                SearchResult(
                    content=doc,
                    filepath=metadata.get('filepath', 'unknown'),
                    distance=distance,
                    metadata=metadata
                )
            )

        return search_results

    def search_by_filepath(
        self,
        filepath: str,
        n_results: int = 5
    ) -> List[SearchResult]:
        """
        Find similar files to a given file.

        Args:
            filepath: Path to the reference file
            n_results: Number of similar files to return

        Returns:
            List of SearchResult objects
        """
        doc_id = self._get_doc_id(filepath)

        try:
            # Get the document
            result = self.collection.get(ids=[doc_id])

            if not result or not result['documents']:
                return []

            content = result['documents'][0]

            # Search for similar content (excluding the original)
            all_results = self.search(content, n_results=n_results + 1)

            # Filter out the original file
            return [
                r for r in all_results
                if r.filepath != filepath
            ][:n_results]

        except Exception:
            return []

    def list_files(self) -> List[str]:
        """
        List all files in the vector store.

        Returns:
            List of file paths
        """
        try:
            result = self.collection.get()

            if not result or not result['metadatas']:
                return []

            return [
                meta.get('filepath', '')
                for meta in result['metadatas']
                if meta.get('filepath')
            ]

        except Exception:
            return []

    def count(self) -> int:
        """
        Get count of documents in store.

        Returns:
            Number of documents
        """
        try:
            return self.collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """Clear all documents from the store."""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            pass

    @staticmethod
    def _get_doc_id(filepath: str) -> str:
        """
        Get document ID for a filepath.

        Args:
            filepath: File path

        Returns:
            Document ID (hash of filepath for uniqueness)
        """
        import hashlib
        return hashlib.md5(filepath.encode()).hexdigest()
