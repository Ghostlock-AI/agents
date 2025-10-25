#!/usr/bin/env python3
"""
Test script for context management and RAG features.

Demonstrates:
1. File context management with deduplication
2. Token budget tracking
3. Vector store semantic search
4. Context compression
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from context import FileContextManager
from memory import VectorStore


def test_file_context_manager():
    """Test file context management."""
    print("=" * 70)
    print("FILE CONTEXT MANAGER TEST")
    print("=" * 70)
    print()

    # Create manager
    manager = FileContextManager(max_tokens=10000)
    print("✓ FileContextManager initialized")
    print()

    # Add some files
    print("-" * 70)
    print("ADDING FILES TO CONTEXT:")
    print("-" * 70)

    test_content = """
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello_world()
"""

    success, message = manager.add_file("test_file.py", test_content)
    print(f"Add test_file.py: {message}")

    # Try adding same file again
    success, message = manager.add_file("test_file.py", test_content)
    print(f"Add test_file.py again: {message}")
    print()

    # List files
    print("-" * 70)
    print("FILES IN CONTEXT:")
    print("-" * 70)

    files = manager.list_files()
    for file_state in files:
        print(f"  {file_state.filepath}")
        print(f"    Version: {file_state.version}")
        print(f"    Tokens: {file_state.token_count}")
        print(f"    References: {file_state.reference_count}")
        print()

    # Get stats
    print("-" * 70)
    print("CONTEXT STATISTICS:")
    print("-" * 70)

    stats = manager.get_stats()
    print(f"  Total files: {stats.total_files}")
    print(f"  Total tokens: {stats.total_tokens}")
    print(f"  Max tokens: {stats.max_tokens}")
    print(f"  Utilization: {stats.utilization:.1%}")
    print(f"  Near limit: {stats.is_near_limit}")
    print(f"  Critical: {stats.is_critical}")
    print()

    # Format for context
    print("-" * 70)
    print("FORMATTED CONTEXT:")
    print("-" * 70)
    formatted = manager.format_for_context()
    print(formatted[:200] + "..." if len(formatted) > 200 else formatted)
    print()

    # Clean up
    manager.clear()
    print("✓ Context cleared")
    print()


def test_vector_store():
    """Test vector store semantic search."""
    print("=" * 70)
    print("VECTOR STORE TEST")
    print("=" * 70)
    print()

    try:
        # Create vector store
        store = VectorStore(persist_directory="./data/test_chroma")
        print("✓ VectorStore initialized")
        print()

        # Add some documents
        print("-" * 70)
        print("ADDING DOCUMENTS:")
        print("-" * 70)

        docs = [
            ("auth.py", "def authenticate_user(username, password): ..."),
            ("database.py", "def connect_to_database(host, port): ..."),
            ("api.py", "def create_api_endpoint(route, handler): ..."),
        ]

        for filepath, content in docs:
            store.add_file(filepath, content)
            print(f"  Added: {filepath}")

        print()

        # Search
        print("-" * 70)
        print("SEMANTIC SEARCH:")
        print("-" * 70)

        query = "user login authentication"
        print(f"Query: '{query}'")
        print()

        results = store.search(query, n_results=2)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.filepath} (similarity: {result.similarity:.3f})")
            print(f"   {result.content[:50]}...")
            print()

        # Count
        count = store.count()
        print(f"Total documents in store: {count}")
        print()

        # Clean up
        store.clear()
        print("✓ Vector store cleared")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Note: Vector store requires ChromaDB dependencies")
        print("Run: pip install chromadb sentence-transformers")
        print()


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CONTEXT & RAG SYSTEM TEST" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    # Test file context manager
    test_file_context_manager()

    # Test vector store
    test_vector_store()

    print("=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Install dependencies: pip install tiktoken chromadb sentence-transformers")
    print("  2. Try with agent: python src/main.py")
    print("  3. Use /context commands in TUI (coming soon)")
    print()


if __name__ == "__main__":
    main()
