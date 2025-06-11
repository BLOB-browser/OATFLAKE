#!/usr/bin/env python3
"""
Vector Store Flow Analysis

This script summarizes how FAISS vector stores are built and when topic stores are generated
based on the analysis of the OATFLAKE codebase.
"""

def print_vector_store_flow():
    """Print a comprehensive analysis of the vector store generation flow"""
    
    print("=" * 80)
    print("VECTOR STORE GENERATION FLOW ANALYSIS")
    print("=" * 80)
    
    print("\n1. WHEN VECTORS ARE REBUILT:")
    print("   • After URL processing completes (when URLs have been scraped)")
    print("   • When skip_vector_generation=False (default)")
    print("   • During STEP 5 of KnowledgeOrchestrator.process_knowledge()")
    print("   • Can be manually triggered via /api/rebuild-faiss-indexes endpoint")
    print("   • Via rebuild_faiss_indexes.py script")
    
    print("\n2. WHAT CONTENT GOES INTO VECTORS:")
    print("   • URL content from processed_urls.csv (scraped web pages)")
    print("   • CSV data from definitions.csv, methods.csv, etc. (reference material)")
    print("   • PDF content if PDFs are processed")
    print("   • NOT the CSV metadata itself - the actual content from URLs")
    
    print("\n3. VECTOR STORE TYPES CREATED:")
    print("   a) REFERENCE STORE (reference_store):")
    print("      - Contains CSV definitions, methods, materials")
    print("      - Built from CSV files in data folder")
    print("      - Added via rebuild_faiss_indexes.py -> add_document_types()")
    print("")
    print("   b) CONTENT STORE (content_store):")
    print("      - Contains scraped URL content")
    print("      - Built from processed URLs")
    print("      - Main store for web content")
    print("")
    print("   c) TOPIC STORES (topic_*):")
    print("      - Created from content_store representative chunks")
    print("      - Generated using clustering (K-means)")
    print("      - Based on tags/topics in document metadata")
    
    print("\n4. TOPIC STORE GENERATION PROCESS:")
    print("   • Triggered when content_store exists and has documents")
    print("   • Uses VectorStoreManager.get_representative_chunks() with K-means clustering")
    print("   • Gets 100 representative documents from content_store")
    print("   • Groups documents by tags/topics from metadata")
    print("   • Creates separate vector stores for each topic (topic_fabrication, topic_electronics, etc.)")
    print("   • Falls back to title-based topic extraction if no tags present")
    
    print("\n5. MAIN GENERATION FLOW:")
    print("   KnowledgeOrchestrator.process_knowledge()")
    print("   ├── URL Processing (scrape web content)")
    print("   ├── Content saved to temp/*.jsonl files")
    print("   ├── STEP 5: Vector Generation Phase")
    print("   │   ├── VectorGenerator.generate_vector_stores()")
    print("   │   ├── Calls rebuild_faiss_indexes.py script")
    print("   │   ├── rebuild_faiss_indexes.py:")
    print("   │   │   ├── Rebuilds reference_store from CSV files")
    print("   │   │   ├── Rebuilds content_store from documents.json")
    print("   │   │   └── Creates topic stores from content_store")
    print("   │   └── Cleanup: Delete temp/*.jsonl files")
    print("   └── Goal/Question generation")
    
    print("\n6. FILE STRUCTURE:")
    print("   data/vector_stores/default/")
    print("   ├── reference_store/")
    print("   │   ├── index.faiss          # FAISS index file")
    print("   │   ├── documents.json       # Document content & metadata")
    print("   │   └── embedding_stats.json # Statistics")
    print("   ├── content_store/")
    print("   │   ├── index.faiss")
    print("   │   ├── documents.json")
    print("   │   └── embedding_stats.json")
    print("   └── topic_*/")
    print("       ├── index.faiss")
    print("       ├── documents.json")
    print("       └── embedding_stats.json")
    
    print("\n7. KEY COMPONENTS:")
    print("   • VectorStoreManager: Main vector store operations")
    print("   • FAISSBuilder: Low-level FAISS index creation")
    print("   • VectorGenerator: High-level vector generation orchestration")
    print("   • rebuild_faiss_indexes.py: Comprehensive rebuild script")
    print("   • DataProcessor.create_vector_store(): Legacy vector creation")
    
    print("\n8. EMBEDDING MODEL:")
    print("   • Uses Ollama with 'nomic-embed-text' model")
    print("   • Batch size: 5 (optimized for Raspberry Pi)")
    print("   • Timeout: 120 seconds")
    print("   • Documents are chunked before embedding (1500 chars, 200 overlap)")
    
    print("\n9. CURRENT STATUS (Based on conversation):")
    print("   • Level 2: 52 URLs processed ✅")
    print("   • Level 3: 149 URLs pending")
    print("   • Level 4: 995 URLs pending")
    print("   • No vector stores exist yet (will be built after all URLs processed)")
    print("   • CSV contamination issue was false alarm - no vectors built yet")
    
    print("\n10. WHEN TOPIC STORES ARE CREATED:")
    print("    • After content_store is built with web content")
    print("    • Using VectorStoreManager.create_topic_stores()")
    print("    • Triggered automatically in rebuild_faiss_indexes.py")
    print("    • Uses clustering to find representative documents")
    print("    • Groups by tags/topics from resource metadata")
    print("    • Fallback topic generation from document titles")

if __name__ == "__main__":
    print_vector_store_flow()
