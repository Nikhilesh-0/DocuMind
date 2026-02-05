"""
Embeddings & Vector Store Module - Knowledge Base Creation Layer
==================================================================

Purpose:
    Converts text chunks into vector embeddings and stores them in a FAISS index.
    Implements session-isolated storage to prevent multi-user data collision.

Author: [Nikhilesh Kumar]
Last Updated: February 2026

Architecture Notes:
    - Uses HuggingFace Inference API (cloud-based, no local GPU needed)
    - Implements batch processing to respect API rate limits
    - Session-specific file storage prevents user data mixing
    - Retries with exponential backoff for transient API failures
"""

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
import os
import time
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure module logger
logger = logging.getLogger(__name__)


def get_vector_store(text_chunks, session_id, batch_size=32, max_retries=3):
    """
    Creates FAISS vector store from text chunks using HuggingFace embeddings API.
    
    Args:
        text_chunks (list[str]): List of text segments to embed
        session_id (str): Unique identifier for this user session (prevents data collision)
        batch_size (int): Number of chunks to process per API call (default: 32)
        max_retries (int): Maximum retry attempts for failed API calls (default: 3)
        
    Returns:
        str: Path to the saved FAISS index directory
        
    Raises:
        ValueError: If API token missing or chunks list is empty
        ConnectionError: If HuggingFace API is unreachable after retries
        RuntimeError: If embedding generation fails
        
    Implementation Strategy:
        1. Validate inputs (token, chunks)
        2. Test API connectivity with probe
        3. Process chunks in batches (avoid timeouts)
        4. Merge batches incrementally (memory efficient)
        5. Save to session-specific directory
        
    Why Batch Processing?
        - HuggingFace free tier has rate limits (~100 req/min)
        - Large documents (1000+ chunks) would timeout
        - Batching with delays ensures reliability
        
    Why Session-Specific Storage?
        - Multiple users on Streamlit Cloud share same server
        - Without isolation: User A's docs overwrite User B's docs
        - Session ID creates unique namespace per user
        
    Security Note:
        - FAISS uses pickle for serialization (potential code execution risk)
        - In production, consider safer alternatives (e.g., JSON + numpy)
        - Currently acceptable for demo/internal tools with trusted users
    """
    
    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================
    if not text_chunks:
        raise ValueError("Cannot create vector store from empty chunks list. "
                        "PDF extraction may have failed.")
    
    logger.info(f"Starting vector store creation for session {session_id[:8]}... "
                f"with {len(text_chunks)} chunks")
    
    # =========================================================================
    # API TOKEN VALIDATION
    # =========================================================================
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not api_token:
        raise ValueError(
            "HuggingFace API token is missing!\n"
            "Please add HUGGINGFACEHUB_API_TOKEN to your .env file.\n"
            "Get your token at: https://huggingface.co/settings/tokens"
        )
    
    # Validate token format (should start with 'hf_')
    if not api_token.startswith('hf_'):
        logger.warning("HuggingFace token doesn't start with 'hf_' - may be invalid")
    
    # =========================================================================
    # INITIALIZE EMBEDDINGS MODEL
    # =========================================================================
    logger.info("Initializing HuggingFace Embeddings API connection...")
    
    try:
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",  # 384-dimensional embeddings
            task="feature-extraction",
            huggingfacehub_api_token=api_token,
        )
    except Exception as e:
        raise ConnectionError(f"Failed to initialize embeddings model: {e}")
    
    # =========================================================================
    # API HEALTH CHECK (Probe)
    # =========================================================================
    # Test with a single word before processing thousands of chunks
    # This fails fast if token is invalid or API is down
    logger.info("Running API health check...")
    
    for attempt in range(max_retries):
        try:
            test_vector = embeddings.embed_query("test")
            
            # Validate response format
            if not test_vector or not isinstance(test_vector, list):
                raise RuntimeError("API returned invalid embedding format")
            
            logger.info(f"✓ API Health Check Passed (vector dimension: {len(test_vector)})")
            break
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"API probe failed (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise ConnectionError(
                    f"HuggingFace API is unreachable after {max_retries} attempts.\n"
                    f"Possible causes:\n"
                    f"  1. Invalid API token (check token has 'Inference' permissions)\n"
                    f"  2. Rate limit exceeded (wait a few minutes)\n"
                    f"  3. Service outage (check status.huggingface.co)\n"
                    f"Error: {e}"
                )
    
    # =========================================================================
    # BATCH PROCESSING WITH INCREMENTAL MERGING
    # =========================================================================
    logger.info(f"Processing {len(text_chunks)} chunks in batches of {batch_size}...")
    
    vector_store = None
    total_batches = (len(text_chunks) + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_num in range(total_batches):
        # Calculate batch slice indices
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(text_chunks))
        batch = text_chunks[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_num + 1}/{total_batches} "
                   f"(chunks {start_idx}-{end_idx})...")
        
        try:
            # Create FAISS index for this batch
            if vector_store is None:
                # First batch: initialize the vector store
                vector_store = FAISS.from_texts(batch, embedding=embeddings)
            else:
                # Subsequent batches: create temporary store and merge
                batch_store = FAISS.from_texts(batch, embedding=embeddings)
                vector_store.merge_from(batch_store)
            
            # Rate limiting: be respectful to free API tier
            # Skip sleep on last batch (no point waiting)
            if batch_num < total_batches - 1:
                time.sleep(0.5)  # 500ms delay between batches
                
        except Exception as e:
            logger.error(f"Failed to process batch {batch_num + 1}: {e}")
            raise RuntimeError(f"Embedding generation failed on batch {batch_num + 1}: {e}")
    
    # =========================================================================
    # SAVE TO SESSION-SPECIFIC DIRECTORY
    # =========================================================================
    # Format: faiss_index_{session_id} (e.g., faiss_index_abc123...)
    # This prevents User A from overwriting User B's index
    index_path = f"faiss_index_{session_id}"
    
    try:
        vector_store.save_local(index_path)
        logger.info(f"✓ Vector store saved successfully to: {index_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save FAISS index: {e}")
    
    return index_path


def load_vector_store(session_id):
    """
    Loads a previously created FAISS vector store for this session.
    
    Args:
        session_id (str): Unique identifier for this user session
        
    Returns:
        FAISS: Loaded vector store ready for similarity search
        
    Raises:
        FileNotFoundError: If no vector store exists for this session
        ValueError: If API token is missing (needed for loading)
        RuntimeError: If vector store is corrupted or incompatible
        
    Usage:
        vector_store = load_vector_store(st.session_state.session_id)
        results = vector_store.similarity_search(query, k=4)
        
    Why Reload Embeddings?
        - FAISS stores vectors, but needs the embedding model for new queries
        - Must use SAME model that created the vectors (dimension mismatch otherwise)
        - This is why we load the exact same model configuration
    """
    
    index_path = f"faiss_index_{session_id}"
    
    # =========================================================================
    # VALIDATE INDEX EXISTS
    # =========================================================================
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No vector store found for session {session_id[:8]}...\n"
            f"Expected path: {index_path}\n"
            f"Please process documents first using the sidebar."
        )
    
    # =========================================================================
    # LOAD API TOKEN (needed for query embeddings)
    # =========================================================================
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not api_token:
        raise ValueError("HuggingFace API token missing - cannot load vector store")
    
    # =========================================================================
    # INITIALIZE SAME EMBEDDINGS MODEL
    # =========================================================================
    # CRITICAL: Must match the model used during creation
    # Different models have different dimensions → incompatible vectors
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",  # Same as creation
        task="feature-extraction",
        huggingfacehub_api_token=api_token,
    )
    
    # =========================================================================
    # LOAD FAISS INDEX
    # =========================================================================
    try:
        # allow_dangerous_deserialization=True required for FAISS (uses pickle)
        # Security Note: Only safe because we control the source of index files
        # In production with untrusted sources, use alternative serialization
        vector_store = FAISS.load_local(
            index_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info(f"✓ Vector store loaded from: {index_path}")
        return vector_store
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load vector store: {e}\n"
            f"The index may be corrupted or created with a different model version."
        )


def cleanup_vector_store(session_id):
    """
    Deletes the vector store for a specific session to free disk space.
    
    Args:
        session_id (str): Session whose vector store should be deleted
        
    Returns:
        bool: True if deleted successfully, False if didn't exist
        
    Usage:
        Called when user uploads new documents or explicitly clears data
        
    Implementation Note:
        FAISS saves as a directory with multiple files (index, metadata)
        We need to remove the entire directory, not just a single file
    """
    import shutil
    
    index_path = f"faiss_index_{session_id}"
    
    try:
        if os.path.exists(index_path):
            shutil.rmtree(index_path)  # Remove directory and all contents
            logger.info(f"✓ Cleaned up vector store: {index_path}")
            return True
        else:
            logger.info(f"No vector store to clean up for session {session_id[:8]}...")
            return False
    except Exception as e:
        logger.error(f"Failed to cleanup vector store {index_path}: {e}")
        return False


# ============================================================================
# USAGE EXAMPLE (for testing independently)
# ============================================================================
if __name__ == "__main__":
    print("Embeddings Module - Ready for import")
    print("\nExample usage:")
    print("  index_path = get_vector_store(chunks, session_id='test123')")
    print("  vector_store = load_vector_store(session_id='test123')")
    print("  results = vector_store.similarity_search('query', k=4)")
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# import time
# from dotenv import load_dotenv

# load_dotenv()

# def get_vector_store(text_chunks):
#     """
#     Uses the modern HuggingFace Endpoint.
#     Includes a debug probe to ensure connection before crashing.
#     """
#     api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#     if not api_token:
#         raise ValueError("HuggingFace Token is missing in .env!")

#     print("Connecting to HuggingFace Cloud API...")
    
#     # NEW CLASS: This is the modern standard suggested by the warning
#     # We use a slight delay to ensure the API is ready
#     embeddings = HuggingFaceEndpointEmbeddings(
#         model="sentence-transformers/all-MiniLM-L6-v2",
#         task="feature-extraction",
#         huggingfacehub_api_token=api_token
#     )
    
#     # --- DEBUG PROBE ---
#     print("Probe: Testing API connection with one word...")
#     try:
#         test_vec = embeddings.embed_query("test")
#         print(f"Probe Success! Received vector of length: {len(test_vec)}")
#     except Exception as e:
#         print(f"Probe Failed: {e}")
#         print("Tip: Check if your HuggingFace Token has 'Inference' permissions.")
#         return # Stop here if API fails
#     # -------------------

#     print(f"Sending {len(text_chunks)} chunks to cloud for embedding...")
    
#     # Process in small batches to avoid timeouts
#     batch_size = 32
#     vector_store = None
    
#     for i in range(0, len(text_chunks), batch_size):
#         batch = text_chunks[i : i + batch_size]
#         print(f"Processing batch {i//batch_size + 1}...")
        
#         if vector_store is None:
#             vector_store = FAISS.from_texts(batch, embedding=embeddings)
#         else:
#             new_vecs = FAISS.from_texts(batch, embedding=embeddings)
#             vector_store.merge_from(new_vecs)
        
#         time.sleep(0.5) # Be polite to the free API
    
#     vector_store.save_local("faiss_index")
#     print("Success! Vector Store Saved.")
