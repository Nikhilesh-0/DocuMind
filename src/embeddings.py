from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
import os
import time
from dotenv import load_dotenv

load_dotenv()

def get_vector_store(text_chunks):
    """
    Uses the modern HuggingFace Endpoint.
    Includes a debug probe to ensure connection before crashing.
    """
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError("HuggingFace Token is missing in .env!")

    print("Connecting to HuggingFace Cloud API...")
    
    # NEW CLASS: This is the modern standard suggested by the warning
    # We use a slight delay to ensure the API is ready
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=api_token
    )
    
    # --- DEBUG PROBE ---
    print("Probe: Testing API connection with one word...")
    try:
        test_vec = embeddings.embed_query("test")
        print(f"Probe Success! Received vector of length: {len(test_vec)}")
    except Exception as e:
        print(f"Probe Failed: {e}")
        print("Tip: Check if your HuggingFace Token has 'Inference' permissions.")
        return # Stop here if API fails
    # -------------------

    print(f"Sending {len(text_chunks)} chunks to cloud for embedding...")
    
    # Process in small batches to avoid timeouts
    batch_size = 32
    vector_store = None
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        if vector_store is None:
            vector_store = FAISS.from_texts(batch, embedding=embeddings)
        else:
            new_vecs = FAISS.from_texts(batch, embedding=embeddings)
            vector_store.merge_from(new_vecs)
        
        time.sleep(0.5) # Be polite to the free API
    
    vector_store.save_local("faiss_index")
    print("Success! Vector Store Saved.")