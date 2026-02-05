"""
DocuMind: AI-Powered Research Assistant
========================================

A production-grade RAG (Retrieval-Augmented Generation) application that enables
users to have intelligent conversations with their PDF documents.

Author: [Nikhilesh Kumar]
Last Updated: February 2026
Tech Stack: Streamlit, LangChain, FAISS, HuggingFace, Google Gemini

Architecture:
    1. Document Ingestion (pdf_handler.py)
    2. Vector Embedding (embeddings.py)
    3. LLM Question Answering (llm_chain.py)
    4. UI Orchestration (this file)

Production Features:
    - Session-isolated storage (multi-user safe)
    - Comprehensive error handling
    - Input validation at every layer
    - Performance optimization (caching)
    - User-friendly error messages
    - Chat history management
"""
#--
import sys
import subprocess

# INTERNAL DIAGNOSTIC: Force install langchain if missing
try:
    import langchain.chains
except ImportError:
    print("⚠️ LangChain main package missing. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain"])
    #--
import streamlit as st
import os
import logging
import uuid
from datetime import datetime

# Import our production-grade modules
from src.pdf_handler import get_pdf_text, get_text_chunks
from src.embeddings import get_vector_store, load_vector_store, cleanup_vector_store
from src.llm_chain import get_conversational_chain, validate_chain_inputs, format_sources

from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables
load_dotenv()

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application constants
MAX_CHAT_HISTORY = 50  # Prevent memory bloat
MAX_FILE_SIZE_MB = 10  # Per PDF file limit
SIMILARITY_SEARCH_K = 4  # Number of chunks to retrieve


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """
    Initializes all session state variables on first load.
    
    Session State Variables:
        - session_id: Unique identifier for this user session
        - messages: Chat history (list of {role, content} dicts)
        - docs_processed: Boolean flag indicating if documents are ready
        - vector_store: Cached FAISS index (avoids reloading from disk)
        - chain: Cached LLM chain (avoids recreation)
        - embeddings: Cached embeddings model
        - index_path: Path to this session's FAISS index
        
    Why Session State?
        Streamlit reruns the entire script on every interaction.
        Session state persists data across reruns within a user session.
        Without it, we'd lose chat history and recreate expensive objects.
        
    Why Cache LLM Objects?
        Creating embeddings/chains involves API connections and initialization.
        Doing this every message would be slow and wasteful.
        Cache once, reuse many times.
    """
    
    # Generate unique session ID (only once per session)
    if "session_id" not in st.session_state:
        # UUID ensures uniqueness even with concurrent users
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"New session started: {st.session_state.session_id[:8]}...")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize document processing flag
    if "docs_processed" not in st.session_state:
        st.session_state.docs_processed = False
    
    # Initialize cached objects (None until needed)
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "chain" not in st.session_state:
        st.session_state.chain = None
    
    if "index_path" not in st.session_state:
        st.session_state.index_path = None


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_documents(pdf_docs):
    """
    Processes uploaded PDFs through the complete RAG pipeline.
    
    Pipeline:
        1. Extract text from PDFs (with error handling)
        2. Split into semantic chunks
        3. Generate embeddings via HuggingFace API
        4. Store in session-specific FAISS index
        5. Cache vector store in session state
        
    Args:
        pdf_docs (list): Uploaded PDF files from Streamlit
        
    Returns:
        tuple: (success: bool, message: str)
        
    Error Handling:
        - Encrypted PDFs → Skip with warning
        - Scanned PDFs → Detect and notify user
        - Empty PDFs → Validation catches
        - API failures → Retry logic in embeddings.py
        - Network issues → Clear error messages
        
    Performance Optimization:
        - Cleans up old index before creating new one
        - Caches vector store to avoid disk I/O on every query
        - Batch processing in embeddings.py prevents timeouts
    """
    
    try:
        # ====================================================================
        # STEP 1: EXTRACT TEXT FROM PDFs
        # ====================================================================
        with st.spinner("📄 Extracting text from PDFs..."):
            logger.info(f"Processing {len(pdf_docs)} PDF files...")
            
            try:
                raw_text = get_pdf_text(pdf_docs)
            except ValueError as e:
                # Specific handling for extraction failures
                logger.error(f"PDF extraction failed: {e}")
                return False, f"❌ PDF Processing Failed\n\n{str(e)}"
            except Exception as e:
                logger.error(f"Unexpected error during extraction: {e}")
                return False, f"❌ Unexpected Error\n\nCouldn't extract text: {str(e)}"
        
        # ====================================================================
        # STEP 2: SPLIT TEXT INTO CHUNKS
        # ====================================================================
        with st.spinner("✂️ Splitting text into semantic chunks..."):
            try:
                text_chunks = get_text_chunks(raw_text)
                logger.info(f"Created {len(text_chunks)} text chunks")
            except ValueError as e:
                logger.error(f"Chunking failed: {e}")
                return False, f"❌ Text Splitting Failed\n\n{str(e)}"
        
        # ====================================================================
        # STEP 3: CLEANUP OLD INDEX (if exists)
        # ====================================================================
        # If user is re-uploading docs, delete the old index
        if st.session_state.index_path:
            logger.info(f"Cleaning up old index: {st.session_state.index_path}")
            cleanup_vector_store(st.session_state.session_id)
            st.session_state.vector_store = None  # Invalidate cache
        
        # ====================================================================
        # STEP 4: CREATE VECTOR EMBEDDINGS
        # ====================================================================
        with st.spinner(f"🧠 Generating embeddings for {len(text_chunks)} chunks... (this may take a minute)"):
            try:
                index_path = get_vector_store(
                    text_chunks=text_chunks,
                    session_id=st.session_state.session_id
                )
                st.session_state.index_path = index_path
                logger.info(f"Vector store created: {index_path}")
                
            except ValueError as e:
                # API token issues
                logger.error(f"Embeddings creation failed: {e}")
                return False, f"❌ Configuration Error\n\n{str(e)}"
            except ConnectionError as e:
                # API connectivity issues
                logger.error(f"API connection failed: {e}")
                return False, f"❌ API Connection Failed\n\n{str(e)}"
            except Exception as e:
                logger.error(f"Unexpected error during embedding: {e}")
                return False, f"❌ Embedding Failed\n\n{str(e)}"
        
        # ====================================================================
        # STEP 5: LOAD AND CACHE VECTOR STORE
        # ====================================================================
        try:
            st.session_state.vector_store = load_vector_store(st.session_state.session_id)
            st.session_state.docs_processed = True
            logger.info("Vector store loaded and cached successfully")
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False, f"❌ Failed to load vector database\n\n{str(e)}"
        
        # ====================================================================
        # SUCCESS
        # ====================================================================
        success_msg = (
            f"✅ **Processing Complete!**\n\n"
            f"📊 Statistics:\n"
            f"- Files processed: {len(pdf_docs)}\n"
            f"- Text chunks: {len(text_chunks)}\n"
            f"- Ready for questions!"
        )
        
        return True, success_msg
        
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.exception("Unexpected error in process_documents")
        return False, f"❌ Unexpected Error\n\n{str(e)}\n\nPlease try again or contact support."


# ============================================================================
# QUERY PROCESSING
# ============================================================================

def process_query(question):
    """
    Processes user question and generates answer from document context.
    
    Pipeline:
        1. Validate vector store exists
        2. Search for relevant document chunks
        3. Validate search results
        4. Initialize LLM chain (cached)
        5. Validate inputs before LLM call
        6. Generate answer
        
    Args:
        question (str): User's question
        
    Returns:
        str: Answer from the LLM or error message
        
    Performance Optimization:
        - Vector store cached in session state (no disk I/O)
        - LLM chain cached (no re-initialization)
        - Only generates embeddings for the query (not documents)
        
    Error Recovery:
        - No results from search → Helpful message
        - API failures → Clear error with suggestions
        - Invalid inputs → Validation catches early
    """
    
    try:
        # ====================================================================
        # VALIDATION: Ensure documents are processed
        # ====================================================================
        if not st.session_state.docs_processed or not st.session_state.vector_store:
            logger.warning("Query attempted without processed documents")
            return "⚠️ **No Documents Loaded**\n\nPlease upload and process PDFs in the sidebar first."
        
        # ====================================================================
        # STEP 1: SIMILARITY SEARCH
        # ====================================================================
        logger.info(f"Searching for relevant chunks for query: {question[:50]}...")
        
        try:
            # Retrieve top-k most relevant chunks
            # k=4 balances between having enough context and not overwhelming the LLM
            context_docs = st.session_state.vector_store.similarity_search(
                question, 
                k=SIMILARITY_SEARCH_K
            )
            
            logger.info(f"Retrieved {len(context_docs)} relevant chunks")
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return f"❌ **Search Failed**\n\nCouldn't search document database: {str(e)}"
        
        # ====================================================================
        # VALIDATION: Check if relevant docs were found
        # ====================================================================
        if not context_docs:
            logger.warning("No relevant documents found for query")
            return (
                "🤔 **No Relevant Information Found**\n\n"
                "I couldn't find any relevant sections in your documents for this question. "
                "Try rephrasing or asking about topics that are actually in the uploaded PDFs."
            )
        
        # ====================================================================
        # STEP 2: INITIALIZE LLM CHAIN (cached)
        # ====================================================================
        # Only create chain once per session, then reuse
        if st.session_state.chain is None:
            logger.info("Initializing LLM chain (first query)")
            try:
                st.session_state.chain = get_conversational_chain()
            except ValueError as e:
                logger.error(f"Chain initialization failed: {e}")
                return f"❌ **Configuration Error**\n\n{str(e)}"
            except ConnectionError as e:
                logger.error(f"LLM API connection failed: {e}")
                return f"❌ **API Connection Failed**\n\n{str(e)}"
        
        # ====================================================================
        # STEP 3: VALIDATE INPUTS
        # ====================================================================
        is_valid, error_msg = validate_chain_inputs(context_docs, question)
        
        if not is_valid:
            logger.warning(f"Input validation failed: {error_msg}")
            return f"⚠️ **Invalid Input**\n\n{error_msg}"
        
        # ====================================================================
        # STEP 4: GENERATE ANSWER
        # ====================================================================
        logger.info("Generating answer from LLM...")
        
        try:
            # Invoke the chain with context and question
            response = st.session_state.chain.invoke({
                "context": context_docs,
                "question": question
            })
            
            logger.info(f"Answer generated successfully ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            
            # Provide helpful error messages based on error type
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                return (
                    "❌ **API Quota Exceeded**\n\n"
                    "The API rate limit has been reached. "
                    "Please wait a few minutes and try again."
                )
            elif "authentication" in str(e).lower() or "api key" in str(e).lower():
                return (
                    "❌ **Authentication Error**\n\n"
                    "There's an issue with the API key. "
                    "Please check your configuration."
                )
            else:
                return f"❌ **Answer Generation Failed**\n\n{str(e)}"
        
    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception("Unexpected error in process_query")
        return f"❌ **Unexpected Error**\n\n{str(e)}"


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application entry point - orchestrates the entire UI and workflow.
    
    UI Structure:
        1. Page configuration and header
        2. Session state initialization
        3. Sidebar for document upload
        4. Main chat interface
        5. Query processing and response
        
    Session Management:
        - Unique session ID per user
        - Isolated file storage
        - Cached objects for performance
        - Bounded chat history
    """
    
    # =========================================================================
    # PAGE CONFIGURATION
    # =========================================================================
    st.set_page_config(
        page_title="DocuMind - Research Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # =========================================================================
    # CUSTOM CSS (Optional polish)
    # =========================================================================
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4;
        }
        .caption {
            font-size: 0.9rem;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # HEADER
    # =========================================================================
    st.markdown('<h1 class="main-header">🧠 DocuMind: Research Assistant</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="caption">🚀 Powered by Gemini 1.5 Flash & HuggingFace | Hallucination-Free RAG</p>', 
                unsafe_allow_html=True)
    
    # =========================================================================
    # INITIALIZE SESSION STATE
    # =========================================================================
    initialize_session_state()
    
    # =========================================================================
    # SIDEBAR - DOCUMENT UPLOAD
    # =========================================================================
    with st.sidebar:
        st.header("📂 Document Upload Center")
        
        # Session info (helpful for debugging)
        with st.expander("ℹ️ Session Info", expanded=False):
            st.code(f"Session ID: {st.session_state.session_id[:8]}...")
            st.caption(f"Documents Processed: {st.session_state.docs_processed}")
            st.caption(f"Messages: {len(st.session_state.messages)}")
        
        # File uploader
        pdf_docs = st.file_uploader(
            "Upload Research PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Upload one or more PDF files (max {MAX_FILE_SIZE_MB}MB each)"
        )
        
        # Process button
        if st.button("🚀 Submit & Process", type="primary", use_container_width=True):
            # Validation: Check if files were uploaded
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF file first.")
            else:
                # File size validation
                oversized_files = [
                    f.name for f in pdf_docs 
                    if f.size > MAX_FILE_SIZE_MB * 1024 * 1024
                ]
                
                if oversized_files:
                    st.error(
                        f"❌ The following files exceed {MAX_FILE_SIZE_MB}MB limit:\n" + 
                        "\n".join(f"- {name}" for name in oversized_files)
                    )
                else:
                    # Process documents
                    success, message = process_documents(pdf_docs)
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        # Clear chat button
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Instructions
        st.divider()
        st.markdown("""
        ### 📖 How to Use
        1. **Upload** your PDF documents
        2. Click **Submit & Process**
        3. **Ask questions** in the chat
        4. Get **grounded answers** from your docs
        
        ### 💡 Tips
        - Ask specific questions for best results
        - Reference topics from your documents
        - Use follow-up questions to dig deeper
        """)
    
    # =========================================================================
    # MAIN AREA - CHAT INTERFACE
    # =========================================================================
    
    # Display chat history
    # Limit to last MAX_CHAT_HISTORY messages to prevent memory bloat
    display_messages = st.session_state.messages[-MAX_CHAT_HISTORY:]
    
    for message in display_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = process_query(prompt)
            
            st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Enforce history limit
        if len(st.session_state.messages) > MAX_CHAT_HISTORY:
            st.session_state.messages = st.session_state.messages[-MAX_CHAT_HISTORY:]
            logger.info(f"Chat history trimmed to {MAX_CHAT_HISTORY} messages")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
# import streamlit as st
# from src.pdf_handler import get_pdf_text, get_text_chunks
# from src.embeddings import get_vector_store
# from src.llm_chain import get_conversational_chain
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# def main():
#     # --- PAGE CONFIGURATION ---
#     st.set_page_config("DocuMind", page_icon=":books:", layout="wide")
    
#     # --- HEADER ---
#     st.title("🧠 DocuMind: Research Assistant")
#     st.caption("🚀 Powered by Gemini 1.5 & HuggingFace | Hallucination-Free RAG")

#     # --- SESSION STATE INITIALIZATION ---
#     # 1. Chat History
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
    
#     # 2. The Gatekeeper (Prevents Zombie Data)
#     if "docs_processed" not in st.session_state:
#         st.session_state.docs_processed = False

#     # --- SIDEBAR (Setup) ---
#     with st.sidebar:
#         st.header("📂 Document Center")
#         pdf_docs = st.file_uploader("Upload Research PDFs", accept_multiple_files=True)
        
#         if st.button("Submit & Process"):
#             if not pdf_docs:
#                 st.warning("⚠️ Please upload a PDF first.")
#             else:
#                 with st.spinner("Processing Knowledge Base..."):
#                     try:
#                         # 1. Extract
#                         raw_text = get_pdf_text(pdf_docs)
#                         # 2. Chunk
#                         text_chunks = get_text_chunks(raw_text)
#                         # 3. Embed
#                         get_vector_store(text_chunks)
                        
#                         # Set the Gatekeeper to TRUE
#                         st.session_state.docs_processed = True
                        
#                         st.success("✅ Documentation Processed! You can now chat.")
#                     except Exception as e:
#                         st.error(f"Error processing document: {e}")

#     # --- DISPLAY CHAT HISTORY ---
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # --- CHAT INPUT ---
#     if prompt := st.chat_input("Ask a question about your documents..."):
        
#         # Display User Message
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         # --- THE GATEKEEPER CHECK ---
#         # If the user hasn't processed docs in THIS session, stop them.
#         if not st.session_state.docs_processed:
#             with st.chat_message("assistant"):
#                 st.warning("⚠️ I need you to upload and process a PDF in the sidebar first!")
#             st.session_state.messages.append({"role": "assistant", "content": "⚠️ Please upload a PDF first."})
#             return  # STOP HERE

#         # --- GENERATE ANSWER ---
#         api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
#         embeddings = HuggingFaceEndpointEmbeddings(
#             model="sentence-transformers/all-MiniLM-L6-v2",
#             task="feature-extraction",
#             huggingfacehub_api_token=api_token
#         )
        
#         try:
#             # Load DB
#             new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#             docs = new_db.similarity_search(prompt)
            
#             # Get Answer
#             chain = get_conversational_chain()
#             response = chain.invoke({"context": docs, "question": prompt})
            
#             # Display Assistant Message
#             with st.chat_message("assistant"):
#                 st.markdown(response)
            
#             st.session_state.messages.append({"role": "assistant", "content": response})
            
#         except Exception as e:
#             with st.chat_message("assistant"):
#                 st.error("Technical Error: API Limit Reached or Key Issue.")
#                 st.info(f"Details: {e}")

# if __name__ == "__main__":
#     main()
# import streamlit as st
# from src.pdf_handler import get_pdf_text, get_text_chunks
# from src.embeddings import get_vector_store
# from src.llm_chain import get_conversational_chain
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# def user_input(user_question):
#     """
#     Handles the user's question using the HuggingFace Cloud Model
#     """
#     api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
#     # 1. Setup the Embedding Model (Must match what we used to create the DB)
#     embeddings = HuggingFaceEndpointEmbeddings(
#         model="sentence-transformers/all-MiniLM-L6-v2",
#         task="feature-extraction",
#         huggingfacehub_api_token=api_token
#     )
    
#     # 2. Load the Vector Database
#     # allow_dangerous_deserialization is needed for local files
#     try:
#         new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
#         # 3. Search for the answer
#         docs = new_db.similarity_search(user_question)
        
#         # 4. Get the chain and run it
#         chain = get_conversational_chain()
        
#         response = chain.invoke(
#             {"context": docs, "question": user_question}
#         )
        
#         # The new chain returns the answer string directly!
#         st.write("Reply: ", response)
        
#     except Exception as e:
#         st.error(f"Error finding answer: {e}")
#         st.info("Did you upload and process a PDF first?")

# def main():
#     # --- PAGE CONFIGURATION ---
#     st.set_page_config("DocuMind", page_icon=":books:")
#     st.header("DocuMind: Hallucination-Free Research Assistant")

#     # --- MAIN INPUT ---
#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     # --- SIDEBAR ---
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
#         if st.button("Submit & Process"):
#             if not pdf_docs:
#                 st.warning("Please upload a PDF first.")
#             else:
#                 with st.spinner("Processing..."):
#                     # 1. Extract
#                     raw_text = get_pdf_text(pdf_docs)
                    
#                     # 2. Chunk
#                     text_chunks = get_text_chunks(raw_text)
                    
#                     # 3. Embed (This triggers the HuggingFace Cloud call)
#                     get_vector_store(text_chunks)
                    
#                     st.success("Done! You can now ask questions.")

# # --- ENTRY POINT (Crucial for the app to run) ---
# if __name__ == "__main__":
#     main()
# import streamlit as st
# from src.pdf_handler import get_pdf_text, get_text_chunks
# from src.embeddings import get_vector_store
# from src.llm_chain import get_conversational_chain
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# def main():
#     # --- PAGE CONFIGURATION ---
#     st.set_page_config("DocuMind", page_icon=":books:", layout="wide")
    
#     # --- HEADER ---
#     st.title("🧠 DocuMind: Research Assistant")
#     st.caption("🚀 Powered by Gemini 1.5 & HuggingFace | Hallucination-Free RAG")

#     # --- SIDEBAR (Setup) ---
#     with st.sidebar:
#         st.header("📂 Document Center")
#         pdf_docs = st.file_uploader("Upload Research PDFs", accept_multiple_files=True)
        
#         if st.button("Submit & Process"):
#             if not pdf_docs:
#                 st.warning("⚠️ Please upload a PDF first.")
#             else:
#                 with st.spinner("Processing Knowledge Base..."):
#                     # 1. Extract
#                     raw_text = get_pdf_text(pdf_docs)
#                     # 2. Chunk
#                     text_chunks = get_text_chunks(raw_text)
#                     # 3. Embed
#                     get_vector_store(text_chunks)
#                     st.success("✅ Documentation Processed! You can now chat.")

#     # --- CHAT HISTORY STATE MANAGEMENT ---
#     # This keeps the chat on screen even when Streamlit reruns
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # --- DISPLAY CHAT HISTORY ---
#     # Loop through the history and show every message
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # --- CHAT INPUT (The "ChatGPT" Box) ---
#     if prompt := st.chat_input("Ask a question about your documents..."):
        
#         # 1. Display User Message immediately
#         with st.chat_message("user"):
#             st.markdown(prompt)
#         # 2. Save to history
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         # 3. Generate Answer
#         api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
#         # Setup Embeddings (Must match what we used to create the DB)
#         embeddings = HuggingFaceEndpointEmbeddings(
#             model="sentence-transformers/all-MiniLM-L6-v2",
#             task="feature-extraction",
#             huggingfacehub_api_token=api_token
#         )
        
#         try:
#             # Load DB
#             new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#             docs = new_db.similarity_search(prompt)
            
#             # Get Answer
#             chain = get_conversational_chain()
#             response = chain.invoke({"context": docs, "question": prompt})
            
#             # 4. Display Assistant Message
#             with st.chat_message("assistant"):
#                 st.markdown(response)
            
#             # 5. Save to history
#             st.session_state.messages.append({"role": "assistant", "content": response})
            
#         except Exception as e:
#             # Friendly error handling
#             with st.chat_message("assistant"):
#                 st.error("I need you to upload a PDF first so I can learn from it!")
#                 st.info(f"Technical Error: {e}")

# if __name__ == "__main__":
#     main()

