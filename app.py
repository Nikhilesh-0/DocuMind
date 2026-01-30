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
import streamlit as st
from src.pdf_handler import get_pdf_text, get_text_chunks
from src.embeddings import get_vector_store
from src.llm_chain import get_conversational_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def main():
    # --- PAGE CONFIGURATION ---
    st.set_page_config("DocuMind", page_icon=":books:", layout="wide")
    
    # --- HEADER ---
    st.title("🧠 DocuMind: Research Assistant")
    st.caption("🚀 Powered by Gemini 1.5 & HuggingFace | Hallucination-Free RAG")

    # --- SIDEBAR (Setup) ---
    with st.sidebar:
        st.header("📂 Document Center")
        pdf_docs = st.file_uploader("Upload Research PDFs", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                with st.spinner("Processing Knowledge Base..."):
                    # 1. Extract
                    raw_text = get_pdf_text(pdf_docs)
                    # 2. Chunk
                    text_chunks = get_text_chunks(raw_text)
                    # 3. Embed
                    get_vector_store(text_chunks)
                    st.success("✅ Documentation Processed! You can now chat.")

    # --- CHAT HISTORY STATE MANAGEMENT ---
    # This keeps the chat on screen even when Streamlit reruns
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- DISPLAY CHAT HISTORY ---
    # Loop through the history and show every message
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- CHAT INPUT (The "ChatGPT" Box) ---
    if prompt := st.chat_input("Ask a question about your documents..."):
        
        # 1. Display User Message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        # 2. Save to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 3. Generate Answer
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        # Setup Embeddings (Must match what we used to create the DB)
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=api_token
        )
        
        try:
            # Load DB
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(prompt)
            
            # Get Answer
            chain = get_conversational_chain()
            response = chain.invoke({"context": docs, "question": prompt})
            
            # 4. Display Assistant Message
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # 5. Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            # Friendly error handling
            with st.chat_message("assistant"):
                st.error("I need you to upload a PDF first so I can learn from it!")
                st.info(f"Technical Error: {e}")

if __name__ == "__main__":
    main()