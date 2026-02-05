"""
LLM Chain Module - Question Answering Layer
============================================

Purpose:
    Creates and manages the conversational chain that answers questions
    based on retrieved document context using Google's Gemini model.

Author: [Nikhilesh Kumar]
Last Updated: February 2026

Architecture Notes:
    - Uses Gemini 1.5 Flash for fast, cost-effective inference
    - Implements "stuff" strategy: concatenates all context into single prompt
    - Enforces grounded responses (no hallucinations outside context)
    - Caches chain for reuse across multiple queries
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure module logger
logger = logging.getLogger(__name__)


def get_conversational_chain(temperature=0.3, max_output_tokens=2048):
    """
    Creates a document-grounded question-answering chain using Gemini 1.5 Flash.
    
    Args:
        temperature (float): Controls randomness (0.0 = deterministic, 1.0 = creative)
                           Default 0.3 for factual Q&A with slight flexibility
        max_output_tokens (int): Maximum length of generated response
                                Default 2048 (roughly 1500 words)
    
    Returns:
        RunnableSequence: LangChain chain that takes {context, question} and returns answer
        
    Raises:
        ValueError: If GOOGLE_API_KEY is missing from environment
        ConnectionError: If unable to connect to Gemini API
        
    Chain Architecture:
        Input: {"context": [Document, ...], "question": str}
          ↓
        1. Stuff Strategy: Concatenates all document chunks
          ↓
        2. Prompt Template: Injects context + question
          ↓
        3. Gemini Model: Generates grounded answer
          ↓
        Output: str (the answer)
        
    Why Gemini 1.5 Flash?
        - Fast inference (~2-3 seconds)
        - Large context window (1M tokens - handles many chunks)
        - Cost-effective for high-volume applications
        - Good balance of quality vs. speed
        
    Why "Stuff" Strategy?
        - Simple: Just concatenates all retrieved chunks
        - Works well when chunks fit in context window
        - Alternative strategies (map-reduce, refine) are slower
        - Gemini's 1M token window handles most use cases
        
    Temperature Explained:
        - 0.0: Completely deterministic (same answer every time)
        - 0.3: Mostly consistent with slight variation (RECOMMENDED for Q&A)
        - 0.7: More creative responses (good for writing tasks)
        - 1.0: Maximum creativity (use for brainstorming)
        
    Security Note:
        - API key loaded from environment (never hardcode!)
        - In production: use Google Cloud Secret Manager
        - Rotate keys regularly
    """
    
    # =========================================================================
    # API KEY VALIDATION
    # =========================================================================
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Google API key is missing!\n"
            "Please add GOOGLE_API_KEY to your .env file.\n"
            "Get your key at: https://makersuite.google.com/app/apikey"
        )
    
    # Basic format validation (Google API keys start with 'AIza')
    if not api_key.startswith('AIza'):
        logger.warning("Google API key doesn't start with 'AIza' - may be invalid")
    
    logger.info("Initializing Gemini 1.5 Flash model...")
    
    # =========================================================================
    # PROMPT ENGINEERING
    # =========================================================================
    # This is the "system prompt" that guides the model's behavior
    # Key principles:
    #   1. Be detailed (use provided context)
    #   2. Be grounded (don't hallucinate)
    #   3. Be honest (admit when answer isn't in context)
    
    prompt_template = """You are a helpful research assistant analyzing documents to answer questions.

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context below
- Provide detailed, comprehensive answers with specific details from the context
- If the exact answer is not in the context, say "The answer is not available in the provided context"
- Never make up or infer information that isn't explicitly stated in the context
- If the context contains partial information, state what you know and what's missing
- Use direct quotes from the context when appropriate

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    
    # =========================================================================
    # INITIALIZE GEMINI MODEL
    # =========================================================================
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Pinned version (not "latest")
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            google_api_key=api_key,
            
            # Additional safety configurations
            convert_system_message_to_human=True,  # Gemini compatibility
        )
        
        logger.info(f"✓ Model initialized: gemini-1.5-flash "
                   f"(temp={temperature}, max_tokens={max_output_tokens})")
        
    except Exception as e:
        raise ConnectionError(f"Failed to initialize Gemini model: {e}")
    
    # =========================================================================
    # CREATE PROMPT TEMPLATE
    # =========================================================================
    # LangChain automatically detects input variables from {bracketed} names
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # =========================================================================
    # CREATE DOCUMENT CHAIN
    # =========================================================================
    # "Stuff" chain: Takes documents + question → Stuffs docs into prompt → LLM response
    try:
        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt,
            document_variable_name="context"  # Maps to {context} in prompt
        )
        
        logger.info("✓ Conversational chain created successfully")
        return chain
        
    except Exception as e:
        raise RuntimeError(f"Failed to create conversational chain: {e}")


def validate_chain_inputs(context_docs, question, max_context_length=50000):
    """
    Validates inputs before sending to the LLM chain.
    
    Args:
        context_docs (list): List of Document objects from vector search
        question (str): User's question
        max_context_length (int): Maximum characters in combined context
                                 Default 50k chars ≈ 12k tokens (safe for Gemini)
    
    Returns:
        tuple: (is_valid, error_message)
        
    Validation Checks:
        1. Context is not empty (found relevant documents)
        2. Question is not empty
        3. Combined context doesn't exceed token limits
        4. Question is not excessively long
        
    Why Validate Context Length?
        - Even though Gemini supports 1M tokens, stuffing too much hurts quality
        - More context = more noise = worse answers
        - Better to use top-k most relevant chunks than everything
        
    Usage:
        is_valid, error = validate_chain_inputs(docs, question)
        if not is_valid:
            return error
        response = chain.invoke({"context": docs, "question": question})
    """
    
    # =========================================================================
    # CHECK CONTEXT EXISTS
    # =========================================================================
    if not context_docs:
        return False, "No relevant documents found for your question."
    
    # =========================================================================
    # CHECK QUESTION EXISTS
    # =========================================================================
    if not question or not question.strip():
        return False, "Question cannot be empty."
    
    # =========================================================================
    # CHECK QUESTION LENGTH
    # =========================================================================
    if len(question) > 2000:  # Roughly 500 tokens
        return False, "Question is too long. Please ask a more concise question."
    
    # =========================================================================
    # CHECK COMBINED CONTEXT LENGTH
    # =========================================================================
    # Calculate total characters in all document chunks
    total_context_length = sum(len(doc.page_content) for doc in context_docs)
    
    if total_context_length > max_context_length:
        logger.warning(f"Context length ({total_context_length} chars) exceeds "
                      f"recommended maximum ({max_context_length} chars)")
        return False, (
            f"The retrieved context is too large ({total_context_length} characters). "
            f"This usually means your question is too broad. "
            f"Try asking a more specific question."
        )
    
    # =========================================================================
    # ALL CHECKS PASSED
    # =========================================================================
    logger.info(f"✓ Input validation passed: {len(context_docs)} docs, "
               f"{total_context_length} chars context, "
               f"{len(question)} chars question")
    
    return True, None


def format_sources(context_docs):
    """
    Extracts source information from retrieved documents for citation.
    
    Args:
        context_docs (list): List of Document objects from vector search
        
    Returns:
        str: Formatted string listing source documents
        
    Usage:
        Used to show users which parts of their PDFs were used to answer
        Helps with transparency and verification
        
    Note:
        Requires that documents have 'source' metadata (added during chunking)
        If metadata not present, returns generic message
    """
    
    if not context_docs:
        return "No sources available."
    
    # Extract unique sources from document metadata
    sources = set()
    for doc in context_docs:
        # Metadata added during PDF processing would include page numbers, file names, etc.
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            sources.add(doc.metadata['source'])
    
    if sources:
        source_list = "\n".join(f"- {source}" for source in sorted(sources))
        return f"**Sources:**\n{source_list}"
    else:
        return f"Based on {len(context_docs)} document chunks from your uploaded PDFs."


# ============================================================================
# USAGE EXAMPLE (for testing independently)
# ============================================================================
if __name__ == "__main__":
    print("LLM Chain Module - Ready for import")
    print("\nExample usage:")
    print("  chain = get_conversational_chain()")
    print("  response = chain.invoke({'context': docs, 'question': 'What is...'})")
    print("\nWith validation:")
    print("  is_valid, error = validate_chain_inputs(docs, question)")
    print("  if is_valid:")
    print("      response = chain.invoke({'context': docs, 'question': question})")
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# def get_conversational_chain():
#     """
#     Creates a chain that takes a list of documents and a question, 
#     and returns an answer.
#     """
    
#     # 1. Define the Prompt (No changes here)
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details.
#     If the answer is not in provided context just say, "The answer is not available in the context", don't provide the wrong answer.
    
#     Context:
#     {context}
    
#     Question: 
#     {question}
    
#     Answer:
#     """
    
#     # 2. Initialize Model
#     model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    
#     # 3. Create Prompt
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
#     # 4. Create the Chain (The Modern Way)
#     # This automatically handles "stuffing" the documents into the {context} variable
#     chain = create_stuff_documents_chain(model, prompt)
    
#     return chain
