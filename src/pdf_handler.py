"""
PDF Handler Module - Document Ingestion Layer
==============================================

Purpose:
    Extracts and preprocesses text from uploaded PDF files for RAG pipeline.
    Implements robust error handling for production scenarios.

Author: [Nikhilesh Kumar]
Last Updated: February 2026
"""

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

# Configure module-level logger for debugging
logger = logging.getLogger(__name__)


def get_pdf_text(pdf_docs):
    """
    Extracts raw text from uploaded PDF documents with comprehensive error handling.
    
    Args:
        pdf_docs (list): List of uploaded PDF file objects from Streamlit file_uploader
        
    Returns:
        str: Concatenated text from all PDFs
        
    Raises:
        ValueError: If no valid text could be extracted from any PDF
        PdfReadError: If PDFs are corrupted or encrypted
        
    Implementation Notes:
        - Handles encrypted PDFs gracefully
        - Detects and reports scanned/image-only PDFs
        - Validates that actual text content was extracted
        - Preserves page structure with newlines between pages
        
    Edge Cases Handled:
        1. Encrypted/password-protected PDFs → Clear error message
        2. Scanned PDFs (image-only) → Warning + skip
        3. Corrupted PDF files → Specific error per file
        4. Empty PDFs → Skip with warning
        5. Mixed valid/invalid PDFs → Process valid ones, report failures
    """
    
    extracted_text = ""
    failed_files = []
    scanned_files = []
    successful_files = []
    
    for pdf_file in pdf_docs:
        try:
            # Attempt to read the PDF
            pdf_reader = PdfReader(pdf_file)
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                logger.warning(f"Skipping encrypted PDF: {pdf_file.name}")
                failed_files.append(f"{pdf_file.name} (encrypted)")
                continue
            
            # Extract text from all pages
            file_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    
                    # pypdf can return None or empty string for image-only pages
                    if page_text:
                        file_text += page_text + "\n"  # Preserve page breaks
                        
                except Exception as page_error:
                    logger.warning(f"Error on page {page_num + 1} of {pdf_file.name}: {page_error}")
                    continue
            
            # Validate that we actually extracted meaningful text
            if not file_text.strip():
                logger.warning(f"No text extracted from {pdf_file.name} - likely a scanned/image PDF")
                scanned_files.append(pdf_file.name)
                continue
            
            # Success - add to corpus
            extracted_text += file_text
            successful_files.append(pdf_file.name)
            logger.info(f"Successfully extracted {len(file_text)} characters from {pdf_file.name}")
            
        except PdfReadError as e:
            logger.error(f"Corrupted PDF file {pdf_file.name}: {e}")
            failed_files.append(f"{pdf_file.name} (corrupted)")
            
        except Exception as e:
            logger.error(f"Unexpected error processing {pdf_file.name}: {e}")
            failed_files.append(f"{pdf_file.name} (unknown error)")
    
    # ============================================================================
    # VALIDATION: Ensure we extracted something useful
    # ============================================================================
    if not extracted_text.strip():
        error_msg = "No text could be extracted from any PDF. "
        
        if scanned_files:
            error_msg += f"\nScanned/image PDFs detected: {', '.join(scanned_files)}. "
            error_msg += "These require OCR processing (not currently supported)."
        
        if failed_files:
            error_msg += f"\nFailed files: {', '.join(failed_files)}"
        
        raise ValueError(error_msg)
    
    # Log summary for debugging
    logger.info(f"Extraction complete: {len(successful_files)} successful, "
                f"{len(failed_files)} failed, {len(scanned_files)} scanned")
    
    return extracted_text


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits extracted text into semantically meaningful chunks for vector embedding.
    
    Args:
        text (str): Raw concatenated text from PDFs
        chunk_size (int): Target size of each chunk in characters (default: 1000)
        chunk_overlap (int): Overlap between chunks to preserve context (default: 200)
        
    Returns:
        list[str]: List of text chunks ready for embedding
        
    Raises:
        ValueError: If input text is empty or chunks could not be generated
        
    Implementation Notes:
        - Uses recursive splitting with semantic separators (paragraphs > lines > words)
        - 200-char overlap ensures context isn't lost at chunk boundaries
        - 1000-char chunks balance between semantic coherence and embedding model limits
        
    Chunking Strategy:
        1. Try splitting on double newlines (paragraphs) - preserves logical sections
        2. Fall back to single newlines (sentences/lines)
        3. Fall back to spaces (word boundaries)
        4. Last resort: split on characters (rare, only for very long words)
        
    Why These Parameters?
        - chunk_size=1000: Fits comfortably in embedding model context (384-512 tokens)
        - chunk_overlap=200: ~20% overlap prevents information loss at boundaries
        - Separators preserve document structure and semantic meaning
    """
    
    # ============================================================================
    # INPUT VALIDATION
    # ============================================================================
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text. PDF extraction may have failed.")
    
    # ============================================================================
    # INITIALIZE SPLITTER
    # ============================================================================
    # RecursiveCharacterTextSplitter tries separators in order until chunk_size is met
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Use character count (can also use token count)
        separators=[
            "\n\n",  # Paragraph breaks (highest priority - best semantic boundaries)
            "\n",    # Line breaks (second priority - sentence/list boundaries)
            " ",     # Word breaks (third priority - avoid mid-word splits)
            ""       # Character-level (last resort - only for very long tokens)
        ]
    )
    
    # ============================================================================
    # PERFORM CHUNKING
    # ============================================================================
    chunks = text_splitter.split_text(text)
    
    # ============================================================================
    # VALIDATION: Ensure chunking succeeded
    # ============================================================================
    if not chunks:
        raise ValueError(f"Text splitting failed. Input length: {len(text)} chars")
    
    # Filter out any empty chunks (shouldn't happen, but defensive programming)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    if not chunks:
        raise ValueError("All chunks were empty after filtering. Check text quality.")
    
    logger.info(f"Created {len(chunks)} chunks from {len(text)} characters "
                f"(avg {len(text)//len(chunks)} chars/chunk)")
    
    return chunks


# ============================================================================
# USAGE EXAMPLE (for testing independently)
# ============================================================================
if __name__ == "__main__":
    # This allows testing the module independently
    print("PDF Handler Module - Ready for import")
    print("Example usage:")
    print("  text = get_pdf_text(uploaded_pdfs)")
    print("  chunks = get_text_chunks(text)")
# from pypdf import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# def get_pdf_text(pdf_docs):
#     """
#     Loops through the uploaded PDF files and extracts raw text.
#     """
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     """
#     Splits the raw text into manageable chunks for the Vector DB.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,       # Reduced to 1000 for better semantic chunks
#         chunk_overlap=200,     # Overlap ensures context isn't lost at cut points
#         separators=["\n\n", "\n", " ", ""]
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks
