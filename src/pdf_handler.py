from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_pdf_text(pdf_docs):
    """
    Loops through the uploaded PDF files and extracts raw text.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits the raw text into manageable chunks for the Vector DB.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Reduced to 1000 for better semantic chunks
        chunk_overlap=200,     # Overlap ensures context isn't lost at cut points
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks