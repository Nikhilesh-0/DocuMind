from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
def get_conversational_chain():
    """
    Creates a chain that takes a list of documents and a question, 
    and returns an answer.
    """
    
    # 1. Define the Prompt (No changes here)
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in provided context just say, "The answer is not available in the context", don't provide the wrong answer.
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    
    # 2. Initialize Model
    model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    
    # 3. Create Prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # 4. Create the Chain (The Modern Way)
    # This automatically handles "stuffing" the documents into the {context} variable
    chain = create_stuff_documents_chain(model, prompt)
    
    return chain