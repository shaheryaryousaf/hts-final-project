# Streamlit and utility imports
import os
import string
import random
import tempfile
import zipfile
from datetime import datetime
import streamlit as st
from streamlit_chat import message

# Langchain imports for document processing and chat functionalities
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# Langchain community imports for advanced functionalities
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI

# Other imports (Consider removing if not used in the application)
import pinecone

# Retrieving API keys and configurations from Streamlit's secret management
qdrant_url = st.secrets["QDRANT_URL"]       # URL for Qdrant vector DB
qdrant_api_key = st.secrets["QDRANT_API_KEY"]  # API key for Qdrant
pinecone_api_key = st.secrets["PINECONE_API_KEY"]  # API key for Pinecone vector database
pinecone_env = st.secrets["PINECONE_ENV"]    # Pinecone environment setting


# Initialize Pinecone service with the provided API key and environment setting
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)


# Options for text splitting methods
TEXT_SPLITTING = (
    "Split by Character",  
    "Recursive Split by Character",  
)

# Embedding model choices for text processing
EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5",  
    "intfloat/e5-large-v2"     
)


# Save uploaded file temporarily
# The function is designed to save an uploaded file to a temporary location on the server where the Streamlit app is running.
# It uses Python's tempfile module to create a temporary file.
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Function to extract and return text from an uploaded file, 
# supporting multiple file formats like DOCX, PDF, and ZIP, 
# with error handling and cleanup.
def get_file_text(uploaded_file):
    # Initialize an empty string for text
    text = ""

    # Save the uploaded file temporarily and get the file path
    temp_file_path = save_uploaded_file(uploaded_file)
    if temp_file_path:
        extension = os.path.splitext(uploaded_file.name)[1]

        # Load text based on file extension
        if extension in [".docx", ".pdf"]:
            try:
                # Call the common loadFile function with the file type
                loaded_text = loadFile(temp_file_path, 'pdf' if extension == ".pdf" else 'docx')
                text += ' '.join(loaded_text) if isinstance(loaded_text, list) else loaded_text
            except ValueError as e:
                text += f"Error loading file: {e}"
        elif extension == ".zip":
            text += loadZip(temp_file_path)
        else:
            text += "Invalid file\n"

        # Remove the temporary file
        os.remove(temp_file_path)

    return text


# Function to load and extract text content from a file. 
# Utilizes appropriate loader based on file_type ('pdf' or 'docx') 
# and returns a list with the content of each page/document.
def loadFile(file_path, file_type):
    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()
    elif file_type == 'docx':
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
    else:
        raise ValueError("Unsupported file type")

    return [doc.page_content for doc in documents]


# Load Zip FIles
# Function to load and extract text from all PDF and DOCX files within a ZIP archive. 
# Extracts files to a temporary directory and aggregates the text content.
def loadZip(zip_file_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        aggregated_text = ""

        for foldername, subfolders, filenames in os.walk(tmp_dir):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                file_type = 'pdf' if filename.endswith('.pdf') else 'docx' if filename.endswith('.docx') else None
                if file_type:
                    file_text = loadFile(file_path, file_type)
                    aggregated_text += " ".join(file_text) if isinstance(file_text, list) else file_text
        return aggregated_text


# Function to split a given text into smaller chunks using the specified text splitter type.
# Each chunk is wrapped in a Document object with metadata, creating a list of documents.
def createChunksByTextSplitter(text, file_name, text_splitter_type="Character"):
    """
    Args:
        text (str): The input text to be split into chunks.
        file_name (str): The name of the source file.
        text_splitter_type (str): The type of text splitter to use ("Character" or "Recursive").

    Returns:
        list: A list of Document objects containing the text chunks.
    """
    if text_splitter_type == "Character":
        content = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=70,
            length_function=len
        )
    elif text_splitter_type == "Recursive":
        content = RecursiveCharacterTextSplitter(
            is_separator_regex=False,
            chunk_size=500,
            chunk_overlap=70,
            length_function=len
        )
    else:
        raise ValueError("Unsupported text splitter type")

    chunks = content.split_text(text)
    doc_list = []
    for chunk in chunks:
        metadata = {"source": file_name}
        doc_string = Document(page_content=chunk, metadata=metadata)
        doc_list.append(doc_string)
    return doc_list



# Create Vector Store
# Function to create a vector store in Qdrant using a list of document chunks. 
# It handles exceptions and returns the knowledge base or None in case of an error.
def createVectorStoreQdrant(chunks, collection_name, embedding):
    # Initialize a variable to store the knowledge base (vector store).
    knowledge_base = None
    try:
        # Attempt to create a Qdrant knowledge base using the provided parameters.
        knowledge_base = Qdrant.from_documents(
            documents=chunks,            # List of document chunks.
            embedding=embedding,          # Embedding information.
            url=qdrant_url,               # Qdrant URL.
            prefer_grpc=True,             # Prefer gRPC for communication.
            api_key=qdrant_api_key,       # API key for authentication.
            collection_name=collection_name  # Name of the collection in Qdrant.
        )
    except Exception as e:
        st.write(f"Error: {e}")
    return knowledge_base



# Retrieval QA
# Function to create a Question-Answer (QA) chain for extracting answers from a context and question.
# It constructs prompts using a template, configures the QA chain, and returns the QA instance.
def get_qa_chain(vectorstore, num_chunks):
    try:
        # Initialize a PromptTemplate object for dynamic prompt generation.
        prompt_template = """
        You are trained to extract Answer from the given Context and Question. 
        Then, precise the Answer must be not more than 40 words. 
        Do not include any other text in the Answer. 
        Do not use any abusive or prohibited language in the answer, and always use a polite and gentle tone.
        Do not ask for personal information from the user.
        Try to answer in the most concise and understandable way.
        If the Answer is not found in the Context, then return "N/A," otherwise return the precise Answer.
        Context: {context}
        Question: {question}"""
        mprompt_url = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
            validate_template=False
        )

        # Configure chain type kwargs with the prompt template.
        chain_type_kwargs = {"prompt": mprompt_url}

        # Create a RetrievalQA instance for question answering.
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo-16k", api_key=st.session_state.api_key),  # Language model
            chain_type="stuff",                     # Chain type
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": num_chunks}
            ),                                     # Retriever based on the vector store
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

        return qa
    except Exception as e:
        st.error(f"Error initializing VectorDB: {e}")
        return None


# Main function
# Main function that sets up the Streamlit app for DocumentGPT.
# It configures the page, handles API key validation, file upload, processing, and the chat interface.
def main():
    # Page Setup
    st.set_page_config(page_title="DocumentGPT")
    st.title("DocumentGPT")
    st.subheader("Transform Your Documents into Conversational Partners!")

    st.write("Key Functionalities:")
    st.write("📤 Upload PDF, DOCX, ZIP")
    st.write("📄 Extract Text from Documents")
    st.write("💬 Interactive Chat Interface")
    st.write("🤖 AI-Powered Document Analysis")
    st.write("🔒 Secure and Efficient Storage")


    # Setup session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "key_validate" not in st.session_state:
        st.session_state.key_validate = False

    # Sidebar
    with st.sidebar:
        st.header("Upload File")

        api = st.text_input("Enter your API Key", type="password")
        validate = st.button("Validate")

        if validate:
            if api.startswith("sk-"):
                st.session_state['api_key'] = api
                st.session_state['key_validate'] = True
            elif not api.startswith('sk-'):
                st.write("Please add a valid API Key")
            else:
                st.write("Please add an API Key")

        if st.session_state['key_validate']:
            # Upload File
            uploaded_files = st.file_uploader(
                "Upload File", accept_multiple_files=True, type=["pdf", "docx", "zip"])
            embedding = st.selectbox("Select Embedding Model", options=EMBEDDING_MODEL)
            splitting_type = st.selectbox("Select Splitting Type", options=TEXT_SPLITTING)
            process = st.button("Process")

            

            # Process Initiate
            if process:
                text_chunks = []
                for file in uploaded_files:
                    file_name = file.name
                    # Get Files
                    text = get_file_text(file)

                    # Create Text Chunks
                    if splitting_type == "Split by Character":
                        chunks = createChunksByTextSplitter(text, file_name, text_splitter_type="Character")
                    else:
                        chunks = createChunksByTextSplitter(text, file_name, text_splitter_type="Recursive")
                    
                    text_chunks.extend(chunks)
                if text_chunks:
                    # Get vector store
                    curr_date = datetime.now().strftime("%Y%m%d%H%M%S")
                    collection_name = "".join(random.choices(
                        string.ascii_lowercase, k=4)) + curr_date
                    
                    # Set up embedding
                    if embedding == "BAAI/bge-small-en-v1.5":
                        embedding_type = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                    else:
                        embedding_type = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

                    vectorStore = createVectorStoreQdrant(text_chunks, collection_name, embedding_type)
                    if vectorStore:
                        st.write("Vector store created successfully")

                    num_chunks = 4
                    st.session_state.conversation = get_qa_chain(
                        vectorStore, num_chunks)
                    st.session_state.processComplete = True

    # Chat
    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            handel_userinput(user_question)


# Handle User Input
# Function to handle user input, generate a response, and update the chat history.
# It displays a loading spinner while generating the response and organizes the layout of input and response messages.
def handel_userinput(user_question):
    with st.spinner('Generating response...'):
        result = st.session_state.conversation({"query": user_question})

        # Check if 'source_documents' is not empty
        if result['source_documents']:
            source = result['source_documents'][0].metadata['source']
        else:
            source = "No source available"

        response = result['result']
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(
            f"{response} \n Source Document: {source}")

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages, is_user=True, key=str(i))
            else:
                message(messages, key=str(i))

# Call Main function
if __name__ == "__main__":
    main()
