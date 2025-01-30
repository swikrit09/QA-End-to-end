import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from components.settings_modal import settings_modal
from components.chat_container import create_chat_container
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Q&A with Documents", layout="wide")
st.title("🐱‍💻🐱‍🐉Q&A with Uploaded Documents")

groq_api_key = st.session_state.get("groq_api_key", None)
google_api_key = st.session_state.get("google_api_key", None)


settings_modal()
        
# Display current settings
    
if google_api_key and groq_api_key:      
    llm = None
    prompt = None
    # Initialize LLM and Prompt
    if "selected_model" in st.session_state:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=st.session_state.selected_model)
    else:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        
    if "prompt_template" not in st.session_state:
        prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
        )
    else:
        prompt = ChatPromptTemplate.from_template(st.session_state.prompt_template)

    # Initialize session state
    if "vectors" not in st.session_state:
        st.session_state.vectors = {}
        st.session_state.conversations = []


    def vector_embedding_from_upload(embedding_name, uploaded_files):
        """Creates and stores embeddings from uploaded files in the session state."""
        if embedding_name not in st.session_state.vectors:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            docs = []
            
            # Load PDFs from user uploads
            for uploaded_file in uploaded_files:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(uploaded_file.name)
                docs.extend(loader.load())

            # Text splitting and vector embedding
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            vectors = FAISS.from_documents(final_documents, embeddings)
            st.session_state.vectors[embedding_name] = {"vectors": vectors, "docs": docs}
            st.success(f"Vector Store '{embedding_name}' Created Successfully!")
        else:
            st.warning(f"Embedding '{embedding_name}' already exists!")


    def delete_embedding(embedding_name):
        """Deletes an embedding from the session state."""
        if embedding_name in st.session_state.vectors:
            del st.session_state.vectors[embedding_name]
            st.success(f"Vector Store '{embedding_name}' Deleted Successfully!")
        else:
            st.warning(f"Embedding '{embedding_name}' does not exist!")


    # Sidebar for managing embeddings
    st.sidebar.header("Manage Embeddings")
    embedding_name = st.sidebar.text_input("New Embedding Name")

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF Documents", type=["pdf"], accept_multiple_files=True
    )

    if st.sidebar.button("Create Embedding"):
        if embedding_name and uploaded_files:
            vector_embedding_from_upload(embedding_name, uploaded_files)
        elif not embedding_name:
            st.sidebar.error("Please provide a name for the embedding!")
        elif not uploaded_files:
            st.sidebar.error("Please upload at least one PDF document!")

    if st.sidebar.button("Delete Embedding"):
        if embedding_name:
            delete_embedding(embedding_name)
        else:
            st.sidebar.error("Please provide the name of the embedding to delete!")

    st.sidebar.subheader("Existing Embeddings")
    for name in st.session_state.vectors.keys():
        st.sidebar.write(f"- {name}")

    create_chat_container(llm,prompt,"Uploaded Document")
        
else:
    st.error("Please add the api keys in the settings")
