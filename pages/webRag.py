import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from components.chat_container import create_chat_container
from components.settings_modal import settings_modal
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="WebRAG", layout="wide")
st.title("🦚🦜Q&A with Website")
   
groq_api_key = st.session_state.get("groq_api_key", None)
google_api_key = st.session_state.get("google_api_key", None)

settings_modal()

if groq_api_key and google_api_key:
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



    if "embedding_created" not in st.session_state:
        st.session_state.embedding_created = {}

    def vector_embedding_from_website(embedding_name, website_url):
        """
        Creates and stores embeddings from a website repository in FAISS.
        """
        if embedding_name in st.session_state.embedding_created:
            st.warning(f"Embedding '{embedding_name}' already exists!")
            return  # Skip execution if already created

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.write("Ingesting the Website...")

        # Load website content
        loader = WebBaseLoader(website_url)
        docs = loader.load()

        # Split content into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:40])
        vectors = FAISS.from_documents(final_documents, embeddings)
        st.session_state.vectors[embedding_name] = {"vectors": vectors, "docs": docs}

        # Mark embedding as created
        st.session_state.embedding_created[embedding_name] = True
        st.success(f"Vector Store '{embedding_name}' Created Successfully!")


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

    website_url = st.sidebar.text_input("Enter website url")
    if st.sidebar.button("Create Embedding"):
        if embedding_name and website_url:
            vector_embedding_from_website(embedding_name, website_url)
        elif not embedding_name:
            st.sidebar.error("Please provide a name for the embedding!")
        elif not website_url:
            st.sidebar.error("Please upload at least one PDF document!")

    if st.sidebar.button("Delete Embedding"):
        if embedding_name:
            delete_embedding(embedding_name)
        else:
            st.sidebar.error("Please provide the name of the embedding to delete!")

    st.sidebar.subheader("Existing Embeddings")
    for name in st.session_state.vectors.keys():
        st.sidebar.write(f"- {name}")

    create_chat_container(llm,prompt,"Website URL")
    
else:
    st.error("Please add the api keys in the settings")

