import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.translation import get_language_names, translate
from components.settings_modal import settings_modal
from dotenv import load_dotenv
import time
from streamlit_modal import Modal

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

    # Main Q&A interface

    prompt1 = st.text_input("Enter Your Question From Website URL")
    if prompt1:
        if st.session_state.vectors:
            selected_embedding = st.sidebar.selectbox("Select Embedding", st.session_state.vectors.keys())
            if selected_embedding:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors[selected_embedding]["vectors"].as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                # Process the query
                start = time.process_time()
                if len(st.session_state.conversations) == 0 or prompt1 != st.session_state.conversations[-1]["question"]:
                    response = retrieval_chain.invoke({'input': prompt1})
                    response_time = time.process_time() - start
                    # With a Streamlit expander
                    with st.expander("Document Similarity Search"):
                        for i, doc in enumerate(response["context"]):
                            st.write(doc.page_content)
                            st.write("--------------------------------")


                    # Save the conversation
                    st.session_state.conversations.append({"question": prompt1, "answer": response['answer'],"response_time":response_time })

                # Retrieve the last response
                last_response = st.session_state.conversations[-1]["answer"]
                time = st.session_state.conversations[-1]['response_time']
                # Display the response
                st.write(f"Response Time: {time:.2f} seconds")
                st.write(last_response)

                selected_language = st.selectbox(
                    "Translate to",
                    (get_language_names()),
                    index=0,  # Default to the first language in the list
                )
                if selected_language and selected_language != "Select":
                    translation = translate("English", selected_language, last_response)
                    with st.expander("Translated Response"):
                        st.write(translation)

        else:
            st.error("No vector embeddings found. Please create one first!")


    # Display saved conversations
    st.subheader("Saved Conversations")
    if st.session_state.conversations:
        for idx, convo in enumerate(st.session_state.conversations):
            with st.expander(f"Conversation {idx + 1}"):
                st.write(f"**Question:** {convo['question']}")
                st.write(f"**Answer:** {convo['answer']}")
    else:
        st.write("No conversations saved yet.")

else:
    st.error("Please add the api keys in the settings")

