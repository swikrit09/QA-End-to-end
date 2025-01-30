import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"  # FastAPI backend URL

st.set_page_config(page_title="Q&A with Documents", layout="wide")
st.title("Q&A with Uploaded Documents")

# Sidebar for managing embeddings
st.sidebar.header("Manage Embeddings")
embedding_name = st.sidebar.text_input("New Embedding Name")
git_url = st.sidebar.text_input("Enter GitHub URL")

if st.sidebar.button("Create Embedding"):
    if embedding_name and git_url:
        response = requests.post(f"{API_BASE}/ingest/", json={
            "embedding_name": embedding_name,
            "github_url": git_url
        })
        if response.status_code == 200:
            st.sidebar.success(response.json().get("message"))
        else:
            st.sidebar.error(response.json().get("detail"))
    else:
        st.sidebar.error("Please provide both embedding name and GitHub URL!")

# Main Q&A interface
st.subheader("Ask Questions")
selected_embedding = st.sidebar.selectbox("Select Embedding", options=[], key="embedding_select")
question = st.text_input("Enter Your Question:")

if st.button("Ask"):
    if selected_embedding and question:
        response = requests.post(f"{API_BASE}/ask/", json={
            "embedding_name": selected_embedding,
            "question": question
        })
        if response.status_code == 200:
            st.write("**Answer:**", response.json().get("answer"))
        else:
            st.error(response.json().get("detail"))
    else:
        st.error("Please select an embedding and enter a question!")
