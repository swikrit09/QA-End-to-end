import os
import tempfile
from gitingest import ingest
from langchain_community.document_loaders import TextLoader
import streamlit as st

# Function to ingest GitHub repository
def ingest_github_repository(github_url):
    """
    Ingests a GitHub repository without assuming it's a coroutine.
    """
    try:
        st.write("Ingesting the GitHub repository...")
        result = ingest(github_url)  # Directly call ingest
        return result
    except Exception as e:
        raise RuntimeError(f"Error during GitHub ingestion: {e}")

# Streamlit Sidebar for GitHub URL input
with st.sidebar:
    st.header("Add your GitHub repository!")
    github_url = st.text_input("Enter GitHub repository URL", placeholder="GitHub URL")
    load_repo = st.button("Load Repository")

# Main Content
st.title("GitHub Repository Ingestion")
if github_url and load_repo:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_name = github_url.split('/')[-1]
            file_key = f"{repo_name}-ingested"

            # Check if repository has already been ingested
            if file_key not in st.session_state.get('file_cache', {}):
                # Ingest GitHub repository
                summary, tree, content = ingest_github_repository(github_url)

                # Save content to a temporary file
                content_path = os.path.join(temp_dir, f"{repo_name}_content.md")
                with open(content_path, "w", encoding="utf-8") as f:
                    f.write(content)

                # Load documents for embedding
                st.write("Loading documents for vectorization...")
                loader = TextLoader(content_path)
                docs = loader.load()

                # Save data to session state cache
                st.session_state.setdefault('file_cache', {})[file_key] = {
                    "summary": summary,
                    "tree": tree,
                    "content": content,
                    "docs": docs,
                }

                st.success(f"Successfully ingested repository: {repo_name}")
            else:
                st.warning("This repository has already been ingested!")
            
            # Display results
            st.subheader("Directory Tree")
            st.code(tree, language="plaintext")

            st.subheader("Summary")
            st.markdown(summary)

            st.subheader("Content")
            st.text(content[:1000])  # Show a snippet of the content for preview
    except Exception as e:
        st.error(f"An error occurred: {e}")

