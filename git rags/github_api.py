from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from gitingest import ingest
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# LLM setup
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Store vector embeddings
vector_stores = {}

class GitHubIngestRequest(BaseModel):
    embedding_name: str
    github_url: str

class QuestionRequest(BaseModel):
    embedding_name: str
    question: str

@app.post("/ingest/")
def ingest_github(request: GitHubIngestRequest):
    if request.embedding_name in vector_stores:
        raise HTTPException(status_code=400, detail="Embedding already exists!")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Step 1: Ingest GitHub repository
    try:
        summary, tree, content = ingest(request.github_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    
    # Step 2: Save content and create vectors
    with tempfile.TemporaryDirectory() as temp_dir:
        content_path = os.path.join(temp_dir, "repo_content.md")
        with open(content_path, "w", encoding="utf-8") as f:
            f.write(content)

        loader = TextLoader(content_path)
        docs = loader.load()

    # Step 3: Split content
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_documents, embeddings)

    # Save vectors and docs in memory
    vector_stores[request.embedding_name] = {"vectors": vectors, "docs": docs}
    return {"message": f"Embedding '{request.embedding_name}' created successfully!"}

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    if request.embedding_name not in vector_stores:
        raise HTTPException(status_code=404, detail="Embedding not found!")

    vectors = vector_stores[request.embedding_name]["vectors"]
    retriever = vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': request.question})
    return {"answer": response['answer']}
