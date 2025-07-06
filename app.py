import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Initialize LLM
llm = OllamaLLM(model="tinyllama", base_url="http://localhost:11434")

# Initialize embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Streamlit UI
st.title("📄 AI Resume Analyzer")
st.write("Upload multiple resumes, enter a job description, and rank candidates!")

# Upload resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF only)", accept_multiple_files=True, type=["pdf"])

candidates = []  # Store candidate names and text content

if uploaded_files:
    all_chunks = []
    
    # Process uploaded resumes
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Extract candidate name from filename
        candidate_name = os.path.splitext(uploaded_file.name)[0]
        
        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load_and_split()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)

        # Store candidate details
        candidate_text = " ".join([chunk.page_content for chunk in chunks])
        candidates.append((candidate_name, candidate_text))

        # Delete temp file
        os.remove(temp_pdf_path)

    # Store in ChromaDB
    db = Chroma.from_documents(all_chunks, embeddings_model, persist_directory="./chroma_db")
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Job Description Input
    job_description = st.text_area("📝 Enter the Job Description:")

    # Rank candidates function
    def rank_candidates(job_description, candidates):
        rankings = []
        
        for candidate_name, candidate_text in candidates:
            query = f"Compare this candidate's experience with the following job description: {job_description}"
            response = qa_chain.invoke({"query": query})
            rankings.append((candidate_name, response["result"]))

        return sorted(rankings, key=lambda x: x[1], reverse=True)  # Sort by relevance

    # Rank Candidates & Display Results
    if job_description and candidates:
        candidate_rankings = rank_candidates(job_description, candidates)
        
        st.subheader("🏆 Best Fit Candidates:")
        for idx, (candidate_name, score) in enumerate(candidate_rankings, 1):
            st.write(f"**{idx}. {candidate_name}** - {score}")

    # Chat Interface
    st.subheader("💬 Chat with AI about the resumes")
    user_query = st.text_input("Ask a question about the resumes:")

    if user_query:
        response = qa_chain.invoke({"query": user_query})
        st.write("🤖 AI Response:")
        st.write(response["result"])

