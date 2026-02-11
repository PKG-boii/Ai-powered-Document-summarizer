import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 🔐 Load Groq API key
GROQ_API_KEY = GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in environment variables.")
    st.stop()


# App Header
st.header("📄 AI PDF Chatbot (Groq Powered)")

# Sidebar
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(
        "Upload a PDF file and start asking questions",
        type="pdf"
    )


# Extract text from PDF
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 🔹 Local embeddings (NO OpenAI needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User input
    user_question = st.text_input("Type your question here")

    if user_question:
        # Similarity search
        docs = vector_store.similarity_search(user_question)

        # Groq LLM
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        # Prompt (LCEL)
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using ONLY the context below.
            If the answer is not in the context, say "I don't know".

            Context:
            {context}

            Question:
            {question}
            """
        )

        # Chain (modern LangChain)
        chain = prompt | llm | StrOutputParser()

        # Run chain
        response = chain.invoke({
            "context": "\n\n".join([doc.page_content for doc in docs]),
            "question": user_question
        })

        st.write("### Answer")
        st.write(response)
