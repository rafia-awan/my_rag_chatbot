import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader

# ---- Page Config ----
st.set_page_config(page_title="RAG Chatbot", page_icon="🧠")
st.title("📄 Document QA Chatbot")
st.markdown("Ask questions based on a software requirements document.")

# ---- Step 1: Load and split document ----
@st.cache_resource
def load_docs():
    url = "https://krazytech.com/projects/sample-software-requirements-specificationsrs-report-airline-database"
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

documents = load_docs()

# ---- Step 2: Create embeddings and vectorstore ----
@st.cache_resource
def get_vectorstore():
    url = "https://krazytech.com/projects/sample-software-requirements-specificationsrs-report-airline-database"
    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever()
# ---- Set up LLM ----
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key="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",
    openai_api_base="https://openrouter.ai/api/v1"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
# ---- Step 4: User input and output ----
query = st.text_input("💬 Ask your question:")

if query:
    with st.spinner("🤔 Thinking..."):
        response = qa_chain.run(query)
        st.markdown("### 📌 Answer:")
        st.write(response)
