import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template

# === ENV SETUP ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
load_dotenv()

# === TEXT SPLITTER ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# === EMBEDDINGS ===
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)

# === VECTORSTORE BUILDER ===
def build_vectorstore_stream(pdf_docs):
    all_chunks = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                chunks = text_splitter.split_text(text)
                all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No text chunks were extracted from PDFs.")

    return FAISS.from_texts(texts=all_chunks, embedding=embedding_model)

# === LLM ===
def load_llm():
    return LlamaCpp(
        model_path="/media/anton/10bc98b6-5387-4b20-8ab7-624f1cf8c462/llama2/llama.cpp/models/llama2-q8.gguf",
        n_gpu_layers=10,
        n_ctx=1024,
        n_batch=32,
        temperature=0.1,
        top_p=0.95,
        verbose=False
    )

# === RAG CHAIN (LangChain 0.2 style) ===
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant specialized in answering questions using the provided context from uploaded PDF documents.

        Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.

        Context:
        {context}

        Chat history:
        {chat_history}

        Question: {question}
        Helpful Answer:
        """
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = (
        RunnableMap({
            "context": lambda x: retriever.invoke(x["question"]),
            "chat_history": lambda x: memory.load_memory_variables({})["chat_history"],
            "question": RunnablePassthrough()
        })
        | prompt_template
        | load_llm()
        | StrOutputParser()
    )

    return chain, memory

# === CHAT HANDLER ===
def handle_userinput(user_question):
    result = st.session_state.chain.invoke({"question": user_question})
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": result})

    for i, msg in enumerate(st.session_state.chat_history):
        template = user_template if msg["role"] == "user" else bot_template
        st.write(template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)

# === MAIN APP ===
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.chain:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                vectorstore = build_vectorstore_stream(pdf_docs)
                chain, memory = build_rag_chain(vectorstore)
                st.session_state.chain = chain
                st.session_state.chat_history = []

if __name__ == '__main__':
    main()
