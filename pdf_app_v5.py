import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load .env variables
load_dotenv()

# === TEXT EXTRACTION ===
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# === CHUNKING STRATEGY ===
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len
    )
    return text_splitter.split_text(text)

# === EMBEDDING + VECTORSTORE ===
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# === CUSTOM PROMPT ===
prompt_template = """
You are a helpful assistant specialized in answering questions using the provided context from uploaded PDF documents.

Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Chat history:
{chat_history}

Question: {question}
Helpful Answer:
"""

# === RAG PIPELINE SETUP ===
def get_conversation_chain(vectorstore):
    llm = LlamaCpp(
        model_path="/media/anton/10bc98b6-5387-4b20-8ab7-624f1cf8c462/llama2/llama.cpp/models/llama2-q8.gguf",
        n_gpu_layers=10,        # Try 10–20 on 8 GB; reduce if crash persists
        n_ctx=1024,             # Moderate context length
        n_batch=32,             # Smaller batches to prevent memory spikes
        temperature=0.1,        # More deterministic answers
        top_p=0.95,
        verbose=False
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return conversation_chain

# === CHAT HANDLER ===
def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# === MAIN STREAMLIT APP ===
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()

