import streamlit as st
#langchain_community.embeddings import LlamaCppEmbeddings
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain_community.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
#from langchain.llms import LlamaCpp
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import LlamaCppEmbeddings
#from langchain.embeddings import LlamaCppEmbeddings
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=50,
        chunk_overlap=10,
        length_function=len
    )
    print("split text")
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    #embeddings  = LlamaCppEmbeddings(model_path="/mnt/sda1/llama2/llama.cpp/models/llama2-q8.gguf", n_gpu_layers=20, n_ctx = 1700, )
    print("geting HuggingFace embedings")
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs = {"device" : "cpu"})
    #db.save_local("faiss_index")
    print("adding text chuncs to FAIAA")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("Save vector store")
    vectorstore.save_local("faiss_index")
    return vectorstore


def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    llm = LlamaCpp(
#    model_path="/mnt/sda1/llama2/llama.cpp/models/llama2-q8.gguf",
    model_path="/media/anton/HIKSEMI/llama2/llama2-q8.gguf",
    n_ctx=512,
    n_batch=126
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
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
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                print("get pdf text")
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                print("get text chuncs")
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                print("get chunks for vector store")
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                print("start conversation")

if __name__ == '__main__':
    main()
