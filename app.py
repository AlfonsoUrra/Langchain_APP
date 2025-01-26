import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import openai
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval
from htmlTemplates import css, bot_template, user_template


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
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = openai.OpenAIChatModel()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = conversational_retrieval.from_llm(
    llm = llm,
    retriever = vectorstore.as_retriever(),
    memory = memory
    )
    return conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDFs  CHAT", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
                       
    st.header("Multiple PDF's CHAT application: books:")
    st.text_input("Ask any question about the documents and I will try to answer it.")

    st.write(user_template.replace("{{MSG}}", "Hello Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)


    with st.sidebar:
        st.subheader("My documents")
        pdf_docs = st.file_uploader("Upload PDF", type=['pdf'], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get ghe text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create the vector store
                vectorstore = get_vector_store(text_chunks)

                # conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore) # para que streamlit no reinicie variable

    


if __name__ == '__main__':
    main()