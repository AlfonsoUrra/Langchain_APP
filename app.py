import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os



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
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = vectorstore.as_retriever(),
    memory = memory
    )
    return conversation_chain

def handler_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
    if st.button("Descargar conversación"):
        chat_history = "\n".join(
            f"Usuario: {message.content}" if i % 2 == 0 else f"Bot: {message.content}"
            for i, message in enumerate(st.session_state.chat_history)
        )
        st.download_button("Descargar conversación", chat_history, "chat_history.txt")



def main():
    load_dotenv()
    st.set_page_config(page_title="Tu Chat de PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
                       
    image_path = os.path.join("img", "stack-of-books.png")

    # Encabezado principal con imagen de libro
    col1, col2 = st.columns([7, 1])  # Divide en columnas para la imagen y el texto
    with col2:
        st.image(
            image_path,  # Ruta relativa de la imagen
            width=50,  # Ancho de la imagen
        )
    with col1:
        st.header("Tu Chat de PDFs")


    user_question = st.text_input("Pregunta cualquier cosa sobre los documentos y trataré de responderte:")
    if user_question:
        handler_userinput(user_question)


    with st.sidebar:
        st.subheader("Mis documentos")
        pdf_docs = st.file_uploader("Upload PDF", type=['pdf'], accept_multiple_files=True)
        if st.button("Procesar"):
            with st.spinner("Procesando..."):
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