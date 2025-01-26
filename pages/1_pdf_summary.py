import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chat_models import ChatOpenAI
import numpy as np

# Función para extraer texto de los PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Función para dividir texto en fragmentos
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# Función para generar un resumen extractivo basado en embeddings
def summarize_with_embeddings(text_chunks, model="gpt-3.5-turbo", num_sentences=5):
    # Generar embeddings para cada fragmento
    embeddings = OpenAIEmbeddings()
    chunk_vectors = embeddings.embed_documents(text_chunks)
    global_vector = np.mean(chunk_vectors, axis=0).reshape(1, -1)

    # Calcular similitud entre fragmentos y el vector global
    similarities = [
        cosine_similarity(global_vector, np.array(chunk_vector).reshape(1, -1))[0][0]
        for chunk_vector in chunk_vectors
    ]

    # Seleccionar los fragmentos más relevantes
    top_indices = np.argsort(similarities)[-num_sentences:][::-1]
    top_chunks = [text_chunks[i] for i in top_indices]

    # Generar un resumen coherente con LLM
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = (
        "A continuación se presentan los fragmentos más relevantes de un documento. "
        "Genera un resumen coherente y explicativo que abarque todos los puntos importantes:\n\n"
        + "\n\n".join(top_chunks)
    )
    response = llm.predict(prompt)
    return response

# Función para generar un resumen abstractivo que destaque los puntos clave
def generate_key_points_summary(extractive_summary, model="gpt-3.5-turbo"):
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = (
        "A continuación se presenta un resumen de un documento. Extrae y lista los puntos clave "
        "del documento de forma clara y estructurada:\n\n"
        + extractive_summary
    )
    response = llm.predict(prompt)
    return response

# Función principal
def main():
    st.set_page_config(page_title="Resumen de PDFs Avanzado", page_icon="📄", layout="wide")

    st.title("📄 Resumen Avanzado de PDFs")
    st.write(
        """
        Suba un archivo PDF y obtenga resúmenes optimizados utilizando embeddings 
        para identificar las partes más relevantes y un modelo avanzado para generar un resumen fluido y destacar puntos clave.
        """
    )

    # Cargar documentos PDF
    pdf_docs = st.file_uploader("Sube tus PDFs para visualizar y resumir", type=["pdf"], accept_multiple_files=True)

    if pdf_docs:
        # Mostrar contenido del PDF
        st.subheader("📜 Contenido del PDF")
        pdf_text = get_pdf_text(pdf_docs)
        st.text_area("Texto extraído del PDF:", pdf_text, height=400)

        # Botón para generar los resúmenes
        if st.button("Generar Resúmenes"):
            st.subheader("📋 Resumen del Documento")
            if pdf_text:
                # Dividir el texto en fragmentos
                text_chunks = get_text_chunks(pdf_text)

                # Generar resumen extractivo basado en embeddings
                st.write("**Resumen Extractivo:**")
                extractive_summary = summarize_with_embeddings(text_chunks, num_sentences=7)
                st.write(extractive_summary)

                # Generar resumen abstractivo basado en LLM
                st.write("**Resumen Abstractivo (Puntos Clave):**")
                abstract_summary = generate_key_points_summary(extractive_summary)
                st.write(abstract_summary)

                # Botón para descargar el resumen abstractivo
                st.download_button(
                    "Descargar Resumen Abstractivo", abstract_summary, file_name="resumen_puntos_clave.txt"
                )
            else:
                st.warning("No se encontró texto en el PDF.")
    else:
        st.info("Sube uno o más archivos PDF para comenzar.")

if __name__ == "__main__":
    main()
