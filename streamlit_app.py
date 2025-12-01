import os
import tempfile
import streamlit as st

from agents import retrieve_top_k, answer_with_gemini
from ingest import ingest_docs

st.set_page_config(page_title="Multimodal Gemini Agent", page_icon="🤖")

st.title("Multimodal Gemini Agent")
st.write(
    "Ask questions about your PDFs. Optionally upload an image (chart, screenshot, diagram)."
)

# Sidebar — ingestion
st.sidebar.header("PDF Ingestion")

if st.sidebar.button("Re-ingest PDFs from data/docs"):
    with st.spinner("Loading PDFs..."):
        try:
            ingest_docs()
            st.sidebar.success("Ingestion complete!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

st.sidebar.write("Ensure your PDFs are placed in the folder: `data/docs/`")

# Main section
st.write("### Ask a question")

question = st.text_area("Question")
uploaded_image = st.file_uploader("Optional image upload", type=["png", "jpg", "jpeg"])

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        st.info("Retrieving information and querying Gemini...")

        img_path = None
        if uploaded_image is not None:
            suffix = os.path.splitext(uploaded_image.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_image.read())
                img_path = tmp.name

        try:
            chunks = retrieve_top_k(question, k=5)
            if not chunks:
                st.error("No PDF chunks found. Run ingestion first.")
            else:
                answer = answer_with_gemini(question, chunks, img_path)
                st.write("### Answer")
                st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")
