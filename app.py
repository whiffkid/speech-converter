import io
import os
from docx import Document
from PIL import Image
import pytesseract
import requests
import streamlit as st

st.set_page_config(page_title="Medical Media to Text", page_icon="🩺")
st.title("🩺 Medical Media to Text Converter")
st.write("Transcribe clinical audio (`.opus`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`) or images (`.jpg`, `.jpeg`, `.png`).")

AUDIO_EXTENSIONS = ["opus", "mp3", "wav", "m4a", "ogg", "flac"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"

# Fetch token from Streamlit Secrets or sidebar fallback
hf_token = st.secrets.get("HF_TOKEN")
if not hf_token:
    hf_token = st.sidebar.text_input("Hugging Face Token (starts with hf_):", type="password")

def create_docx(text: str, title: str) -> io.BytesIO:
    doc = Document()
    doc.add_heading(title, level=1)
    doc.add_paragraph(text)
    doc_io = io.BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    return doc_io

def query_whisper_large(audio_bytes: bytes, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(API_URL, headers=headers, data=audio_bytes, timeout=60)
    return response.json()

uploaded_file = st.file_uploader(
    "Upload audio or image file", 
    type=AUDIO_EXTENSIONS + IMAGE_EXTENSIONS
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    base_name = uploaded_file.name.rsplit(".", 1)[0]

    # --- IMAGE OCR ---
    if file_ext in IMAGE_EXTENSIONS:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Extract Text from Image", type="primary"):
            with st.spinner("Extracting text with OCR..."):
                try:
                    extracted_text = pytesseract.image_to_string(image).strip()
                    if extracted_text:
                        st.success("Extraction complete!")
                        st.text_area("Extracted Text", value=extracted_text, height=250)

                        col1, col2 = st.columns(2)
                        with col1:
                            docx_file = create_docx(extracted_text, f"Document - {uploaded_file.name}")
                            st.download_button(
                                label="📥 Download Word (.docx)",
                                data=docx_file,
                                file_name=f"{base_name}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            )
                        with col2:
                            st.download_button(
                                label="📄 Download Text (.txt)",
                                data=extracted_text,
                                file_name=f"{base_name}.txt",
                                mime="text/plain",
                            )
                    else:
                        st.warning("No readable text found in this image.")
                except Exception as e:
                    st.error(f"Error extracting text: {e}")

    # --- AUDIO TRANSCRIPTION (Whisper Large v3) ---
    elif file_ext in AUDIO_EXTENSIONS:
        st.audio(uploaded_file)

        if st.button("Transcribe Audio", type="primary"):
            if not hf_token:
                st.error("Missing Hugging Face Token! Please add it to your Streamlit Secrets or sidebar.")
            else:
                with st.spinner("Transcribing with Whisper Large v3..."):
                    try:
                        audio_data = uploaded_file.read()
                        output = query_whisper_large(audio_data, hf_token)

                        if "error" in output:
                            st.error(f"Inference error: {output['error']}")
                        else:
                            transcription = output.get("text", "").strip()

                            st.success("Transcription complete!")
                            st.text_area("Transcribed Text", value=transcription, height=250)

                            col1, col2 = st.columns(2)
                            with col1:
                                docx_file = create_docx(transcription, f"Clinical Note - {uploaded_file.name}")
                                st.download_button(
                                label="📥 Download Word (.docx)",
                                data=docx_file,
                                file_name=f"{base_name}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            )
                            with col2:
                                st.download_button(
                                    label="📄 Download Text (.txt)",
                                    data=transcription,
                                    file_name=f"{base_name}.txt",
                                    mime="text/plain",
                                )
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
