import io
import os
import tempfile
from docx import Document
from PIL import Image
import pytesseract
import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

st.set_page_config(page_title="Medical Media to Text", page_icon="🩺")
st.title("🩺 Medical Media to Text Converter")
st.write("Convert **Audio** (`.opus`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`) or **Images** (`.jpg`, `.jpeg`, `.png`) to text.")

AUDIO_EXTENSIONS = ["opus", "mp3", "wav", "m4a", "ogg", "flac"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]

# Load the dedicated medical speech recognition model from Hugging Face
@st.cache_resource
def load_medical_asr():
    model_id = "mahwizzzz/medwhishper"
    
    # Load model in float32 for CPU inference
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Initialize Hugging Face ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float32,
        device="cpu",
    )
    return pipe

def create_docx(text: str, title: str) -> io.BytesIO:
    doc = Document()
    doc.add_heading(title, level=1)
    doc.add_paragraph(text)
    doc_io = io.BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    return doc_io

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

    # --- AUDIO TRANSCRIPTION (Using Hugging Face Medical Whisper) ---
    elif file_ext in AUDIO_EXTENSIONS:
        st.audio(uploaded_file)

        if st.button("Transcribe Audio", type="primary"):
            with st.spinner("Loading medical model and transcribing clinical speech..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as raw_audio:
                    raw_audio.write(uploaded_file.read())
                    raw_path = raw_audio.name

                try:
                    asr_pipeline = load_medical_asr()

                    # Transcribe using Hugging Face chunking for audio stability
                    result = asr_pipeline(
                        raw_path,
                        chunk_length_s=30,
                        generate_kwargs={"language": "english", "task": "transcribe"}
                    )

                    transcription = result["text"].strip()

                    st.success("Transcription complete!")
                    st.text_area("Transcribed Medical Note", value=transcription, height=250)

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
                    st.error(f"Medical transcription failed: {e}")
                finally:
                    if os.path.exists(raw_path):
                        os.remove(raw_path)
