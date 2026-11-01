import io
import os
import tempfile
from docx import Document
from faster_whisper import WhisperModel
from PIL import Image
import pytesseract
import streamlit as st

st.set_page_config(page_title="Medical Media to Text", page_icon="🩺")
st.title("🩺 Medical Media to Text Converter")
st.write("Convert **Audio** (`.opus`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`) or **Images** (`.jpg`, `.jpeg`, `.png`) to text.")

AUDIO_EXTENSIONS = ["opus", "mp3", "wav", "m4a", "ogg", "flac"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]

# Load faster-whisper with distil-large-v3 (Large-v3 accuracy in a compact footprint)
# Fallback to small.en if memory is constrained
@st.cache_resource
def load_whisper_engine():
    try:
        # distil-large-v3 provides Large-v3 vocabulary while fitting inside ~750MB RAM under int8
        return WhisperModel("distil-whisper/distil-large-v3", device="cpu", compute_type="int8", cpu_threads=2)
    except Exception:
        return WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=2)

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

    # --- AUDIO TRANSCRIPTION ---
    elif file_ext in AUDIO_EXTENSIONS:
        st.audio(uploaded_file)

        if st.button("Transcribe Audio", type="primary"):
            with st.spinner("Transcribing with deterministic clinical decoding..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as raw_audio:
                    raw_audio.write(uploaded_file.read())
                    raw_path = raw_audio.name

                try:
                    model = load_whisper_engine()

                    # Decoding parameters tuned for clinical terminology:
                    # 1. language="en": Prevents language hunting on accents
                    # 2. temperature=0.0: Strictly deterministic (no random guessing)
                    # 3. beam_size=5 & patience=1.0: Full search path exploration
                    # 4. initial_prompt: Primes the decoder on hospital acronyms (ED, casualty, vitals)
                    segments, info = model.transcribe(
                        raw_path,
                        language="en",
                        temperature=0.0,
                        beam_size=5,
                        patience=1.0,
                        vad_filter=True,
                        initial_prompt="Clinical admission note: Patient presented to ED (Emergency Department) with hypertension, fatigue, heaviness of head."
                    )

                    transcription = " ".join([segment.text.strip() for segment in segments]).strip()

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
                finally:
                    if os.path.exists(raw_path):
                        os.remove(raw_path)
