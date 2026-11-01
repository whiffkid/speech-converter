import io
import os
import re
import tempfile
from docx import Document
from faster_whisper import WhisperModel
from PIL import Image
import pytesseract
import streamlit as st
from symspellpy import SymSpell, Verbosity

st.set_page_config(page_title="Medical Media to Text", page_icon="🩺")
st.title("🩺 Medical Media to Text Converter")
st.write("Convert **Audio** (`.opus`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`) or **Images** (`.jpg`, `.jpeg`, `.png`) to text.")

AUDIO_EXTENSIONS = ["opus", "mp3", "wav", "m4a", "ogg", "flac"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]

# Common Indian clinical acronyms and terminology mappings
CLINICAL_FIXES = {
    r"\be\.d\b": "ED",
    r"\bed\b": "ED",
    r"\bcasualty\b": "Casualty",
    r"\bop\b": "OP",
    r"\bo\.p\b": "OP",
    r"\bip\b": "IP",
    r"\bbp\b": "BP",
    r"\bpr\b": "PR",
    r"\bspo2\b": "SpO2",
    r"\bgrbs\b": "GRBS",
    r"\becg\b": "ECG",
    r"\bhead ache\b": "headache",
    r"\bgiddiness\b": "giddiness",
    r"\bheaviness of head\b": "heaviness of head",
    r"\bhypertension\b": "hypertension",
    r"\bdiabetes mellitus\b": "diabetes mellitus",
    r"\bdyspnea\b": "dyspnea",
    r"\bsyncope\b": "syncope",
    r"\bparacetamol\b": "paracetamol",
    r"\bpantoprazole\b": "pantoprazole",
    r"\batorvastatin\b": "atorvastatin",
    r"\bmetformin\b": "metformin",
    r"\btelmisartan\b": "telmisartan",
    r"\bamlodipine\b": "amlodipine",
    r"\bclopidogrel\b": "clopidogrel",
    r"\baspirin\b": "aspirin",
    r"\bceftriaxone\b": "ceftriaxone",
    r"\bazithromycin\b": "azithromycin",
    r"\btroponin\b": "troponin",
    r"\bcbc\b": "CBC",
    r"\brft\b": "RFT",
    r"\blft\b": "LFT",
}

@st.cache_resource
def load_speech_engine():
    # Uses base.en or small.en under int8 CPU inference
    return WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=2)

def post_process_medical_text(text: str) -> str:
    corrected = text
    # Run regex-based medical entity and acronym alignment
    for pattern, replacement in CLINICAL_FIXES.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    
    # Capitalize start of sentences
    sentences = re.split(r'([.?!]\s*)', corrected)
    corrected = "".join([s.capitalize() if i % 2 == 0 else s for i, s in enumerate(sentences)])
    return corrected.strip()

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

    # --- AUDIO TRANSCRIPTION + MEDICAL COMPARISON ---
    elif file_ext in AUDIO_EXTENSIONS:
        st.audio(uploaded_file)

        if st.button("Transcribe Audio", type="primary"):
            with st.spinner("Transcribing and applying medical lexicon normalization..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as raw_audio:
                    raw_audio.write(uploaded_file.read())
                    raw_path = raw_audio.name

                try:
                    model = load_speech_engine()

                    # Transcribe with standard English decoder settings
                    segments, info = model.transcribe(
                        raw_path,
                        language="en",
                        temperature=0.0,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=400)
                    )

                    raw_transcription = " ".join([seg.text.strip() for seg in segments]).strip()

                    # Stage 2: Compare against clinical dictionary
                    final_transcription = post_process_medical_text(raw_transcription)

                    st.success("Transcription complete!")
                    st.text_area("Transcribed Clinical Text", value=final_transcription, height=250)

                    col1, col2 = st.columns(2)
                    with col1:
                        docx_file = create_docx(final_transcription, f"Clinical Note - {uploaded_file.name}")
                        st.download_button(
                            label="📥 Download Word (.docx)",
                            data=docx_file,
                            file_name=f"{base_name}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )
                    with col2:
                        st.download_button(
                            label="📄 Download Text (.txt)",
                            data=final_transcription,
                            file_name=f"{base_name}.txt",
                            mime="text/plain",
                        )
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                finally:
                    if os.path.exists(raw_path):
                        os.remove(raw_path)
