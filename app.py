import io
import os
import re
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

# Direct clinical terminology & acronym corrections
CLINICAL_FIXES = {
    # Acronyms & Hospital departments
    r"\be\.d\b": "ED",
    r"\bed\b": "ED",
    r"\bcasualty\b": "Casualty",
    r"\bo\.?p\.?d?\b": "OP",
    r"\bi\.?p\.?d?\b": "IP",
    r"\bb\.?p\b": "BP",
    r"\bp\.?r\b": "PR",
    r"\bsp\s?o2\b": "SpO2",
    r"\bspo\s?2\b": "SpO2",
    r"\bgrbs\b": "GRBS",
    r"\be\.?c\.?g\b": "ECG",
    r"\bcbc\b": "CBC",
    r"\brft\b": "RFT",
    r"\blft\b": "LFT",
    r"\bcva\b": "CVA",
    r"\bcad\b": "CAD",
    r"\bcopd\b": "COPD",
    # Symptoms & findings
    r"\bhead ache\b": "headache",
    r"\bgiddiness\b": "giddiness",
    r"\bheaviness of head\b": "heaviness of head",
    r"\bhypertension\b": "hypertension",
    r"\bdiabetes mellitus\b": "diabetes mellitus",
    r"\bdyspnea\b": "dyspnea",
    r"\bsyncope\b": "syncope",
    r"\btroponin\b": "troponin",
    # Common medications
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
}

@st.cache_resource
def load_whisper():
    return WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)

def clean_clinical_text(text: str) -> str:
    cleaned = text
    for pattern, replacement in CLINICAL_FIXES.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    # Ensure proper sentence casing
    parts = re.split(r'([.?!]\s*)', cleaned)
    return "".join([p.capitalize() if i % 2 == 0 else p for i, p in enumerate(parts)]).strip()

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
            with st.spinner("Transcribing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as raw_audio:
                    raw_audio.write(uploaded_file.read())
                    raw_path = raw_audio.name

                try:
                    model = load_whisper()

                    # Transcribe cleanly without hallucination-inducing prompt stuffing
                    segments, info = model.transcribe(
                        raw_path,
                        language="en",
                        beam_size=3,
                        temperature=0.0,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=400)
                    )

                    raw_text = " ".join([seg.text.strip() for seg in segments]).strip()
                    final_text = clean_clinical_text(raw_text)

                    st.success("Transcription complete!")
                    st.text_area("Transcribed Text", value=final_text, height=250)

                    col1, col2 = st.columns(2)
                    with col1:
                        docx_file = create_docx(final_text, f"Clinical Note - {uploaded_file.name}")
                        st.download_button(
                            label="📥 Download Word (.docx)",
                            data=docx_file,
                            file_name=f"{base_name}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )
                    with col2:
                        st.download_button(
                            label="📄 Download Text (.txt)",
                            data=final_text,
                            file_name=f"{base_name}.txt",
                            mime="text/plain",
                        )
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                finally:
                    if os.path.exists(raw_path):
                        os.remove(raw_path)
