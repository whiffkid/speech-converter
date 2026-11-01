import io
import os
import subprocess
import tempfile
from docx import Document
import noisereduce as nr
from PIL import Image
import pytesseract
import soundfile as sf
import streamlit as st
import whisper

st.set_page_config(page_title="Media to Text Converter", page_icon="📝")
st.title("📝 Media to Text Converter")
st.write("Convert **Audio** (`.mp3`, `.opus`, `.wav`, `.m4a`, `.ogg`, `.flac`) or **Images** (`.jpg`, `.jpeg`, `.png`) to text.")

AUDIO_EXTENSIONS = ["opus", "mp3", "wav", "m4a", "ogg", "flac"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]

# Permanent background vocabulary hint for medical, clinical notes, and abbreviations
DEFAULT_SYSTEM_PROMPT = (
    "Medical history, clinical notes, patient presented to the ED (Emergency Department) or OP. "
    "Hypertension, diabetes, fatigue, headache, fever, vitals, examination, treatment, diagnosis."
)

@st.cache_resource
def load_whisper_model(model_name: str):
    return whisper.load_model(model_name)

def create_docx(text: str, title: str) -> io.BytesIO:
    doc = Document()
    doc.add_heading(title, level=1)
    doc.add_paragraph(text)
    doc_io = io.BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    return doc_io

def denoise_audio(input_path: str, output_path: str):
    temp_wav = input_path + "_temp.wav"
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        samples, sr = sf.read(temp_wav, dtype="float32")
        reduced_noise = nr.reduce_noise(y=samples, sr=sr, stationary=True, prop_decrease=0.7)
        sf.write(output_path, reduced_noise, sr, format="WAV")
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

uploaded_file = st.file_uploader(
    "Upload a file", 
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
                            docx_file = create_docx(extracted_text, f"Transcription for {uploaded_file.name}")
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

        enable_denoise = st.checkbox("Apply Noise Reduction", value=False)

        if st.button("Transcribe Audio", type="primary"):
            with st.spinner("Transcribing..."):
                # Defaults to small.en for high English accuracy without prompting
                model = load_whisper_model("small.en")

                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as raw_audio:
                    raw_audio.write(uploaded_file.read())
                    raw_path = raw_audio.name

                clean_path = raw_path + "_clean.wav"
                transcription_target = raw_path

                try:
                    if enable_denoise:
                        with st.spinner("Filtering background audio..."):
                            denoise_audio(raw_path, clean_path)
                            transcription_target = clean_path

                    result = model.transcribe(
                        transcription_target,
                        fp16=False,
                        initial_prompt=DEFAULT_SYSTEM_PROMPT
                    )
                    transcription = result["text"].strip()

                    st.success("Transcription complete!")
                    st.text_area("Extracted Text", value=transcription, height=250)

                    col1, col2 = st.columns(2)
                    with col1:
                        docx_file = create_docx(transcription, f"Transcription for {uploaded_file.name}")
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
                    for p in [raw_path, clean_path]:
                        if os.path.exists(p):
                            os.remove(p)
