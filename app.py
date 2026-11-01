import io
import os
import tempfile
from docx import Document
from google import genai
from PIL import Image
import pytesseract
import streamlit as st

st.set_page_config(page_title="Medical Media to Text", page_icon="🩺")
st.title("🩺 Medical Media to Text Converter")
st.write("Convert **Audio** (`.opus`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`) or **Images** (`.jpg`, `.jpeg`, `.png`) to text.")

AUDIO_EXTENSIONS = ["opus", "mp3", "wav", "m4a", "ogg", "flac"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key:", type="password")

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

    # --- AUDIO TRANSCRIPTION (Medical Understanding) ---
    elif file_ext in AUDIO_EXTENSIONS:
        st.audio(uploaded_file)

        if st.button("Transcribe Audio", type="primary"):
            if not api_key:
                st.error("Missing Gemini API Key! Add it in Streamlit Secrets.")
            else:
                with st.spinner("Transcribing clinical audio..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as raw_audio:
                        raw_audio.write(uploaded_file.read())
                        raw_path = raw_audio.name

                    uploaded_file_ref = None
                    try:
                        client = genai.Client(api_key=api_key)

                        mime_map = {
                            "opus": "audio/ogg",
                            "mp3": "audio/mp3",
                            "wav": "audio/wav",
                            "m4a": "audio/mp4",
                            "ogg": "audio/ogg",
                            "flac": "audio/flac",
                        }
                        chosen_mime = mime_map.get(file_ext, "audio/mp3")

                        # Upload to Gemini File API
                        uploaded_file_ref = client.files.upload(
                            file=raw_path,
                            config=dict(mime_type=chosen_mime)
                        )

                        system_prompt = (
                            "You are an expert medical transcriptionist specializing in Indian English clinical dictations. "
                            "Transcribe the audio accurately verbatim. Format clinical terms, medical abbreviations "
                            "(e.g., ED, BP, PR, SpO2, GRBS, ECG), symptoms, and drug names properly. "
                            "Return only the final transcription without introductory conversational phrases."
                        )

                        response = client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=[uploaded_file_ref, system_prompt]
                        )

                        transcription = response.text.strip()

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
                        if uploaded_file_ref:
                            try:
                                client.files.delete(name=uploaded_file_ref.name)
                            except Exception:
                                pass
                        if os.path.exists(raw_path):
                            os.remove(raw_path)
