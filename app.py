import io
import time
from docx import Document
from google import genai
from google.genai import types
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

    # --- AUDIO TRANSCRIPTION (With Automatic Retry & Fallback) ---
    elif file_ext in AUDIO_EXTENSIONS:
        st.audio(uploaded_file)

        if st.button("Transcribe Audio", type="primary"):
            if not api_key:
                st.error("Missing Gemini API Key! Add it in Streamlit Secrets.")
            else:
                with st.spinner("Transcribing clinical audio..."):
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

                        # Pass bytes inline for fast execution
                        audio_part = types.Part.from_bytes(
                            data=uploaded_file.getvalue(),
                            mime_type=chosen_mime
                        )

                        system_prompt = (
                            "You are an expert medical transcriptionist specializing in Indian English clinical dictations. "
                            "Transcribe the audio accurately verbatim. Format clinical terms, medical abbreviations "
                            "(e.g., ED, BP, PR, SpO2, GRBS, ECG), symptoms, and drug names properly. "
                            "Return only the final transcription text without conversational preamble or markdown code blocks."
                        )

                        # Cascade of supported active flash models
                        candidate_models = [
                            "gemini-3.8-flash",
                            "gemini-3.6-flash",
                            "gemini-3.5-flash-lite",
                        ]

                        response = None
                        last_exception = None

                        # Retry loop with backoff and model degradation
                        for model_name in candidate_models:
                            for attempt in range(2):  # up to 2 attempts per model
                                try:
                                    response = client.models.generate_content(
                                        model=model_name,
                                        contents=[audio_part, system_prompt]
                                    )
                                    if response and response.text:
                                        break
                                except Exception as err:
                                    last_exception = err
                                    err_str = str(err)
                                    # If server is overloaded (503/429), wait and retry
                                    if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                                        time.sleep(2)
                                        continue
                                    else:
                                        break  # non-transient error, move to next model
                            if response and response.text:
                                break

                        if not response or not response.text:
                            raise last_exception

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
