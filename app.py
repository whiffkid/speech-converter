import os
import tempfile
import streamlit as st
import whisper

st.set_page_config(page_title="Audio to Text Converter", page_icon="🎙️")
st.title("🎙️ Audio to Text Converter")
st.write("Upload an audio file (`.opus`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`) to extract text.")

# Load the free open-source Whisper model into cache
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Expanded file uploader accepting multiple audio formats including opus
uploaded_file = st.file_uploader(
    "Upload audio file", 
    type=["opus", "mp3", "wav", "m4a", "ogg", "flac"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Convert to Text", type="primary"):
        with st.spinner("Processing audio... This might take a minute."):
            # Preserve the original file extension so ffmpeg handles decoding properly
            file_extension = os.path.splitext(uploaded_file.name)[1]
            if not file_extension:
                file_extension = ".mp3"

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_audio:
                temp_audio.write(uploaded_file.read())
                temp_audio_path = temp_audio.name

            try:
                result = model.transcribe(temp_audio_path)
                text = result["text"].strip()
                detected_lang = result.get("language", "unknown")

                st.success("Conversion finished!")
                st.caption(f"Detected language: **{detected_lang.upper()}**")
                st.text_area("Transcribed Text", value=text, height=250)

                base_name = uploaded_file.name.rsplit(".", 1)[0]
                st.download_button(
                    label="Download text file",
                    data=text,
                    file_name=f"{base_name}.txt",
                    mime="text/plain",
                )
            except Exception as err:
                st.error(f"Failed to process file: {err}")
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
