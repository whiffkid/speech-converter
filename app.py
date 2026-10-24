import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="MP3 to Text", page_icon="📄")
st.title("📄 MP3 to Text Converter")

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

uploaded_file = st.file_uploader("Upload your MP3 file", type=["mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("Convert to Text", type="primary"):
        with st.spinner("Processing MP3 file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_audio.write(uploaded_file.read())
                temp_audio_path = temp_audio.name

            try:
                result = model.transcribe(temp_audio_path)
                text = result["text"].strip()

                st.success("Conversion finished!")
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
