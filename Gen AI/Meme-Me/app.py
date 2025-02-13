import streamlit as st
import whisper
import tempfile
import os
from dotenv import load_dotenv
from pydub import AudioSegment
import openai
from PIL import Image, ImageDraw, ImageFont
import requests

load_dotenv()
# Set your OpenAI API key (replace with your actual key or set via your .env file)
openai.api_key = os.getenv("OPENAI_API_KEY")



# Streamlit app layout
st.logo(
    "logo.png",
    size="medium",
    link="https://github.com/ChaudharyUsman",
)

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()



# Function to transcribe speech to text using Whisper
def transcribe_audio(audio_file):
    # Convert the uploaded file to a format Whisper can understand (wav)
    audio = AudioSegment.from_file(audio_file)
    
    # Save the audio as a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        audio.export(tmp_file, format="wav")
        tmp_file_path = tmp_file.name

    # Load the audio into Whisper
    audio_array = whisper.load_audio(tmp_file_path)
    
    # Transcribe the audio
    result = model.transcribe(audio_array)
    
    # Cleanup: Delete the temporary file after processing
    os.remove(tmp_file_path)
    
    return result['text']


transcription = None

st.title("AI Facts Finder")
st.write("Record or upload an audio clip, let AI generate a find facts about cricket!")
audio_file = st.file_uploader("Upload your voice recording (in mp3 or wav format)", type=['mp3', 'wav'])


if audio_file:
    transcription = transcribe_audio(audio_file) 
    st.write("Transcription:",transcription)
# Process recorded audio input using st.audio_input
st.write("OR")
audio_value = st.audio_input("Record a voice message to transcribe", key="audio_input_1")
if audio_value:
    # Write the uploaded audio to a temporary file and then pass its path to model.transcribe
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_value.read())
        temp_audio_path = temp_audio.name
    result = model.transcribe(temp_audio_path)
    transcription = result["text"]
    st.write(transcription)
    os.remove(temp_audio_path)


if transcription:
    if st.button("Facts"):
        prompt = f"Generate a facts about cricket for the following text and it has be accurate fact:\n\n{transcription}\n\n Meme Caption:"
        with st.spinner("Generating Facts..."):
            response = openai.chat.completions.create(
             model="gpt-4o",
             messages=[
                 {"role": "assistant", "content": "You are a helpful assistant to asnwer about he facts of cricket."},
        {
            "role": "user",
            "content": prompt
        }
    ]
)

        st.write(response.choices[0].message.content)
