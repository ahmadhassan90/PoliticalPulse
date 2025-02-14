import streamlit as st
import whisper
import yt_dlp
import google.generativeai as genai
import os
import fal_client
import re
from PIL import Image
import requests
from io import BytesIO
import subprocess
import imageio_ffmpeg

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_binary()


FAL_KEY = st.secrets["FAL_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


# 🔹 Configure API Keys

os.environ["FAL_KEY"] = FAL_KEY
genai.configure(api_key=GEMINI_API_KEY)

# 🔹 Load Whisper Model (Use Medium for Better Urdu-to-English Translation)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")  # Change "medium" to "large" if needed


# 🔹 Download Audio from YouTube
def download_audio(youtube_url):
    
    ydl_opts = {
    'format': 'bestaudio/best',
    'ffmpeg_location': FFMPEG_PATH,  # Use the Python-installed ffmpeg
    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
    'outtmpl': 'temp_audio.%(ext)s'
}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            file_path = ydl.prepare_filename(info_dict).replace('.webm', '.mp3').replace('.m4a', '.mp3')
        return file_path
    except yt_dlp.utils.DownloadError as e:
        print(f"❌ Error downloading audio: {e}")
        return None


# 🔹 Transcribe and Translate Urdu to English
def transcribe_audio(model, audio_file):
    result = model.transcribe(audio_file, task="translate", language="ur")  # Ensure Urdu detection
    return result["text"]

# 🔹 Clean Transcription (Remove Filler Words)
def clean_transcription(text):
    text = re.sub(r'\b(uh|hmm|um|okay|yes|no)\b', '', text, flags=re.IGNORECASE)  # Remove filler words
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# 🔹 Generate News Headlines (Neutral, Left-Wing, Right-Wing)
def generate_headlines(transcription):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    Generate three different news headlines from this speech which are sentences:
    - A neutral headline
    - A left-wing biased headline
    - A right-wing biased headline

    Speech: {transcription}
    """
    response = model.generate_content(prompt)

    if response and response.candidates:
        return response.candidates[0].content.parts[0].text.strip()
    
    return "❌ Error generating headlines"

# 🔹 Extract Key Topics
def extract_keywords(transcription):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Extract the most important key topics from this speech: {transcription}"
    
    try:
        response = model.generate_content(prompt)

        if response and response.candidates:
            return response.candidates[0].content.parts[0].text.strip()

        return "❌ This speech may contain hate speech or sensitive content. Review before proceeding."
    
    except Exception as e:
        return f"❌ Error extracting keywords: {str(e)}"

# 🔹 Generate AI Image Prompt
# def generate_image_prompt(headline):
#     if not headline.strip():
#         return "❌ Error: No neutral headline provided for image generation."

#     model = genai.GenerativeModel('gemini-pro')
#     prompt = f"Describe an image that represents the main action in this headline: {headline}"
#     response = model.generate_content(prompt)

#     if response and response.candidates:
#         return response.candidates[0].content.parts[0].text.strip()

#     return "❌ Error generating image prompt"

# 🔹 Generate AI Image Using FAL AI
def generate_image(prompt):
    result = fal_client.subscribe("fal-ai/flux/dev", arguments={"prompt": prompt})
    return result["images"][0]["url"] if isinstance(result, dict) and "images" in result else None

# 🔹 Streamlit UI Configuration
st.set_page_config(page_title="AI News Generator", layout="wide")
st.title("📰 AI-Generated News Headlines from Political Speeches")
st.markdown("### 🎤 Enter a YouTube Video Link to Generate News Headlines and an AI-Generated Image!")

# 🔹 Input YouTube URL
youtube_url = st.text_input("📺 Enter YouTube Video URL:")

if youtube_url:
    with st.spinner("🔄 Downloading and processing audio..."):
        audio_path = download_audio(youtube_url)
    
    model = load_whisper_model()
    
    with st.spinner("🔄 Transcribing speech (in English)..."):
        transcription = transcribe_audio(model, audio_path)
        transcription = clean_transcription(transcription)
    
    st.success("✅ Transcription Completed!")
    st.text_area("📜 Speech Transcription (English):", transcription, height=150)

    with st.spinner("🔄 Generating headlines..."):
        headlines = generate_headlines(transcription)
    
    st.success("✅ Headlines Generated!")
    st.markdown(f"### 📰 **Generated News Headlines:**\n{headlines}")

    with st.spinner("🔄 Extracting key topics..."):
        keywords = extract_keywords(transcription)
    
    st.success("✅ Key Topics Identified!")
    st.markdown(f"### 🔑 **Key Topics:**\n{keywords}")

    # Extract Neutral Headline for Image Generation
    headline_list = headlines.split("\n")
    neutral_headline = headline_list[0] if len(headline_list) > 0 else ""

    with st.spinner("🔄 Generating image prompt..."):
        image_prompt = neutral_headline
    
    if "❌" in image_prompt:
        st.error(image_prompt)
    else:
        st.success("✅ Image Prompt Generated!")
        st.markdown(f"### 🎨 **AI Image Prompt:**\n{image_prompt}")

        with st.spinner("🔄 Generating AI image..."):
            image_url = generate_image(image_prompt)

        if image_url:
            st.success("✅ Image Successfully Generated!")

            # Load and Display Image
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="🖼️ AI-Generated Image", use_column_width=True)
            else:
                st.error("❌ Error: Unable to fetch generated image.")
        else:
            st.error("❌ Error: AI image generation failed.")

    os.remove(audio_path)  # Cleanup temporary audio file
