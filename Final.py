import streamlit as st
import whisper
import google.generativeai as genai
import os
import fal_client
import re
from PIL import Image
import requests
from io import BytesIO

FAL_KEY = st.secrets["FAL_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# 🔹 Configure API Keys
os.environ["FAL_KEY"] = FAL_KEY
genai.configure(api_key=GEMINI_API_KEY)

# 🔹 Load Whisper Model (Use Medium for Better Urdu-to-English Translation)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")  # Change "medium" to "large" if needed

# 🔹 Transcribe and Translate Urdu to English
def transcribe_audio(model, audio_file):
    result = model.transcribe(audio_file, task="translate", language="ur") 
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
def generate_image_prompt_gemini(headline):
    
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    Convert the following news headline into a highly detailed visual scene description for AI image generation. 
    Include the setting, action, emotions, and atmosphere.

    Headline: "{headline}"
    
    Image prompt:
    """
    
    response = model.generate_content(prompt)
    
    if response and response.candidates:
        return response.candidates[0].content.parts[0].text.strip()
    
    return "❌ Error generating image prompt"


# 🔹 Generate AI Image Using FAL AI
def generate_image(prompt):
    result = fal_client.subscribe("fal-ai/flux/dev", arguments={"prompt": prompt})
    return result["images"][0]["url"] if isinstance(result, dict) and "images" in result else None

# 🔹 Streamlit UI Configuration
st.set_page_config(page_title="AI News Generator", layout="wide")
st.title("📰 AI-Generated News Headlines from Political Speeches")
st.markdown("### 🎤 Upload an Audio File to Generate News Headlines and an AI-Generated Image!")

# 🔹 File Upload
uploaded_file = st.file_uploader("🎵 Upload an audio file (MP3, WAV, etc.)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    model = load_whisper_model()
    
    with st.spinner("🔄 Transcribing speech (in English)..."):
        transcription = transcribe_audio(model, file_path)
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
        image_prompt = generate_image_prompt_gemini(neutral_headline)
    
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
    
    os.remove(file_path)  # Cleanup temporary audio file
