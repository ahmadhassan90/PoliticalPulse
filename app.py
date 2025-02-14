import streamlit as st
import whisper
import google.generativeai as genai
import os
from transformers import pipeline

# Set up Whisper model
@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("base")  # You can use "small", "medium", or "large" for better accuracy
    return model

# Set up Gemini API
def configure_gemini():
    # Replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key
    gemini_api_key = "AIzaSyCfeMl9BeG7f5krDjxEyHYSWaqmomV2zt8"
    genai.configure(api_key=gemini_api_key)

# Transcribe audio using Whisper
def transcribe_audio(model, audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

# Generate summary using Gemini
def generate_summary(prompt):
    model = genai.GenerativeModel('gemini-pro')
    try:
        response = model.generate_content(prompt)
        
        # Check if the response was blocked due to safety concerns
        if response.candidates and response.candidates[0].finish_reason == "SAFETY":
            return "⚠️ The content was flagged as inappropriate and cannot be summarized."
        
        # Return the generated text
        return response.text
    except Exception as e:
        return f"❌ An error occurred while generating the summary: {str(e)}"

# Perform sentiment analysis using Hugging Face model
@st.cache_resource
def load_sentiment_model():
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline

def analyze_sentiment(text, sentiment_pipeline):
    # Split the text into chunks of 500 tokens (to avoid exceeding the model's token limit)
    max_length = 500
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    # Analyze sentiment for each chunk
    results = []
    for chunk in chunks:
        result = sentiment_pipeline(chunk)
        results.append(result[0])
    
    # Aggregate results (e.g., average confidence and majority label)
    positive_count = sum(1 for r in results if r["label"] == "POSITIVE")
    negative_count = sum(1 for r in results if r["label"] == "NEGATIVE")
    
    # Determine the majority label
    majority_label = "POSITIVE" if positive_count > negative_count else "NEGATIVE"
    
    # Calculate average confidence
    average_confidence = sum(r["score"] for r in results) / len(results)
    
    return {"label": majority_label, "score": average_confidence}

# Streamlit app
def main():
    st.title("Audio Summary and Sentiment Analysis App")
    st.write("Upload an audio file, and I'll transcribe it, summarize it using Gemini, and analyze its sentiment!")

    # Upload audio file
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])

    if audio_file is not None:
        # Save the uploaded file temporarily
        file_extension = audio_file.name.split(".")[-1]
        temp_file_path = f"temp_audio.{file_extension}"
        with open(temp_file_path, "wb") as f:
            f.write(audio_file.getbuffer())

        # Load Whisper model
        model = load_whisper_model()

        # Transcribe audio
        st.write("Transcribing audio...")
        transcription = transcribe_audio(model, temp_file_path)
        st.write("**Transcription:**")
        st.write(transcription)

        # Configure Gemini (API key is hardcoded)
        configure_gemini()

        # Generate summary
        st.write("Generating summary...")
        prompt = f"Summarize the following text: {transcription}"
        summary = generate_summary(prompt)
        st.write("**Summary:**")
        st.write(summary)

        # Perform sentiment analysis
        st.write("Analyzing sentiment...")
        sentiment_pipeline = load_sentiment_model()
        sentiment_result = analyze_sentiment(transcription, sentiment_pipeline)
        st.write("**Sentiment Analysis Result:**")
        st.write(f"Label: {sentiment_result['label']}, Confidence: {sentiment_result['score']:.2f}")

        # Clean up temporary file
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()