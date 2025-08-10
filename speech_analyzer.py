import os
import gradio as gr
from pydub import AudioSegment
import tempfile
import math
import google.generativeai as genai

# ----------------------------
# CONFIGURATION
# ----------------------------
# Directly set your Gemini API Key
genai.configure(api_key="AIzaSyDax-WCaWT6WPXr4riX2_6Qz-pGgiuY3RQ")

# ----------------------------
# SPEECH TO TEXT WITH CHUNKING
# ----------------------------
def transcribe_audio(audio_file, chunk_length_min=5):
    """
    Transcribes long audio by splitting into chunks.
    chunk_length_min: length of each chunk in minutes.
    """
    if audio_file is None:
        return "No audio file uploaded.", "No transcription."

    try:
        # Load the audio file
        audio = AudioSegment.from_file(audio_file)

        # Split into chunks
        chunk_length_ms = chunk_length_min * 60 * 1000
        chunks = math.ceil(len(audio) / chunk_length_ms)

        full_transcript = ""

        for i in range(chunks):
            start = i * chunk_length_ms
            end = min((i + 1) * chunk_length_ms, len(audio))
            chunk = audio[start:end]

            # Export chunk to temp WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                chunk.export(temp_wav.name, format="wav")
                temp_path = temp_wav.name

            # Upload to Gemini for transcription
            uploaded_file = genai.upload_file(temp_path)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                [f"Transcribe the following audio accurately:", uploaded_file]
            )
            full_transcript += response.text + "\n"

        return None, full_transcript.strip()

    except Exception as e:
        return str(e), "Error during transcription."

# ----------------------------
# GEMINI CALL (Updated to 1.5 Flash)
# ----------------------------
def call_gemini(prompt_text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"Error calling Gemini: {e}"

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def analyze_meeting(audio_file):
    # Step 1: Transcription (with chunking)
    error, transcription = transcribe_audio(audio_file)
    if error:
        return f"Transcription Error: {error}", ""

    # Step 2: AI Analysis
    analysis_prompt = f"""
    You are a meeting summarizer AI.
    Transcript: {transcription}

    Tasks:
    1. Summarize the meeting in bullet points.
    2. List key decisions made.
    3. Suggest follow-up actions.
    """
    analysis = call_gemini(analysis_prompt)

    return transcription, analysis

# ----------------------------
# GRADIO UI
# ----------------------------
ui = gr.Interface(
    fn=analyze_meeting,
    inputs=gr.Audio(type="filepath", label="Upload Meeting Audio"),
    outputs=[
        gr.Textbox(label="Transcription", lines=10),
        gr.Textbox(label="AI Meeting Analysis", lines=10)
    ],
    title="Meeting Companion - Speech to Insights",
    description="Upload your meeting audio (MP3, WAV, M4A, FLAC) and get an instant transcription & AI analysis."
)

if __name__ == "__main__":
    ui.launch(share=True)
