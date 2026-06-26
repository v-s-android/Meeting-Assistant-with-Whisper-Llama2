import gradio as gr
from transformers import pipeline

# Function to transcribe audio using the OpenAI Whisper model
def transcribe_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model = "openai/whisper-tiny.en",
        chunk_length_s = 30
    )
    # Transcribe the audio file and return the result
    result = pipe(audio_file , batch_size=8)["text"]
    return result

# Set up Gradio interface
audio_input = gr.Audio(sources = "upload", type = "filepath") # Audio Input
text_output = gr.Textbox() # Text output

# Create the Gradio interface with the function, inputs, and outputs
demo = gr.Interface(
    fn = transcribe_audio,
    inputs = audio_input,
    outputs = text_output,
    title = "Audio Transcription App",
    description = "Upload an Audio file"
)

# Launch the Gradio app
demo.launch(server_name="0.0.0.0", server_port=7860)