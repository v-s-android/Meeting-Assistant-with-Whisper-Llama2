import gradio as gr
from transformers import pipeline

def transcribe_audio(audio_file):

    pipe = pipeline(
        "automatic-speech-recognition",
        model = "openai/whisper-tiny.en",
        chunk_length_s = 30
    )

    result = pipe(audio_file , batch_size=8)["text"]
    return result

audio_input = gr.Audio(sources = "upload", type = "filepath")
text_output = gr.Textbox()

demo = gr.Interface(
    fn = transcribe_audio,
    inputs = audio_input,
    outputs = text_output,
    title = "Audio Transcription App",
    description = "Upload an Audio file"
)

demo.launch(server_name="0.0.0.0", server_port=7860)