import torch
import os
import gradio as gr

#from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub

from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

my_credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}

params = {
        GenParams.MAX_NEW_TOKENS: 700, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }

LLAMA2_model = Model(
    model_id='meta-llama/llama-4-maverick-17b-128e-instruct-fp8',
    credentials=my_credentials,
    params=params,
    project_id="skills-network",
)

llm = WatsonxLLM(LLAMA2_model)

#######------------- Prompt Template -------------####

temp = """
<s><<sys>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</sys>>
"""

pt = PromptTemplate(
    input_variables=["context"],
    template=temp
)

# feeding the Prompt template to LLM
prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)

#######------------- Speech2text -------------####
def transcript_audio(audio_file):

    # Initialize the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    # Transcribe the audio file and return the result
    transcript_text = pipe(audio_file, batch_size = 8)["text"]
    result = prompt_to_LLAMA2.run(transcript_text)

    return result


#######------------- Gradio-------------####
audio_input = gr.Audio(sources = "upload", type = "filepath")
output_text = gr.Textbox()

demo = gr.Interface(
    fn = transcript_audio,
    inputs = audio_input,
    outputs = output_text,
    title = "Meeting AI Assitant",
    description = "Upload your file"
)

demo.launch(server_name = "0.0.0.0", server_port = 7860)