# Meeting-Assistant-with-Whisper-Llama2
The Project uses Open Ai's. Whisper and Meta Llama 2 for capturing coversations in Meetings

Consider you're attending a business meeting where all conversations are being captured by an advanced AI application. This application not only transcribes the discussions with high accuracy but also provides a concise summary of the meeting, emphasizing the key points and decisions made.

In our project, we'll use OpenAI's Whisper to transform speech to text. Next, we'll use IBM Watson's AI to summarize and find key points. We'll make an app with Hugging Face Gradio as the user interface.

Preparing the environment and install required libraries:
```
pip3 install virtualenv 
virtualenv my_env # create a virtual environment my_env
source my_env/bin/activate # activate my_env
```
```
# installing required libraries in my_env
pip install --force-reinstall "setuptools<70" transformers==4.36.0 torch==2.1.1 gradio==5.23.2 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.28.1
```
We need to install ffmpeg to be able to work with audio files in python.
```
sudo apt update
```
Then run:
```
sudo apt install ffmpeg -y
```

Procedure

we'll set up a language model (LLM) instance, which could be **IBM WatsonxLLM, HuggingFaceHub, or an OpenAI model**. Then, we'll establish a prompt template. These templates are structured guides to generate prompts for language models, aiding in output organization 

Next, we'll develop a transcription function that employs the OpenAI Whisper model to convert speech-to-text. This function takes an audio file uploaded through a Gradio app interface (preferably in .mp3 format). The transcribed text is then fed into an **LLMChain**, which integrates the text with the prompt template and forwards it to the chosen LLM. The final output from the LLM is then displayed in the Gradio app's output textbox.
